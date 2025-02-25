# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Tuple, Type
from segment_anything.modeling.transformer import TwoWayTransformer


# Assume LayerNorm2d is defined as follows
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.norm = nn.GroupNorm(1, num_channels, eps=eps)

    def forward(self, x):
        return self.norm(x)


class FeatureEnhancement(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention
        ca = self.channel_attention(x)
        x = x * ca

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))

        return x * sa


# Cross-Attention Fusion module
class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        """
        Use cross-attention mechanism to fuse two inputs:
          - query from the upsampled image embedding
          - key/value from the guide matrix (after interpolation to match the spatial resolution of the image embedding)
        A residual connection is used to ensure that excessive noise is not introduced.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = d_model // num_heads
        self.scale = head_dim ** -0.5

        # Add input normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Use more efficient depthwise separable convolutions
        self.q_conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, 1),
            nn.Conv2d(d_model, d_model, 3, padding=1, groups=d_model)
        )
        self.k_conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, 1),
            nn.Conv2d(d_model, d_model, 3, padding=1, groups=d_model)
        )
        self.v_conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, 1),
            nn.Conv2d(d_model, d_model, 3, padding=1, groups=d_model)
        )

        self.out_conv = nn.Conv2d(d_model, d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, image_embedding: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """
        image_embedding: (B, d_model, H, W) used as the query
        guide: (B, d_model, H, W) used as the key and value
        Returns the cross-attention output, which is added with the original image_embedding via a residual connection and used for subsequent mask prediction.
        """
        residual = image_embedding

        # Spatial adaptive normalization
        B, C, H, W = image_embedding.shape
        image_embedding = image_embedding.permute(0, 2, 3, 1)
        guide = guide.permute(0, 2, 3, 1)
        image_embedding = self.norm1(image_embedding)
        guide = self.norm2(guide)
        image_embedding = image_embedding.permute(0, 3, 1, 2)
        guide = guide.permute(0, 3, 1, 2)

        q = self.q_conv(image_embedding)
        k = self.k_conv(guide)
        v = self.v_conv(guide)

        head_dim = C // self.num_heads
        q = q.view(B, self.num_heads, head_dim, H * W).transpose(2, 3)
        k = k.view(B, self.num_heads, head_dim, H * W).transpose(2, 3)
        v = v.view(B, self.num_heads, head_dim, H * W).transpose(2, 3)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_probs = self.dropout(torch.softmax(attn_scores, dim=-1))

        out = torch.matmul(attn_probs, v)
        out = out.transpose(2, 3).reshape(B, C, H, W)
        out = self.out_conv(out)

        # Residual connection
        return residual + self.dropout(out)


class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int,
            sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x


class MaskDecoder(nn.Module):
    def __init__(
            self,
            *,
            transformer_dim: int,
            transformer: nn.Module,
            num_multimask_outputs: int = 9,
            activation: Type[nn.Module] = nn.GELU,
            iou_head_depth: int = 3,
            iou_head_hidden_dim: int = 256,
            num_refinement_iterations: int = 1,
    ) -> None:
        """
        MaskDecoder:
         - Fuses guide matrix information
         - Multi-scale upsampling path
         - Dynamic mask generation and iterative refinement

        Arguments:
          transformer_dim (int): the number of channels for the transformer
          transformer (nn.Module): the transformer module used to predict masks
          num_multimask_outputs (int): the number of masks predicted when outputting multiple masks
          activation (nn.Module): the activation function used during upsampling
          iou_head_depth (int): the number of MLP layers used for predicting mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP used for predicting mask quality
          num_refinement_iterations (int): the number of iterations for mask refinement
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs
        self.num_refinement_iterations = num_refinement_iterations

        # Token used for IoU prediction
        self.iou_token = nn.Embedding(1, transformer_dim)
        # Number of mask tokens (including one main mask and several branch masks)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        # Modified multi-scale upsampling path (input resolution 14x14, output resolution 56x56)
        self.up1 = nn.ConvTranspose2d(transformer_dim, transformer_dim // 2, kernel_size=2, stride=2)  # 14 -> 28
        self.ln1 = LayerNorm2d(transformer_dim // 2)
        self.act1 = activation()

        self.up2 = nn.ConvTranspose2d(transformer_dim // 2, transformer_dim // 4, kernel_size=2, stride=2)  # 28 -> 56
        self.ln2 = LayerNorm2d(transformer_dim // 4)
        self.act2 = activation()

        self.final_conv = nn.Conv2d(transformer_dim // 4, transformer_dim // 8, kernel_size=1)

        # Fuse guide matrix information (assume the guide matrix has transformer_dim channels)
        self.guide_fusion_conv = nn.Conv2d(transformer_dim, transformer_dim // 8, kernel_size=1)

        # Hypernetwork: generate dynamic convolution weights based on each mask token to produce the initial mask
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for _ in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

        # Mask refinement module: iteratively refine each mask
        # The input is the concatenation of the current mask (1 channel) and the upsampled features (transformer_dim//8 channels)
        self.mask_refiner = nn.Sequential(
            nn.Conv2d(1 + transformer_dim // 8, transformer_dim // 8, kernel_size=3, padding=1),
            activation(),
            nn.Conv2d(transformer_dim // 8, 1, kernel_size=1),
        )

    def forward(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            guide_matrix: torch.Tensor = None,
            multimask_output: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Arguments:
          image_embeddings (B, C, H, W): embeddings output by the image encoder
          image_pe (B, C, H, W): corresponding positional encoding
          sparse_prompt_embeddings (B, N_prompt, C): embeddings of sparse prompts (e.g., points, boxes)
          dense_prompt_embeddings (B, C, H, W): embeddings of dense prompts (e.g., coarse masks)
          guide_matrix (B, C, H, W): guide matrix information (optional)
          multimask_output (bool): whether to output multiple branch masks

        Returns:
          masks (B, num_masks, H_out, W_out)
          iou_pred (B, num_masks)
        """
        # print("Use new decoder")

        masks, iou_pred = self.predict_masks(
            image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, guide_matrix
        )

        # Depending on multimask_output, choose to return a single or multiple masks
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        return masks, iou_pred

    def predict_masks(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            guide_matrix: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict masks and improve results using upsampling, dynamic generation, and iterative refinement."""
        B = sparse_prompt_embeddings.size(0)
        # Concatenate output tokens (including IoU token and mask tokens)
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight],
                                  dim=0)  # (num_mask_tokens, transformer_dim)
        output_tokens = output_tokens.unsqueeze(0).expand(B, -1, -1)  # (B, num_mask_tokens, transformer_dim)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings),
                           dim=1)  # (B, num_mask_tokens + N_prompt, transformer_dim)

        # Add dense prompt to image embeddings (other fusion strategies can also be used)
        src = image_embeddings + dense_prompt_embeddings
        pos_src = image_pe

        # Run the transformer to obtain the interacted token representations and updated image features
        hs, transformer_output, attn_out = self.transformer(src, pos_src, tokens)
        B = transformer_output.shape[0]
        # First convert transformer_output from (B,196,256) to (B,256,196)
        transformer_output = transformer_output.permute(0, 2, 1)
        # Then reshape to (B,256,14,14)
        transformer_output = transformer_output.view(B, self.transformer_dim, 14, 14)

        # print('transformer_output',transformer_output.shape)
        # print('hs', hs.shape)

        iou_token_out = hs[:, 0, :]  # (B, transformer_dim)
        mask_tokens_out = hs[:, 1: (1 + self.num_mask_tokens), :]  # (B, num_mask_tokens, transformer_dim)

        # Multi-scale upsampling: restore transformer output features to a higher resolution
        x = self.act1(self.ln1(self.up1(transformer_output)))
        x = self.act2(self.ln2(self.up2(x)))
        upscaled_embedding = self.final_conv(x)  # (B, transformer_dim//8, H_out, W_out)

        # If a guide matrix is provided, resize and fuse it first
        if guide_matrix is not None:
            guide_resized = F.interpolate(guide_matrix, size=upscaled_embedding.shape[-2:], mode='bilinear',
                                          align_corners=False)
            guide_features = self.guide_fusion_conv(guide_resized)
            upscaled_embedding = upscaled_embedding + guide_features

        # Use hypernetworks to generate dynamic convolution weights based on mask tokens and produce the initial mask
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)  # (B, num_mask_tokens, transformer_dim//8)

        b, c, h, w = upscaled_embedding.shape  # c = transformer_dim//8
        up_flat = upscaled_embedding.view(b, c, h * w)  # (B, c, H*W)
        # Dynamic convolution: each mask token computes inner product with upsampled features to obtain the initial mask
        masks = torch.matmul(hyper_in, up_flat).view(b, self.num_mask_tokens, h, w)

        # Iterative refinement: refine the mask using the current mask and the upsampled features
        for _ in range(self.num_refinement_iterations):
            delta_list = []
            for i in range(self.num_mask_tokens):
                mask_i = masks[:, i: i + 1, :, :]  # (B, 1, h, w)
                refine_input = torch.cat([mask_i, upscaled_embedding], dim=1)  # (B, 1 + c, h, w)
                delta = self.mask_refiner(refine_input)  # (B, 1, h, w)
                delta_list.append(delta)
            delta = torch.cat(delta_list, dim=1)  # (B, num_mask_tokens, h, w)
            masks = masks + delta

        # Generate IoU predictions
        iou_pred = self.iou_prediction_head(iou_token_out)  # (B, num_mask_tokens)

        return masks, iou_pred


# ----------------- Below is test code -----------------
if __name__ == "__main__":
    # Construct dummy inputs (batch size 2, to observe multiple mask outputs)
    B = 2

    image_emb = torch.randn(B, 256, 14, 14)  # image embeddings
    image_pe = torch.randn(1, 256, 14, 14)  # positional encoding

    guide_mat = torch.randn(B, 256, 14, 14)  # guide matrix

    sparse_prompt = torch.randn(1, 0, 256)
    dense_prompt = torch.randn(1, 256, 14, 14)

    # Construct MaskDecoder with a dummy transformer
    twoway = TwoWayTransformer(
        depth=2,
        embedding_dim=256,
        mlp_dim=2048,
        num_heads=8,
    )

    # New version
    num_classes = 8 + 1  # including background
    prompt_embed_dim = 256
    pixel_mean = [0, 0, 0]
    pixel_std = [1, 1, 1]

    decoder = MaskDecoder(
        num_multimask_outputs=num_classes,
        transformer=twoway,
        transformer_dim=prompt_embed_dim,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
    )

    masks, iou_pred = decoder(
        image_embeddings=image_emb,
        image_pe=image_pe,
        guide_matrix=guide_mat,
        sparse_prompt_embeddings=sparse_prompt,
        dense_prompt_embeddings=dense_prompt,
        multimask_output=True
    )

    print(f"Output masks shape: {masks.shape}")  # For example, output shape (B, num_masks, 224, 224)
    print(f"IOU predictions shape: {iou_pred.shape}")  # For example, output shape (B, num_masks)
