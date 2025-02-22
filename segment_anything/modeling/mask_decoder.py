# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Tuple, Type
from segment_anything.modeling.transformer import TwoWayTransformer

# 假设 LayerNorm2d 定义如下
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
        # 通道注意力
        ca = self.channel_attention(x)
        x = x * ca

        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))

        return x * sa

# Cross-Attention 融合模块
class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        """
        采用交叉注意力机制融合两个输入：
          - query 来自上采样后的 image embedding
          - key/value 来自 guide matrix（经过插值调整尺寸后与 image embedding 空间一致）
        通过残差连接保证不会引入过多噪声。
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = d_model // num_heads
        self.scale = head_dim ** -0.5
        
        # 添加输入归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 使用更高效的深度可分离卷积
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
        image_embedding: (B, d_model, H, W) 作为 query
        guide: (B, d_model, H, W) 作为 key 与 value
        返回交叉注意力输出，与原始 image_embedding 残差相加后用于后续掩模预测
        """
        residual = image_embedding
        
        # 空间自适应归一化
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
        
        # 残差连接
        return residual + self.dropout(out)

class MaskDecoder_origin(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 8,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        tranformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Prepare output
        return masks, iou_pred

    def predict_masks(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src, _ = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1: (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred



# 优化后的 MaskDecoder 模块
class MaskDecoderaa(nn.Module):
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
        MaskDecoder：
         - 融合 guide matrix 信息
         - 多尺度上采样路径
         - 动态掩模生成及迭代 Refinement

        Arguments:
          transformer_dim (int): transformer 的通道数
          transformer (nn.Module): 用于预测掩模的 transformer 模块
          num_multimask_outputs (int): 多掩模输出时预测的掩模数量
          activation (nn.Module): 上采样时使用的激活函数
          iou_head_depth (int): 用于预测掩模质量的 MLP 层数
          iou_head_hidden_dim (int): 用于预测掩模质量的 MLP 隐藏层维度
          num_refinement_iterations (int): 掩模 Refinement 的迭代次数
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs
        self.num_refinement_iterations = num_refinement_iterations

        # 用于 IoU 预测的 token
        self.iou_token = nn.Embedding(1, transformer_dim)
        # 掩模 token 数（包括一个主掩模和多分支掩模）
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        # 修改后的多尺度上采样路径（输入分辨率 14x14，输出分辨率 56x56）
        self.up1 = nn.ConvTranspose2d(transformer_dim, transformer_dim // 2, kernel_size=2, stride=2)  # 14 -> 28
        self.ln1 = LayerNorm2d(transformer_dim // 2)
        self.act1 = activation()

        self.up2 = nn.ConvTranspose2d(transformer_dim // 2, transformer_dim // 4, kernel_size=2, stride=2)  # 28 -> 56
        self.ln2 = LayerNorm2d(transformer_dim // 4)
        self.act2 = activation()

        self.final_conv = nn.Conv2d(transformer_dim // 4, transformer_dim // 8, kernel_size=1)

        # 融合 guide matrix 信息（假设 guide matrix 的通道数为 transformer_dim）
        self.guide_fusion_conv = nn.Conv2d(transformer_dim, transformer_dim // 8, kernel_size=1)

        # Hypernetwork：根据每个 mask token 生成动态卷积的权重，用于生成初始掩模
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for _ in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

        # 掩模 Refinement 模块：对每个掩模进行迭代细化
        # 输入为当前掩模（1通道）与上采样特征（transformer_dim//8 通道）的拼接
        self.mask_refiner = nn.Sequential(
            nn.Conv2d(1 + transformer_dim // 8, transformer_dim // 4, 3, padding=1),
            nn.BatchNorm2d(transformer_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(transformer_dim // 4, transformer_dim // 8, 3, padding=1),
            nn.BatchNorm2d(transformer_dim // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(transformer_dim // 8, 1, 1)
        )

        # 交叉注意力融合模块，将 guide matrix 信息与上采样特征融合
        self.cross_attention_fusion = CrossAttentionFusion(d_model=transformer_dim // 8, num_heads=4)

        # 特征增强模块
        self.feature_enhancement = FeatureEnhancement(transformer_dim)

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
          image_embeddings (B, C, H, W): 图像编码器输出的嵌入
          image_pe (B, C, H, W): 对应的位置编码
          sparse_prompt_embeddings (B, N_prompt, C): 稀疏提示的嵌入（例如点、框）
          dense_prompt_embeddings (B, C, H, W): 密集提示（例如粗掩模）的嵌入
          guide_matrix (B, C, H, W): 引导矩阵信息（可选）
          multimask_output (bool): 是否输出多分支掩模

        Returns:
          masks (B, num_masks, H_out, W_out)
          iou_pred (B, num_masks)
        """
        # print("Use new decoder no dropout")

        masks, iou_pred = self.predict_masks(
            image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, guide_matrix
        )

        # 根据 multimask_output 选择返回单一或多个掩模
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
        """预测掩模，并利用上采样、动态生成和迭代 Refinement 提升效果。"""
        B = sparse_prompt_embeddings.size(0)

        # Token 准备
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(B, -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # 特征融合
        src = image_embeddings + dense_prompt_embeddings

        # Transformer 处理
        hs, transformer_output, _ = self.transformer(src, image_pe, tokens)
        B = transformer_output.shape[0]
        # 先将 transformer_output 从 (B,196,256) 转换为 (B,256,196)
        transformer_output = transformer_output.permute(0, 2, 1)
        # 然后重塑为 (B,256,14,14)
        transformer_output = transformer_output.view(B, self.transformer_dim, 14, 14)

        # 提取 token 输出
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1:(1 + self.num_mask_tokens), :]

        # 特征增强
        enhanced_features = self.feature_enhancement(transformer_output)

        # 多尺度上采样
        x = self.act1(self.ln1(self.up1(enhanced_features)))
        x = self.act2(self.ln2(self.up2(x)))
        upscaled_embedding = self.final_conv(x)

        # Guide matrix 融合
        if guide_matrix is not None:
            guide_resized = F.interpolate(
                guide_matrix,
                size=upscaled_embedding.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            guide_features = self.guide_fusion_conv(guide_resized)
            fused_features = self.cross_attention_fusion(upscaled_embedding, guide_features)
            upscaled_embedding = upscaled_embedding + 0.1 * fused_features  # 添加缩放因子

        # 动态掩模生成
        hyper_in_list = [
            self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            for i in range(self.num_mask_tokens)
        ]
        hyper_in = torch.stack(hyper_in_list, dim=1)

        b, c, h, w = upscaled_embedding.shape
        masks = torch.matmul(hyper_in, upscaled_embedding.view(b, c, h * w)).view(b, self.num_mask_tokens, h, w)

        # 迭代细化
        for _ in range(self.num_refinement_iterations):
            delta = torch.cat([
                self.mask_refiner(torch.cat([masks[:, i:i+1], upscaled_embedding], dim=1))
                for i in range(self.num_mask_tokens)
            ], dim=1)
            masks = masks + 0.1 * delta  # 添加缩放因子控制更新步长

        # IoU 预测
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred






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
        MaskDecoder：
         - 融合 guide matrix 信息
         - 多尺度上采样路径
         - 动态掩模生成及迭代 Refinement

        Arguments:
          transformer_dim (int): transformer 的通道数
          transformer (nn.Module): 用于预测掩模的 transformer 模块
          num_multimask_outputs (int): 多掩模输出时预测的掩模数量
          activation (nn.Module): 上采样时使用的激活函数
          iou_head_depth (int): 用于预测掩模质量的 MLP 层数
          iou_head_hidden_dim (int): 用于预测掩模质量的 MLP 隐藏层维度
          num_refinement_iterations (int): 掩模 Refinement 的迭代次数
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs
        self.num_refinement_iterations = num_refinement_iterations

        # 用于 IoU 预测的 token
        self.iou_token = nn.Embedding(1, transformer_dim)
        # 掩模 token 数（包括一个主掩模和多分支掩模）
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        # 修改后的多尺度上采样路径（输入分辨率 14x14，输出分辨率 56x56）
        self.up1 = nn.ConvTranspose2d(transformer_dim, transformer_dim // 2, kernel_size=2, stride=2)  # 14 -> 28
        self.ln1 = LayerNorm2d(transformer_dim // 2)
        self.act1 = activation()

        self.up2 = nn.ConvTranspose2d(transformer_dim // 2, transformer_dim // 4, kernel_size=2, stride=2)  # 28 -> 56
        self.ln2 = LayerNorm2d(transformer_dim // 4)
        self.act2 = activation()

        self.final_conv = nn.Conv2d(transformer_dim // 4, transformer_dim // 8, kernel_size=1)

        # 融合 guide matrix 信息（假设 guide matrix 的通道数为 transformer_dim）
        self.guide_fusion_conv = nn.Conv2d(transformer_dim, transformer_dim // 8, kernel_size=1)

        # Hypernetwork：根据每个 mask token 生成动态卷积的权重，用于生成初始掩模
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for _ in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

        # 掩模 Refinement 模块：对每个掩模进行迭代细化
        # 输入为当前掩模（1通道）与上采样特征（transformer_dim//8 通道）的拼接
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
          image_embeddings (B, C, H, W): 图像编码器输出的嵌入
          image_pe (B, C, H, W): 对应的位置编码
          sparse_prompt_embeddings (B, N_prompt, C): 稀疏提示的嵌入（例如点、框）
          dense_prompt_embeddings (B, C, H, W): 密集提示（例如粗掩模）的嵌入
          guide_matrix (B, C, H, W): 引导矩阵信息（可选）
          multimask_output (bool): 是否输出多分支掩模

        Returns:
          masks (B, num_masks, H_out, W_out)
          iou_pred (B, num_masks)
        """
        # print("Use new decoder")

        masks, iou_pred = self.predict_masks(
            image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, guide_matrix
        )

        # 根据 multimask_output 选择返回单一或多个掩模
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
        """预测掩模，并利用上采样、动态生成和迭代 Refinement 提升效果。"""
        B = sparse_prompt_embeddings.size(0)
        # 拼接输出 token（包括 IoU token 与 mask tokens）
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight],
                                  dim=0)  # (num_mask_tokens, transformer_dim)
        output_tokens = output_tokens.unsqueeze(0).expand(B, -1, -1)  # (B, num_mask_tokens, transformer_dim)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings),
                           dim=1)  # (B, num_mask_tokens + N_prompt, transformer_dim)

        # 将 dense prompt 与图像嵌入相加（也可以采用其他融合策略）
        src = image_embeddings + dense_prompt_embeddings
        pos_src = image_pe

        # 运行 transformer，得到交互后的 token 表示与更新后的图像特征
        hs, transformer_output, attn_out = self.transformer(src, pos_src, tokens)
        B = transformer_output.shape[0]
        # 先将 transformer_output 从 (B,196,256) 转换为 (B,256,196)
        transformer_output = transformer_output.permute(0, 2, 1)
        # 然后重塑为 (B,256,14,14)
        transformer_output = transformer_output.view(B, self.transformer_dim, 14, 14)

        # print('transformer_output',transformer_output.shape)
        # print('hs', hs.shape)

        iou_token_out = hs[:, 0, :]  # (B, transformer_dim)
        mask_tokens_out = hs[:, 1: (1 + self.num_mask_tokens), :]  # (B, num_mask_tokens, transformer_dim)

        # 多尺度上采样，将 transformer 输出的特征还原到更高分辨率
        x = self.act1(self.ln1(self.up1(transformer_output)))
        x = self.act2(self.ln2(self.up2(x)))
        upscaled_embedding = self.final_conv(x)  # (B, transformer_dim//8, H_out, W_out)

        # 如果提供了 guide matrix，则先调整分辨率再融合
        if guide_matrix is not None:
            guide_resized = F.interpolate(guide_matrix, size=upscaled_embedding.shape[-2:], mode='bilinear',
                                          align_corners=False)
            guide_features = self.guide_fusion_conv(guide_resized)
            upscaled_embedding = upscaled_embedding + guide_features

        # 利用 hypernetwork 根据 mask token 生成动态卷积权重，初步预测掩模
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)  # (B, num_mask_tokens, transformer_dim//8)

        b, c, h, w = upscaled_embedding.shape  # c = transformer_dim//8
        up_flat = upscaled_embedding.view(b, c, h * w)  # (B, c, H*W)
        # 动态卷积：每个 mask token 与上采样特征做内积得到初始掩模
        masks = torch.matmul(hyper_in, up_flat).view(b, self.num_mask_tokens, h, w)

        # 迭代 Refinement：利用当前掩模与上采样特征进行细化
        for _ in range(self.num_refinement_iterations):
            delta_list = []
            for i in range(self.num_mask_tokens):
                mask_i = masks[:, i: i + 1, :, :]  # (B, 1, h, w)
                refine_input = torch.cat([mask_i, upscaled_embedding], dim=1)  # (B, 1 + c, h, w)
                delta = self.mask_refiner(refine_input)  # (B, 1, h, w)
                delta_list.append(delta)
            delta = torch.cat(delta_list, dim=1)  # (B, num_mask_tokens, h, w)
            masks = masks + delta

        # 生成 IoU 预测
        iou_pred = self.iou_prediction_head(iou_token_out)  # (B, num_mask_tokens)

        return masks, iou_pred


# ----------------- 以下为测试代码 -----------------
if __name__ == "__main__":
    # 构造 dummy 输入（batch 大小 2，方便观察多掩模输出）
    B = 2

    image_emb = torch.randn(B, 256, 14, 14)  # 图像嵌入
    image_pe = torch.randn(1, 256, 14, 14)  # 位置编码

    guide_mat = torch.randn(B, 256, 14, 14)  # 引导矩阵

    sparse_prompt = torch.randn(1, 0, 256)
    dense_prompt = torch.randn(1, 256, 14, 14)

    # 构造 MaskDecoder，传入 dummy transformer
    twoway = TwoWayTransformer(
        depth=2,
        embedding_dim=256,
        mlp_dim=2048,
        num_heads=8,
    )

    # 原版
    # decoder = MaskDecoder_origin(transformer_dim=256, transformer=twoway)
    #
    # masks, iou_pred = decoder(
    #     image_embeddings=image_emb,
    #     image_pe=image_pe,
    #     sparse_prompt_embeddings=sparse_prompt,
    #     dense_prompt_embeddings=dense_prompt,
    #     multimask_output=True
    # )

    # 新版
    num_classes = 8 + 1 # 背景
    prompt_embed_dim = 256
    pixel_mean = [0, 0, 0]
    pixel_std = [1, 1, 1]

    decoder = MaskDecoder(
        num_multimask_outputs= num_classes,
        transformer= twoway,
        transformer_dim= prompt_embed_dim,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
    )


    masks, iou_pred = decoder(
        image_embeddings=image_emb,
        image_pe=image_pe,
        guide_matrix = guide_mat,
        sparse_prompt_embeddings=sparse_prompt,
        dense_prompt_embeddings=dense_prompt,
        multimask_output=True
    )

    print(f"Output masks shape: {masks.shape}")  # 例如输出形状 (B, num_masks, 224, 224)
    print(f"IOU predictions shape: {iou_pred.shape}")  # 例如输出形状 (B, num_masks)


