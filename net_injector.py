import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from Lora import LoRA_Sam
from segment_anything import sam_model_registry
from torch.nn.parameter import Parameter
from transformers import AutoTokenizer
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


############################################################
# 1. Define the wrapped LoRA Multi-Head Attention module (_LoRA_MHA)
############################################################
class _LoRA_MHA(nn.Module):
    def __init__(self, mha: nn.MultiheadAttention, r: int = 4):
        super().__init__()
        self.mha = mha  # The original MultiheadAttention
        self.embed_dim = mha.embed_dim
        self.num_heads = mha.num_heads
        self.r = r
        self.dropout = mha.dropout

        # Freeze the original MHA parameters
        for param in self.mha.parameters():
            param.requires_grad = False

        # Construct LoRA adapters: for Q and V branches
        self.linear_a_q = nn.Linear(self.embed_dim, r, bias=False)
        self.linear_b_q = nn.Linear(r, self.embed_dim, bias=False)
        self.linear_a_v = nn.Linear(self.embed_dim, r, bias=False)
        self.linear_b_v = nn.Linear(r, self.embed_dim, bias=False)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        # Extract original in_proj_weight/in_proj_bias for computing Q/K/V
        in_proj_weight = self.mha.in_proj_weight
        in_proj_bias = self.mha.in_proj_bias

        q_weight = in_proj_weight[:self.embed_dim, :]
        k_weight = in_proj_weight[self.embed_dim:2*self.embed_dim, :]
        v_weight = in_proj_weight[2*self.embed_dim:3*self.embed_dim, :]

        if in_proj_bias is not None:
            q_bias = in_proj_bias[:self.embed_dim]
            k_bias = in_proj_bias[self.embed_dim:2*self.embed_dim]
            v_bias = in_proj_bias[2*self.embed_dim:3*self.embed_dim]
        else:
            q_bias = k_bias = v_bias = None

        # Compute the original Q, K, V (assuming input shape [L, B, embed_dim])
        Q_orig = F.linear(query, q_weight, q_bias)
        K_orig = F.linear(key, k_weight, k_bias)
        V_orig = F.linear(value, v_weight, v_bias)

        # For computing the LoRA update, cast input to float (full precision), then cast back to input dtype (e.g., fp16)
        Q_lora = self.linear_b_q(self.linear_a_q(query.float())).to(query.dtype)
        V_lora = self.linear_b_v(self.linear_a_v(value.float())).to(value.dtype)

        # Obtain updated Q and V (K remains unchanged)
        Q = Q_orig + Q_lora
        V = V_orig + V_lora

        # Call the built-in multi-head attention function. Note that in_proj_weight/in_proj_bias are not used here.
        attn_output, attn_output_weights = F.multi_head_attention_forward(
            query=Q,
            key=K_orig,
            value=V,
            embed_dim_to_check=self.embed_dim,
            num_heads=self.num_heads,
            in_proj_weight=None,
            in_proj_bias=None,
            bias_k=self.mha.bias_k,
            bias_v=self.mha.bias_v,
            add_zero_attn=self.mha.add_zero_attn,
            dropout_p=self.dropout,
            out_proj_weight=self.mha.out_proj.weight,
            out_proj_bias=self.mha.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=True,
            q_proj_weight=torch.eye(self.embed_dim, device=query.device, dtype=query.dtype),
            k_proj_weight=torch.eye(self.embed_dim, device=key.device, dtype=key.dtype),
            v_proj_weight=torch.eye(self.embed_dim, device=value.device, dtype=value.dtype)
        )

        return attn_output



############################################################
# 2. Modify the LoRA_CLIP module to inject LoRA into the attention layers of the CLIP text encoder
############################################################

class LoRA_CLIP(nn.Module):
    """
    Apply LoRA adapters to the text encoder of the CLIP model.
    Here, we iterate over each transformer block's attention layer,
    and if the attention layer is an instance of nn.MultiheadAttention, we replace it with the _LoRA_MHA wrapper.
    """

    def __init__(self, clip_model, r: int = 4, lora_layers=None):
        super().__init__()
        self.clip_model = clip_model
        self.r = r

        # Freeze all parameters of the CLIP model;
        # if you only want to train the LoRA parameters, make sure the LoRA parts remain requires_grad=True.
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # According to the CLIP implementation you use, assume that the text transformer is stored in clip_model.transformer.resblocks
        if lora_layers is None:
            self.lora_layers = list(range(len(self.clip_model.transformer.resblocks)))
        else:
            self.lora_layers = lora_layers

        # print(self.clip_model.transformer.resblocks)

        self._init_lora_adapters()

    def _init_lora_adapters(self):
        r = self.r
        for i, block in enumerate(self.clip_model.transformer.resblocks):
            if i not in self.lora_layers:
                continue
            if isinstance(block.attn, nn.MultiheadAttention):
                block.attn = _LoRA_MHA(block.attn, r=r)
            else:
                # If your CLIP version has c_attn inside attn, you can adopt the previous method (see previous code example)
                try:
                    c_attn = block.attn.c_attn
                except AttributeError:
                    raise AttributeError("Cannot find attn.c_attn and it is not an instance of nn.MultiheadAttention. Please check the CLIP model implementation.")
                pass

        # for i, block in enumerate(self.clip_model.transformer.resblocks):
        #     if isinstance(block.attn, _LoRA_MHA):
        #         for name, param in block.attn.named_parameters():
        #             # If the name contains "linear_a" or "linear_b", set it as trainable
        #             if "linear_a" in name or "linear_b" in name:
        #                 param.requires_grad = True
        #                 print(f"Unfroze LoRA param in block {i}: {name}")

    def forward(self, *args, **kwargs):
        return self.clip_model(*args, **kwargs)







class GuideMatrixGenerator(nn.Module):
    def __init__(self, sam, lora_clip):
        """
        Class for generating the guide matrix.

        Parameters:
            sam: the original SAM model instance
            lora_clip: the CLIP model instance with LoRA
        """
        super().__init__()
        self.sam_model = sam
        self.lora_clip = lora_clip

        self.text_proj = nn.Linear(512, 256)

        self.sim_proj = nn.Linear(512, 196)

        self.tokenizer = AutoTokenizer.from_pretrained(
            "./dual-sam/bert-base-uncased"
        )

        self.layer_norm = torch.nn.LayerNorm(196)

    def get_enhanced_text_features(self, text_batch, device):
        """Enhanced text feature extraction"""
        if text_batch is not None:
            tokenized = self.tokenizer(
                text_batch,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt"
            ).to(device)
            return self.lora_clip.clip_model.encode_text(tokenized.input_ids)
        else:
            # Use preset prompt enhancement
            prompt_embeds = self.load_preset_prompts(device)
            return prompt_embeds.mean(dim=0, keepdim=True)

    def dynamic_similarity(self, img_feat, txt_feat):
        """Dynamic similarity calculation module"""
        # Multi-modal interaction
        cross_attn = torch.einsum('bc,bc->b', img_feat, txt_feat)  # (B,)
        # Change output dimension to (B, 196)
        spatial_weights = torch.sigmoid(self.sim_proj(img_feat.float()))  # (B, 196)
        # Multiply cross attention and spatial weights, with automatic broadcasting (B, 1) * (B, 196) -> (B, 196)
        return (cross_attn.unsqueeze(1) * spatial_weights) / 10.0  # Control the numerical range

    def hierarchical_norm(self, x):
        """Hierarchical normalization structure"""
        # Channel-level normalization
        x = F.layer_norm(x.permute(0, 2, 3, 1), [x.size(1)]).permute(0, 3, 1, 2)
        # Spatial-level normalization
        spatial_mean = x.mean(dim=[2, 3], keepdim=True)
        spatial_std = x.std(dim=[2, 3], keepdim=True)
        return (x - spatial_mean) / (spatial_std + 1e-6)

    def forward(self, image, text_batch):
        # 1. Feature extraction and text encoding
        image_features = self.lora_clip.clip_model.encode_image(image)
        sam_features, _ = self.sam_model.image_encoder(image)  # (B, C, H, W)
        device = image.device

        B, C, H, W = sam_features.shape
        L = H * W  # Total number of spatial locations, ideally L == 196 = 14 * 14

        # Retrieve text features (optimized implementation)
        text_features = self.get_enhanced_text_features(text_batch, device)
        # 2. Multi-modal feature interaction (new module)
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)

        # Dynamic similarity weighting (improved implementation)
        similarity_weights = self.dynamic_similarity(image_features, text_features)  # (B, 196)

        # 3. Compute spatial attention matrix (matrix optimization)
        sam_flat = sam_features.view(B, C, L)
        sam_norm = F.normalize(sam_flat, p=2, dim=1)

        # Replace loop with batch matrix multiplication
        spatial_attn = torch.einsum('bcl,bcm->blm', sam_norm, sam_norm) / math.sqrt(C)
        spatial_attn = F.softmax(spatial_attn, dim=-1)  # (B, L, L)

        # 4. Feature aggregation (vectorized implementation)
        # Weighted feature computation
        weighted_feats = torch.einsum('bcl,blm->bcm', sam_flat, spatial_attn)  # (B, C, L)
        # Local feature enhancement
        local_feats = sam_flat  # (B, C, L)
        combined_feats = (local_feats + weighted_feats) * 0.5  # Average fusion

        # 5. Generate multi-modal guide matrix
        # Use unsqueeze on the channel dimension to insert 1, so that similarity_weights (B, 196) becomes (B, 1, 196)
        guide_matrix = combined_feats * similarity_weights.unsqueeze(1)  # (B, C, L)
        guide_matrix = guide_matrix.view(B, C, H, W)  # Restore spatial dimensions

        # 6. Hierarchical normalization (improved structure)
        guide_matrix = self.hierarchical_norm(guide_matrix)

        return guide_matrix


class LoRASegmentor(nn.Module):
    def __init__(self, sam, lora_rank):
        """
        Class for segmentation using LoRA-SAM.

        Parameters:
            lora_sam: SAM model instance wrapped with LoRA
            align_injector: multi-modal alignment injector instance
            text_proj: text feature projection layer
        """
        super().__init__()

        self.lora_sam = LoRA_Sam(sam, r=lora_rank)

    def forward(self, image, multimask_output, image_size, guide_matrix, gt):
        """
        During forward inference, use the modified lora-sam, where guide_matrix is used as weights.

        Parameters:
            image: input image (B, C, H, W)
            multimask_output: whether to output multiple masks
            image_size: image size
            guide_matrix: spatial guide matrix generated by GuideMatrixGenerator

        Returns:
            refined_masks: refined segmentation masks (B, 1, H, W)
        """

        return self.lora_sam(batched_input=image, multimask_output=multimask_output, image_size=image_size, gt=gt, guide_matrix=guide_matrix)

class FusionModule(nn.Module):
    def __init__(self, guide_channels=256, image_channels=3):
        super(FusionModule, self).__init__()
        # Reduce the channel dimension of the guide tensor
        self.guide_conv = nn.Conv2d(guide_channels, image_channels, kernel_size=1, stride=1, padding=0)
        # Spatial interpolation mode
        self.alpha = 0.5  # Default weight

    def forward(self, image, guide_matrix):
        # Check input shapes
        if guide_matrix.shape[0] != image.shape[0]:
            raise ValueError("Batch size of guide_matrix and image must match.")

        # Step 1: Adjust the resolution of the guide tensor
        guide_resized = F.interpolate(guide_matrix, size=(image.shape[2], image.shape[3]), mode='bilinear', align_corners=False)

        # Step 2: Reduce the channel number of guide_matrix to match image
        guide_reduced = self.guide_conv(guide_resized)  # [B, 3, H, W]

        # Step 3: Fuse image and guide_matrix
        fused_tensor = self.alpha * guide_reduced + (1 - self.alpha * 0.001) * image

        return fused_tensor

class MultiModalSegmentor(nn.Module):
    def __init__(self, sam, classnames, lora_rank=4):
        """
        Multimodal segmentor combining SAM (Segment Anything Model) and CLIP model.
        """
        super().__init__()
        self.lora_clip = None

        self.load_clip_model()
        # Freeze all parameters of the CLIP model, so they do not participate in training
        for param in self.lora_clip.parameters():
            param.requires_grad = False

        print("Use Clip Lora")
        self.lora_clip = LoRA_CLIP(self.lora_clip, r=4)
        # self.lora_clip.to(device)
        self.lora_clip.train()

        # Print the number of classes for later use in generating text prompts
        self.classnames = classnames
        print('Number of classes : ', len(classnames))

        # Build the network architecture
        self.GuideMatrixGenerator = GuideMatrixGenerator(sam, self.lora_clip)

        self.loRASegmentor = LoRASegmentor(sam, lora_rank)

    def load_clip_model(self):
        print("Initializing CLIP model...")
        self.lora_clip, _ = clip.load("ViT-B/32", device="cuda")


    def save_all_weights(self, filename: str) -> None:
        """
        Save the entire model (including LoRA parameters, SAM parameters, and any other modules) to the file `filename`.
        """
        # Get the current model's state_dict
        state_dict = self.state_dict()

        torch.save(state_dict, filename)
        print(f"All model weights saved to {filename}")

    def load_all_weights(self, filename: str) -> None:
        """
        Load the entire model (including LoRA, SAM, etc.) weights from `filename`.
        """
        state_dict = torch.load(filename, map_location="cpu")  # or "cuda", etc.

        # Similarly, for single GPU or normal cases, directly call .load_state_dict()
        # If using DP or DDP, you may need to call self.module.load_state_dict()
        self.load_state_dict(state_dict)
        print(f"All model weights loaded from {filename}")

    def forward(self, image, text_batch, multimask_output, image_size, gt):

        guide_matrix = self.GuideMatrixGenerator(image, text_batch)

        # print('guide_matrix', guide_matrix.shape)

        # outputs1, outputs2, attn1, attn1 = self.loRASegmentor(image, multimask_output, image_size, gt = gt , guide_matrix = guide_matrix)

        outputs1 = self.loRASegmentor(image, multimask_output, image_size, gt=gt, guide_matrix=guide_matrix)

        return outputs1
