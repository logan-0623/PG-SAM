import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from segment_anything.modeling import Sam
from segment_anything.modeling.sam import Sam_Encoder, Sam_OLD

from torch.nn.parameter import Parameter
from segment_anything import sam_model_registry

# from segment_anything.modeling.Newmask_Decoder import GuidedMaskDecoder



class _LoRA_qkv(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        return qkv


class MultiScaleCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        self.skip_connections = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(num_layers - 1)
        ])

    def forward(self, src, guide):
        # Multi-scale feature fusion
        # src = src + guide
        src = src + F.interpolate(guide, size=src.shape[2:], mode='bilinear', align_corners=False)

        B, C, H, W = src.shape
        src = src.view(B, C, -1).permute(0, 2, 1)
        guide = guide.view(B, C, -1).permute(0, 2, 1)

        # Multi-layer cross attention
        for i, layer in enumerate(self.layers):
            residual = src
            src = layer(torch.cat([src, guide], dim=1))[:, :src.size(1)]
            if i > 0:
                src = src + self.skip_connections[i - 1](residual)
        return src.permute(0, 2, 1).view(B, C, H, W)


class AdaptiveFeatureFusion(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dim, embed_dim // 8, 1),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 8, embed_dim, 1),
            nn.Sigmoid()
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, image_feat, guide_feat):
        # Channel attention
        channel_weight = self.channel_attn(image_feat + guide_feat)

        # Spatial attention
        avg_pool = torch.mean(image_feat, dim=1, keepdim=True)
        max_pool = torch.max(image_feat, dim=1, keepdim=True)[0]
        spatial_weight = self.spatial_attn(torch.cat([avg_pool, max_pool], dim=1))

        # Adaptive fusion
        fused_feat = channel_weight * spatial_weight * guide_feat
        return image_feat + fused_feat

class DynamicPromptGenerator(nn.Module):
    def __init__(self, guide_dim=256, num_points=5):
        super().__init__()
        self.num_points = num_points
        self.coord_conv = nn.Sequential(
            nn.Conv2d(guide_dim, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1)
        )
        self.point_head = nn.Conv2d(64, num_points * 2, 1)  # (x, y) coordinates
        self.box_head = nn.Conv2d(64, 4, 1)  # (x1, y1, x2, y2)

    def forward(self, guide_matrix):
        B, C, H, W = guide_matrix.shape
        feat = self.coord_conv(guide_matrix)  # [B, 64, H, W]

        # Generate key points
        point_logits = self.point_head(feat)  # [B, num_points * 2, H, W]
        point_logits = point_logits.mean(dim=(2, 3))  # Global average pooling, resulting in [B, num_points * 2]
        point_coords = torch.sigmoid(point_logits.view(B, self.num_points, 2))  # [B, num_points, 2]

        # Generate bounding boxes
        box_params = torch.sigmoid(
            self.box_head(feat).mean(dim=(2, 3))  # [B, 4]
        )
        boxes = box_params * torch.tensor([W, H, W, H], device=guide_matrix.device)  # [B, 4]

        return point_coords, boxes



class LoRA_Sam(nn.Module):
    def __init__(self, sam_model: Sam_OLD, r: int = 8, lora_layers=None):
        super().__init__()
        self.sam = sam_model
        self.r = r

        # Freeze original parameters
        for param in self.sam.parameters():
            param.requires_grad = False

        # Initialize enhancement modules
        self.cross_attn = MultiScaleCrossAttention(256, 8)

        self.feature_fusion = AdaptiveFeatureFusion(256)

        # Dynamic LoRA layer configuration
        self.lora_layer = list(
            range(len(sam_model.image_encoder.blocks)))
        self.w_As = []  # These are linear layers
        self.w_Bs = []

        # Let's freeze first
        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        self._init_lora_adapters()

        self.reset_parameters()

        self.dynamic_prompt_generator = DynamicPromptGenerator()

    def _init_lora_adapters(self):
        r = 4
        for t_layer_i, blk in enumerate(self.sam.image_encoder.blocks):
            # If we only want a few LoRA layers instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)


    def forward(self, batched_input, multimask_output, image_size, gt=None, guide_matrix=None):
        # Extract base features
        image_embeddings, low_image_embeddings = self.sam.image_encoder(batched_input)

        # print(f"Image embeddings require grad: {image_embeddings.requires_grad}")

        # Multi-stage feature enhancement
        if guide_matrix is not None:
            # -------------------------------guide matrix method -------------------------------
            # First stage: Cross-scale attention fusion
            attn_feat = self.cross_attn(image_embeddings, guide_matrix)

            # Second stage: Adaptive feature fusion
            enhanced_feat = self.feature_fusion(image_embeddings, attn_feat)

            # Third stage: Residual connection
            image_embeddings = image_embeddings + 0.1 * enhanced_feat


        # Subsequent processing flow
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None, boxes=None, masks=None
        )

        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            # Using guide matrix
            guide_matrix=guide_matrix
        )
        masks = self.sam.postprocess_masks(
            low_res_masks,
            input_size=(image_size, image_size),
            original_size=(image_size, image_size)
        )
        outputs = {
            'masks': masks,
            'iou_predictions': iou_predictions,
            'low_res_logits': low_res_masks
        }

        return outputs


    def save_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.

        Save both LoRA and FC parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')

        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        prompt_encoder_tensors = {}
        mask_decoder_tensors = {}

        # Save prompt encoder, only `state_dict`, the `named_parameter` is not permitted
        if isinstance(self.sam, torch.nn.DataParallel) or isinstance(self.sam, torch.nn.parallel.DistributedDataParallel):
            state_dict = self.sam.module.state_dict()
        else:
            state_dict = self.sam.state_dict()
        for key, value in state_dict.items():
            if 'prompt_encoder' in key:
                prompt_encoder_tensors[key] = value
            if 'mask_decoder' in key:
                mask_decoder_tensors[key] = value


        merged_dict = {**a_tensors, **b_tensors, **prompt_encoder_tensors, **mask_decoder_tensors}
        torch.save(merged_dict, filename)


    def load_state_dict(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.

        Load both LoRA and FC parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')

        state_dict = torch.load(filename)

        for i, w_A_linear in enumerate(self.w_As):
            saved_key = f"w_a_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_A_linear.weight = Parameter(saved_tensor)

        for i, w_B_linear in enumerate(self.w_Bs):
            saved_key = f"w_b_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_B_linear.weight = Parameter(saved_tensor)

        sam_dict = self.sam.state_dict()
        sam_keys = sam_dict.keys()

        # Load prompt encoder
        prompt_encoder_keys = [k for k in sam_keys if 'prompt_encoder' in k]
        prompt_encoder_values = [state_dict[k] for k in prompt_encoder_keys]
        prompt_encoder_new_state_dict = {k: v for k, v in zip(prompt_encoder_keys, prompt_encoder_values)}
        sam_dict.update(prompt_encoder_new_state_dict)

        # Load mask decoder
        mask_decoder_keys = [k for k in sam_keys if 'mask_decoder' in k]
        mask_decoder_values = [state_dict[k] for k in mask_decoder_keys]
        mask_decoder_new_state_dict = {k: v for k, v in zip(mask_decoder_keys, mask_decoder_values)}
        sam_dict.update(mask_decoder_new_state_dict)


        self.sam.load_state_dict(sam_dict)



if __name__ == "__main__":
    sam, _ = sam_model_registry["build_sam_vit_b_new"](checkpoint="model_weights/sam_vit_b_01ec64.pth", image_size=224, num_classes=2)
    lora_sam = LoRA_Sam(sam, r=4)
    guide_matrix = torch.rand(size=(2, 256, 14,14))
    output = lora_sam(torch.rand(size=(2, 3, 224, 224)), multimask_output = True, image_size = 224, guide_matrix = guide_matrix)
    output_masks = output['masks']
    low_res_logits = output['low_res_logits']
    print('output_masks', output_masks.shape)
    print('low_res_logits', low_res_logits.shape)
