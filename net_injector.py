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
# 1. 定义包装后的 LoRA 多头注意力模块 (_LoRA_MHA)
############################################################
class _LoRA_MHA(nn.Module):
    def __init__(self, mha: nn.MultiheadAttention, r: int = 4):
        super().__init__()
        self.mha = mha  # 原始的 MultiheadAttention
        self.embed_dim = mha.embed_dim
        self.num_heads = mha.num_heads
        self.r = r
        self.dropout = mha.dropout

        # 冻结原始 MHA 参数
        for param in self.mha.parameters():
            param.requires_grad = False

        # 构造 LoRA adapter：针对 Q 和 V 分支
        self.linear_a_q = nn.Linear(self.embed_dim, r, bias=False)
        self.linear_b_q = nn.Linear(r, self.embed_dim, bias=False)
        self.linear_a_v = nn.Linear(self.embed_dim, r, bias=False)
        self.linear_b_v = nn.Linear(r, self.embed_dim, bias=False)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        # 提取原始 in_proj_weight/in_proj_bias 用于计算 Q/K/V
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

        # 计算原始 Q, K, V（假设输入形状为 [L, B, embed_dim]）
        Q_orig = F.linear(query, q_weight, q_bias)
        K_orig = F.linear(key, k_weight, k_bias)
        V_orig = F.linear(value, v_weight, v_bias)

        # 为了计算 LoRA 更新项，将输入转为 float（全精度），计算完成后再转换回输入的 dtype（比如 fp16）
        Q_lora = self.linear_b_q(self.linear_a_q(query.float())).to(query.dtype)
        V_lora = self.linear_b_v(self.linear_a_v(value.float())).to(value.dtype)

        # 得到更新后的 Q 和 V（K 保持不变）
        Q = Q_orig + Q_lora
        V = V_orig + V_lora

        # 调用内置多头注意力函数，注意这里不再用 in_proj_weight/in_proj_bias
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
# 2. 修改 LoRA_CLIP 模块，将 LoRA 注入应用到 CLIP 文本编码器中的注意力层
############################################################

class LoRA_CLIP(nn.Module):
    """
    将 LoRA 适配器应用到 CLIP 模型的文本编码器上。
    这里对每个 transformer block 中的注意力层进行遍历，
    如果注意力层为 nn.MultiheadAttention，则用 _LoRA_MHA 包装替换。
    """

    def __init__(self, clip_model, r: int = 4, lora_layers=None):
        super().__init__()
        self.clip_model = clip_model
        self.r = r

        # 这里可以先冻结 CLIP 模型的全部参数，
        # 如果只想训练 LoRA 参数，请确保后续 LoRA 部分 remains requires_grad=True
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # 根据你所使用的 CLIP 实现，假设文本 transformer 存放于 clip_model.transformer.resblocks
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
                # 如果你使用的 CLIP 版本中，attn 里有 c_attn，可以采用之前的方法（见之前代码示例）
                try:
                    c_attn = block.attn.c_attn
                except AttributeError:
                    raise AttributeError("无法找到 attn.c_attn，也不属于 nn.MultiheadAttention，请检查 CLIP 模型实现。")
                pass

        # for i, block in enumerate(self.clip_model.transformer.resblocks):
        #     if isinstance(block.attn, _LoRA_MHA):
        #         for name, param in block.attn.named_parameters():
        #             # 如果名称中含有 "linear_a" 或 "linear_b"，则设为可训练
        #             if "linear_a" in name or "linear_b" in name:
        #                 param.requires_grad = True
        #                 print(f"Unfroze LoRA param in block {i}: {name}")

    def forward(self, *args, **kwargs):
        return self.clip_model(*args, **kwargs)







class GuideMatrixGenerator(nn.Module):
    def __init__(self, sam, lora_clip):
        """
        用于生成 guide matrix 的类。

        参数:
            sam: 原始 SAM 模型实例
            lora_clip: CLIP 模型实例 - Lora
        """
        super().__init__()
        self.sam_model = sam
        self.lora_clip = lora_clip

        self.text_proj = nn.Linear(512, 256)

        self.sim_proj = nn.Linear(512, 196)

        self.tokenizer = AutoTokenizer.from_pretrained(
            "/mnt/sda/feilongtang/John/Miccai_sam/code/dual-sam/bert-base-uncased"
        )

        self.layer_norm = torch.nn.LayerNorm(196)

    def get_enhanced_text_features(self, text_batch, device):
        """增强型文本特征提取"""
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
            # 使用预设提示的增强实现
            prompt_embeds = self.load_preset_prompts(device)
            return prompt_embeds.mean(dim=0, keepdim=True)

    def dynamic_similarity(self, img_feat, txt_feat):
        """动态相似度计算模块"""
        # 多模态交互
        cross_attn = torch.einsum('bc,bc->b', img_feat, txt_feat)  # (B,)
        # 输出维度改为 (B, 196)
        spatial_weights = torch.sigmoid(self.sim_proj(img_feat.float()))  # (B, 196)
        # 交叉注意力和空间权重相乘，自动广播 (B, 1) * (B, 196) -> (B, 196)
        return (cross_attn.unsqueeze(1) * spatial_weights) / 10.0  # 控制数值范围

    def hierarchical_norm(self, x):
        """层次化归一化结构"""
        # 通道级归一化
        x = F.layer_norm(x.permute(0, 2, 3, 1), [x.size(1)]).permute(0, 3, 1, 2)
        # 空间级归一化
        spatial_mean = x.mean(dim=[2, 3], keepdim=True)
        spatial_std = x.std(dim=[2, 3], keepdim=True)
        return (x - spatial_mean) / (spatial_std + 1e-6)

    def forward(self, image, text_batch):
        # 1. 特征提取与文本编码
        image_features = self.lora_clip.clip_model.encode_image(image)
        sam_features, _ = self.sam_model.image_encoder(image)  # (B, C, H, W)
        device = image.device

        B, C, H, W = sam_features.shape
        L = H * W  # 空间位置总数，理想情况下 L == 196 = 14 * 14

        # 文本特征获取 (优化后的实现)
        text_features = self.get_enhanced_text_features(text_batch, device)
        # 2. 多模态特征交互 (新增模块)
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)

        # 动态相似度权重 (改进实现)
        similarity_weights = self.dynamic_similarity(image_features, text_features)  # (B, 196)

        # 3. 空间注意力矩阵计算 (矩阵化优化)
        sam_flat = sam_features.view(B, C, L)
        sam_norm = F.normalize(sam_flat, p=2, dim=1)

        # 批量矩阵乘法代替循环
        spatial_attn = torch.einsum('bcl,bcm->blm', sam_norm, sam_norm) / math.sqrt(C)
        spatial_attn = F.softmax(spatial_attn, dim=-1)  # (B, L, L)

        # 4. 特征聚合 (向量化实现)
        # 加权特征计算
        weighted_feats = torch.einsum('bcl,blm->bcm', sam_flat, spatial_attn)  # (B, C, L)
        # 局部特征增强
        local_feats = sam_flat  # (B, C, L)
        combined_feats = (local_feats + weighted_feats) * 0.5  # 平均融合

        # 5. 多模态引导矩阵生成
        # 使用 unsqueeze 在通道维度插入 1，使得 similarity_weights (B, 196) 变为 (B, 1, 196)
        guide_matrix = combined_feats * similarity_weights.unsqueeze(1)  # (B, C, L)
        guide_matrix = guide_matrix.view(B, C, H, W)  # 恢复空间维度

        # 6. 层次化归一化 (改进结构)
        guide_matrix = self.hierarchical_norm(guide_matrix)

        return guide_matrix


class LoRASegmentor(nn.Module):
    def __init__(self, sam, lora_rank):
        """
        使用 LoRA-SAM 进行分割的类。

        参数:
            lora_sam: 封装了 LoRA 的 SAM 模型实例
            align_injector: 多模态对齐注入器实例
            text_proj: 文本特征投影层
        """
        super().__init__()

        self.lora_sam = LoRA_Sam(sam, r = lora_rank)

    def forward(self, image, multimask_output, image_size, guide_matrix, gt):
        """
        执行前向推理过程中，使用修改后的lora-sam，其中guide_matrix用作权重。

        参数:
            image: 输入图像 (B, C, H, W)
            multimask_output: 是否输出多掩码
            image_size: 图像大小
            guide_matrix: 由 GuideMatrixGenerator 生成的空间引导矩阵

        返回:
            refined_masks: 精细化后的分割掩码 (B, 1, H, W)
        """


        return self.lora_sam(batched_input = image, multimask_output = multimask_output, image_size = image_size, gt = gt, guide_matrix = guide_matrix)

class FusionModule(nn.Module):
    def __init__(self, guide_channels=256, image_channels=3):
        super(FusionModule, self).__init__()
        # 引导张量通道降维
        self.guide_conv = nn.Conv2d(guide_channels, image_channels, kernel_size=1, stride=1, padding=0)
        # 空间插值模式
        self.alpha = 0.5  # 默认权重

    def forward(self, image, guide_matrix):
        # 检查输入形状
        if guide_matrix.shape[0] != image.shape[0]:
            raise ValueError("Batch size of guide_matrix and image must match.")

        # 步骤1: 调整引导张量的分辨率
        guide_resized = F.interpolate(guide_matrix, size=(image.shape[2], image.shape[3]), mode='bilinear', align_corners=False)

        # 步骤2: 将 guide_matrix 通道数降维到与 image 一致
        guide_reduced = self.guide_conv(guide_resized)  # [B, 3, H, W]

        # 步骤3: 融合 image 和 guide_matrix
        fused_tensor = self.alpha * guide_reduced + (1 - self.alpha * 0.001) * image

        return fused_tensor

class MultiModalSegmentor(nn.Module):
    def __init__(self, sam, classnames, lora_rank=4):
        """
        多模态分割器，结合 SAM (Segment Anything Model) 与 CLIP 模型。

        """
        super().__init__()
        self.lora_clip = None

        self.load_clip_model()
        # 冻结 CLIP 模型所有参数，不参与训练
        for param in self.lora_clip.parameters():
            param.requires_grad = False

        print("Use Clip Lora")
        self.lora_clip = LoRA_CLIP(self.lora_clip, r=4)
        # self.lora_clip.to(device)
        self.lora_clip.train()

        # print(self.lora_clip)

        # 保存类别名称，后续生成文本提示时需要
        self.classnames = classnames
        print('Number of classes : ', len(classnames))

        # 生成网络框架
        self.GuideMatrixGenerator = GuideMatrixGenerator(sam, self.lora_clip)

        self.loRASegmentor = LoRASegmentor(sam, lora_rank)

        self.fusion_module = FusionModule(guide_channels=256, image_channels=3)

    def load_clip_model(self):
        print("Initializing CLIP model...")
        self.lora_clip, _ = clip.load("ViT-B/32", device="cuda")


    def save_all_weights(self, filename: str) -> None:
        """
        保存整个模型（包括 LoRA 参数、SAM 的参数、以及其它可能的模块）的权重到 `filename` 文件。
        """
        # 获取当前模型的 state_dict
        state_dict = self.state_dict()

        torch.save(state_dict, filename)
        print(f"All model weights saved to {filename}")

    def load_all_weights(self, filename: str) -> None:
        """
        从 `filename` 加载整个模型（包括 LoRA、SAM 等）的权重。
        """
        state_dict = torch.load(filename, map_location="cpu")  # 或 "cuda" 等

        # 同理，如果是单卡或普通情况，直接 .load_state_dict()
        # 如果是 DP 或 DDP，可能要调用 self.module.load_state_dict()
        self.load_state_dict(state_dict)
        print(f"All model weights loaded from {filename}")

    def forward(self, image, text_batch, multimask_output, image_size, gt):

        guide_matrix = self.GuideMatrixGenerator(image, text_batch)

        # print('guide_matrix', guide_matrix.shape)

        # fused_tensor = self.fusion_module(image, guide_matrix)

        # outputs1, outputs2, attn1, attn1 = self.loRASegmentor(image, multimask_output, image_size, gt = gt , guide_matrix = guide_matrix)

        outputs1 = self.loRASegmentor(image, multimask_output, image_size, gt=gt, guide_matrix=guide_matrix)

        return outputs1
