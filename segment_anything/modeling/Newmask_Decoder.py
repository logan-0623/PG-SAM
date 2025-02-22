import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Type


class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        layers = []
        in_dims = [input_dim] + [hidden_dim] * (num_layers - 1)
        out_dims = [hidden_dim] * (num_layers - 1) + [output_dim]
        for i, (in_d, out_d) in enumerate(zip(in_dims, out_dims)):
            layers.append(nn.Linear(in_d, out_d))
            if i < num_layers - 1:
                layers.append(nn.GELU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class DynamicCrossAttention(nn.Module):
    """动态权重交叉注意力模块（修复版）"""

    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim

        # 投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # 动态门控机制
        self.gate_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )

        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, guide_matrix=None):
        batch_size = query.size(0)

        # 如果稀疏提示为空，则直接返回空，避免后续插值错误
        if query.size(1) == 0:
            return query

        # 投影操作
        q = self.q_proj(query)  # [B, L_q, D]
        k = self.k_proj(key)  # [B, L_k, D]
        v = self.v_proj(value)  # [B, L_k, D]

        # 动态特征融合
        if guide_matrix is not None:
            # 维度对齐
            if guide_matrix.size(1) != q.size(1):
                # 使用插值代替池化进行特征对齐
                guide_aligned = F.interpolate(
                    guide_matrix.permute(0, 2, 1),
                    size=q.size(1),
                    mode='nearest'
                ).permute(0, 2, 1)
            else:
                guide_aligned = guide_matrix

            # 门控融合
            fusion_gate = self.gate_mlp(torch.cat([q, guide_aligned], dim=-1))  # [B, L_q, 1]
            q = q * fusion_gate + guide_aligned * (1 - fusion_gate)

        # 多头注意力重塑
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L_q, D/H]
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L_k, D/H]
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L_k, D/H]

        # 注意力计算
        attn_weights = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, L_q, L_k]
        attn_weights = F.softmax(attn_weights, dim=-1)

        # 上下文聚合
        attn_output = (attn_weights @ v).transpose(1, 2)  # [B, L_q, H, D/H]
        attn_output = attn_output.contiguous().view(batch_size, -1, self.embed_dim)  # [B, L_q, D]

        return self.out_proj(attn_output)

class GuidedMaskDecoder(nn.Module):
    def __init__(
            self,
            transformer_dim: int = 256,
            num_multimask_outputs: int = 8,
            activation: Type[nn.Module] = nn.GELU,
            iou_head_depth: int = 3,
            iou_head_hidden_dim: int = 256,
            guide_dim: int = 256,
            num_attention_heads: int = 8,
            num_classes: int = 1
    ):
        super().__init__()
        # 基础参数设置
        self.transformer_dim = transformer_dim
        self.num_multimask_outputs = num_multimask_outputs

        # 提示嵌入初始化
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.mask_tokens = nn.Embedding(num_multimask_outputs + 1, transformer_dim)

        # Guide Matrix处理模块
        self.guide_proj = nn.Sequential(
            nn.Conv2d(guide_dim, transformer_dim // 4, 3, padding=1),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.Conv2d(transformer_dim // 4, transformer_dim, 1)
        )

        # 动态特征交互模块
        self.sparse_attention = DynamicCrossAttention(transformer_dim, num_attention_heads)
        self.dense_fusion = nn.Sequential(
            nn.Conv2d(transformer_dim * 2, transformer_dim, 1),
            LayerNorm2d(transformer_dim),
            activation()
        )

        # 多尺度上采样路径
        self.upsampling = nn.ModuleList([
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            GuideAwareBlock(transformer_dim // 4, guide_dim),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation()
        ])

        # 输出预测头
        self.iou_head = MLP(transformer_dim, iou_head_hidden_dim, num_multimask_outputs + 1, iou_head_depth)
        self.mask_hypernets = nn.ModuleList([
            DynamicMLP(transformer_dim, transformer_dim // 8)
            for _ in range(num_multimask_outputs + 1)
        ])

        # 动态点采样模块
        self.point_sampler = GumbelPointSampler(transformer_dim // 8)  # 匹配上采样后的通道数

        self.mask_classifier = nn.Conv2d(transformer_dim // 8, num_classes, kernel_size=1)


    def build_grid(self, size, device):
        h, w = size
        y_coord = torch.linspace(0, 1, h, device=device)
        x_coord = torch.linspace(0, 1, w, device=device)
        grid_y, grid_x = torch.meshgrid(y_coord, x_coord)
        return torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # [1,H,W,2]

    def forward(
            self,
            image_embeddings: torch.Tensor,
            guide_matrix: torch.Tensor,  # [B, C, H, W]
            sparse_prompts: torch.Tensor,
            dense_prompts: torch.Tensor,
            multimask_output: bool = True,
            **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # 阶段1: Guide Matrix融合
        guide_feat = self.guide_proj(guide_matrix)

        fused_embeddings = image_embeddings + dense_prompts + guide_feat

        # 阶段2: 动态特征交互
        sparse_enhanced = self.sparse_attention(
            sparse_prompts,
            fused_embeddings.flatten(2).permute(0, 2, 1),
            fused_embeddings.flatten(2).permute(0, 2, 1),
            guide_matrix.flatten(2).permute(0, 2, 1)
        )

        # 阶段3: 提示token生成
        tokens = self._generate_tokens(sparse_enhanced, fused_embeddings)

        # 阶段4: 特征解码与上采样
        x = fused_embeddings
        for layer in self.upsampling:
            if isinstance(layer, GuideAwareBlock):
                x = layer(x, guide_matrix)  # 显式传递引导矩阵
            else:
                x = layer(x)
        up_features = x

        up_features = F.interpolate(up_features, size=(224, 224), mode='bilinear', align_corners=False)

        # 阶段5: 动态掩模生成
        masks = self._generate_masks(tokens, up_features)

        # 阶段6: 优化点采样与迭代
        if self.training or kwargs.get('refine', False):
            sampled_points, _ = self.point_sampler(up_features)
            # 更新稀疏提示（这里 _update_sparse_prompts 可以在后续进行实际实现）
            refined_sparse = self._update_sparse_prompts(sparse_enhanced, sampled_points)
            # 重新生成完整的 tokens（基础 tokens + 更新后的稀疏提示）
            tokens = self._generate_tokens(refined_sparse, fused_embeddings)
            masks = self._generate_masks(tokens, up_features)

        # 阶段7: 输出处理
        iou_pred = self.iou_head(tokens[:, 0])
        final_masks = self._select_masks(masks, iou_pred, multimask_output)

        final_masks = self.mask_classifier(final_masks)

        return {
            'masks': final_masks,
            'iou_pred': iou_pred,
            'sampled_points': sampled_points if self.training else None
        }

    def _generate_tokens(self, sparse_prompts, image_feat):
        """Token生成修正版（关键修复）"""
        B = image_feat.size(0)  # Batch size

        # 生成iou_token [B,1,D]
        iou_tokens = self.iou_token.weight.unsqueeze(0).expand(B, -1, -1)

        # 生成mask_tokens [B,N+1,D]
        mask_tokens = self.mask_tokens.weight.unsqueeze(0).expand(
            B, self.num_multimask_outputs + 1, -1
        )

        # 拼接基础tokens
        base_tokens = torch.cat([iou_tokens, mask_tokens], dim=1)  # [B, (1)+(N+1), D]

        # 合并稀疏提示（假设sparse_prompts形状为[B, L, D]）
        return torch.cat([base_tokens, sparse_prompts], dim=1)  # [B, (N+2)+L, D]

    def _generate_masks(self, tokens, features):
        mask_weights = []
        for i in range(self.num_multimask_outputs + 1):
            # 修正索引偏移量（原i+1错误）
            token_idx = 1 + i  # 对应mask_tokens的位置
            weight_gen = self.mask_hypernets[i](tokens[:, token_idx], features)
            mask_weights.append(weight_gen)
        return torch.stack(mask_weights, dim=1)

    def _select_masks(self, masks, iou_pred, multimask):
        if multimask:
            weights = F.softmax(iou_pred, dim=-1)
            return torch.einsum('bnchw,bn->bchw', masks, weights)
        else:
            best_idx = torch.argmax(iou_pred, dim=-1)
            return masks[torch.arange(masks.size(0)), best_idx]

    def _update_sparse_prompts(self, prompts, sampled_points):
        """动态更新稀疏提示点"""
        # 此处实现点坐标到嵌入的转换逻辑
        # 需要与prompt encoder配合实现
        return prompts  # 示例返回，实际需实现坐标编码逻辑


class GuideAwareBlock(nn.Module):
    """引导感知特征细化块"""

    def __init__(self, feat_dim, guide_dim):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(feat_dim + guide_dim, feat_dim * 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(feat_dim * 2, feat_dim, 1)
        )
        self.norm = LayerNorm2d(feat_dim)

    def forward(self, x, guide):
        guide = F.interpolate(guide, x.shape[2:], mode='bilinear', align_corners=False)
        fused = self.fusion(torch.cat([x, guide], dim=1))
        return self.norm(x + fused)

class DynamicMLP(nn.Module):
    """动态权重生成网络"""

    def __init__(self, token_dim, feat_dim):
        super().__init__()
        self.weight_gen = nn.Sequential(
            nn.Linear(token_dim, token_dim // 2),
            nn.GELU(),
            nn.Linear(token_dim // 2, feat_dim)
        )

    def forward(self, token, features):
        weights = self.weight_gen(token).unsqueeze(-1).unsqueeze(-1)
        return weights * features


class GumbelPointSampler(nn.Module):
    """可微分点采样模块（最终修正版）"""

    def __init__(self, in_channels):
        super().__init__()
        self.density_net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels // 4, 1, 1)
        )

    def build_grid(self, size, device):
        """生成归一化坐标网格（优化版本）"""
        h, w = size
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, h, device=device),
            torch.linspace(0, 1, w, device=device),
            indexing='ij'
        )
        return torch.stack([grid_x, grid_y], dim=-1).flatten(0, 1)  # [H*W, 2]

    def forward(self, feature_map, num_points=4, temperature=0.1):
        B, _, H, W = feature_map.shape

        # 密度预测 → [B,1,H,W]
        density = self.density_net(feature_map)

        # 权重生成 → [B,1,HW]
        point_weights = F.softmax(
            (density.flatten(2) + self.gumbel_noise(B, H * W, feature_map.device)) / temperature,
            dim=-1
        )

        # 坐标计算（关键修正部分）
        grid = self.build_grid((H, W), feature_map.device)  # [HW,2]
        grid = grid.unsqueeze(0).expand(B, -1, -1)  # [B,HW,2]

        # 矩阵乘法维度对齐
        sampled_points = torch.bmm(point_weights, grid)  # [B,1,HW] @ [B,HW,2] → [B,1,2]

        return sampled_points.squeeze(1), point_weights  # [B,2], [B,1,HW]

    def gumbel_noise(self, batch_size, dim, device):
        """生成Gumbel噪声"""
        return -torch.log(-torch.log(torch.rand(batch_size, 1, dim, device=device)))

# 使用示例
if __name__ == "__main__":
    decoder = GuidedMaskDecoder(guide_dim=256, num_classes = 9)

    # 合法输入示例（至少包含一个提示点）
    image_emb = torch.randn(12, 256, 14, 14)    # 图像嵌入
    guide_mat = torch.randn(12, 256, 14, 14)    # 引导矩阵

    sparse_prompt = torch.randn(1, 0, 256)     # 至少一个提示点
    dense_prompt = torch.randn(1, 256, 14, 14) # 密集提示


    outputs = decoder(
        image_embeddings=image_emb,
        guide_matrix=guide_mat,
        sparse_prompts=sparse_prompt,
        dense_prompts=guide_mat,
        multimask_output=True
    )


    print(f"Output masks shape: {outputs['masks'].shape}")    # 应输出 [2, 9, 224, 224]
    print(f"IOU predictions shape: {outputs['iou_pred'].shape}")  # 应输出 [2, 9]