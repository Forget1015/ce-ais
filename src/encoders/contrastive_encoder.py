"""对比预训练视觉-本体编码器（Contrastive Encoder）。

将高维多模态观测 o_t = (I_rgb, I_depth, p_ee) 映射为
低维连续潜变量 z_t ∈ R^{d_z}，并进行 L2 归一化。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config.schema import EncoderConfig
from src.encoders.backbone import build_backbone


# 各骨干网络的输出维度映射
_BACKBONE_OUTPUT_DIMS = {
    "resnet18": 512,
    "vit_small": 384,
}


class ContrastiveEncoder(nn.Module):
    """
    多模态对比预训练编码器。

    输入: o_t = (I_rgb ∈ R^{3×H×W}, I_depth ∈ R^{1×H×W}, p_ee ∈ R^{d_p})
    输出: z_t ∈ R^{d_z}  (默认 d_z = 128，L2 归一化)
    """

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config

        # 视觉骨干网络（RGB 和深度图共享）
        self.visual_backbone = build_backbone(config.backbone_type)

        # 深度图通道适配：1 → 3 通道，使其兼容 RGB 骨干
        self.depth_adapter = nn.Conv2d(1, 3, kernel_size=1)

        # 本体姿态 MLP
        self.pose_mlp = nn.Sequential(
            nn.Linear(config.pose_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
        )

        # 获取骨干实际输出维度
        backbone_dim = _BACKBONE_OUTPUT_DIMS.get(
            config.backbone_type, config.visual_dim
        )

        # 多模态融合头：拼接 RGB 特征 + 深度特征 + 姿态特征 → 潜变量
        fusion_input_dim = backbone_dim * 2 + 128  # rgb + depth + pose
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
        )

    def forward(
        self, rgb: torch.Tensor, depth: torch.Tensor, pose: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播。

        Args:
            rgb:   [B, 3, H, W] RGB 图像
            depth: [B, 1, H, W] 深度图
            pose:  [B, d_p]     末端执行器姿态

        Returns:
            z: [B, d_z] L2 归一化的潜变量
        """
        v_rgb = self.visual_backbone(rgb)
        v_depth = self.visual_backbone(self.depth_adapter(depth))
        v_pose = self.pose_mlp(pose)

        v_fused = torch.cat([v_rgb, v_depth, v_pose], dim=-1)
        z = self.fusion_head(v_fused)
        return F.normalize(z, dim=-1)

    def encode(self, observation: dict) -> torch.Tensor:
        """
        统一编码接口。

        Args:
            observation: {
                "rgb":   Tensor[B, 3, H, W],
                "depth": Tensor[B, 1, H, W],
                "pose":  Tensor[B, d_p]
            }

        Returns:
            z: Tensor[B, d_z] 归一化潜变量
        """
        return self.forward(
            observation["rgb"],
            observation["depth"],
            observation["pose"],
        )
