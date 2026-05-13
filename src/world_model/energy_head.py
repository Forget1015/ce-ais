"""MLP 能量头。

将 Mamba-3 时序引擎的最终隐状态映射为标量能量值 E ∈ R，
用于评估状态-动作元组的物理合法性。

正样本（专家轨迹）→ 低能量
负样本（对抗摄动）→ 高能量
"""

import torch
import torch.nn as nn


class EnergyHead(nn.Module):
    """
    MLP 能量头。

    架构: Linear(d_model → d_model//2) → GELU → Linear(d_model//2 → 1)

    Args:
        d_model: 输入隐藏维度
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        计算标量能量值。

        Args:
            h: [B, d_model] 最终时间步的隐状态

        Returns:
            energy: [B] 标量能量值
        """
        return self.mlp(h).squeeze(-1)
