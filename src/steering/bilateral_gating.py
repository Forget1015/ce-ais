"""双向不确定性门控防御（Bilateral Epistemic Uncertainty Gating）。

根据 CE-WM 的认知不确定性动态调节能量引导强度，
防止极端 OOD 场景下错误能量梯度毒化动作流。

核心公式:
    λ(u_t) = λ_max · exp(-(u_t - μ_u)² / (2·σ_u²))

行为特性:
    - u_t ≈ μ_u（低不确定性）→ λ → λ_max，强力能量纠偏
    - u_t >> μ_u（高不确定性）→ λ → 0，回退至 VLA 保守先验
"""

from collections import deque
from typing import Optional

import torch
import torch.nn as nn

from src.config.schema import BilateralGatingConfig


class BilateralGating(nn.Module):
    """双向不确定性门控防御。

    Args:
        config: BilateralGatingConfig 配置对象。
    """

    def __init__(self, config: BilateralGatingConfig):
        super().__init__()
        self.lambda_max = config.lambda_max
        self.sensitivity = config.sensitivity  # σ_u
        self.window_size = config.window_size
        self.hard_uncertainty_threshold = getattr(config, "hard_uncertainty_threshold", None)
        self.min_lambda = float(getattr(config, "min_lambda", 0.0) or 0.0)
        self.history: deque = deque(maxlen=config.window_size)

    def compute_lambda(
        self, uncertainty: torch.Tensor
    ) -> torch.Tensor:
        """计算门控引导强度 λ(u_t)。

        Args:
            uncertainty: [B] 当前认知不确定性值。

        Returns:
            gating_lambda: [B] 引导强度参数，范围 [0, λ_max]。
        """
        # 更新历史均值
        self.history.append(uncertainty.mean().item())
        mu_u = sum(self.history) / len(self.history)

        # 高斯门控函数
        gating_lambda = self.lambda_max * torch.exp(
            -(uncertainty - mu_u) ** 2 / (2 * self.sensitivity ** 2)
        )

        return gating_lambda

    def get_mean_uncertainty(self) -> float:
        """获取当前历史不确定性均值。"""
        if not self.history:
            return 0.0
        return sum(self.history) / len(self.history)

    def reset(self) -> None:
        """重置历史缓冲。"""
        self.history.clear()
