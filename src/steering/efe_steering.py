"""基于期望自由能最小化的推理时动作偏转（EFE-Guided Steering）。

在推理阶段通过退火朗之万动力学在动作空间中偏转候选动作，
使其符合物理因果律约束。

关键约束: 仅修改动作输出张量，不修改任何网络参数。
"""

import torch
import torch.nn as nn

from src.config.schema import SteeringConfig
from src.steering.langevin import run_langevin_dynamics


class EFESteering(nn.Module):
    """基于期望自由能最小化的推理时动作偏转。

    将 VLA 输出的最高概率动作序列作为朗之万动力学采样的初始粒子位置，
    执行可配置步数的退火朗之万迭代，沿能量下降方向偏转动作。

    关键约束: 不修改任何网络参数（requires_grad=False），
    仅修改动作输出张量。

    Args:
        config: SteeringConfig 配置对象。
    """

    def __init__(self, config: SteeringConfig):
        super().__init__()
        self.n_steps = config.n_steps
        self.step_size = config.step_size
        self.anneal_rate = config.anneal_rate
        self.noise_scale = config.noise_scale
        self.kl_weight = config.kl_weight
        self.grad_mode = getattr(config, "grad_mode", "finite_diff")

    def steer(
        self,
        a_init: torch.Tensor,
        z_t: torch.Tensor,
        ce_wm: nn.Module,
        gating_lambda: torch.Tensor,
    ) -> torch.Tensor:
        """执行 EFE 引导偏转。

        Args:
            a_init: [B, T, d_a] VLA 输出的候选动作序列。
            z_t: [B, d_z] 当前观测潜变量。
            ce_wm: CausalEnergyWorldModel 冻结的能量世界模型。
            gating_lambda: [B] 双向门控引导强度。

        Returns:
            a_star: [B, T, d_a] 校正后的动作序列。
        """
        return run_langevin_dynamics(
            action_init=a_init,
            z_t=z_t,
            energy_fn=ce_wm,
            n_steps=self.n_steps,
            step_size=self.step_size,
            anneal_rate=self.anneal_rate,
            noise_scale=self.noise_scale,
            kl_weight=self.kl_weight,
            gating_lambda=gating_lambda,
            grad_mode=self.grad_mode,
        )

    def forward(
        self,
        a_init: torch.Tensor,
        z_t: torch.Tensor,
        ce_wm: nn.Module,
        gating_lambda: torch.Tensor,
    ) -> torch.Tensor:
        """forward 别名，调用 steer()。"""
        return self.steer(a_init, z_t, ce_wm, gating_lambda)
