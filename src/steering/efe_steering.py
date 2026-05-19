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
        self.enable_trust_region = getattr(config, "enable_trust_region", True)
        self.action_delta_max = float(getattr(config, "action_delta_max", 0.05))
        self.enable_accept_reject = getattr(config, "enable_accept_reject", True)
        self.accept_energy_margin = float(getattr(config, "accept_energy_margin", 0.0))
        self.diagnostics = getattr(config, "diagnostics", True)
        self.mode = getattr(config, "mode", "langevin")
        self.candidate_count = int(getattr(config, "candidate_count", 0) or 0)
        self.candidate_noise_std = float(getattr(config, "candidate_noise_std", 0.01))
        self.deviation_weight = float(getattr(config, "deviation_weight", 10.0))

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
        if self.mode == "rerank":
            return self._rerank(a_init, z_t, ce_wm)

        a_star = run_langevin_dynamics(
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
        return self._apply_trust_region(a_init, a_star)

    def _apply_trust_region(
        self,
        a_init: torch.Tensor,
        a_candidate: torch.Tensor,
    ) -> torch.Tensor:
        if not self.enable_trust_region:
            return a_candidate
        delta = (a_candidate - a_init).clamp(-self.action_delta_max, self.action_delta_max)
        return a_init + delta

    def _rerank(
        self,
        a_init: torch.Tensor,
        z_t: torch.Tensor,
        ce_wm: nn.Module,
    ) -> torch.Tensor:
        candidate_count = max(self.candidate_count, 0)
        if candidate_count == 0:
            return a_init

        candidates = [a_init]
        for _ in range(candidate_count):
            noise = torch.randn_like(a_init) * self.candidate_noise_std
            candidates.append(self._apply_trust_region(a_init, a_init + noise))

        actions = torch.stack(candidates, dim=1)
        B, C = actions.shape[:2]
        flat_actions = actions.view(B * C, *actions.shape[2:])
        T = flat_actions.shape[1]
        z_seq = z_t.unsqueeze(1).expand(-1, C, -1).reshape(B * C, -1)
        z_seq = z_seq.unsqueeze(1).expand(-1, T, -1)

        with torch.no_grad():
            energy = ce_wm(z_seq, flat_actions).view(B, C)
            deviation = (actions - a_init.unsqueeze(1)).pow(2).mean(dim=tuple(range(2, actions.dim())))
            scores = energy + self.deviation_weight * deviation
            best = scores.argmin(dim=1)
            batch_idx = torch.arange(B, device=a_init.device)
            return actions[batch_idx, best]

    def forward(
        self,
        a_init: torch.Tensor,
        z_t: torch.Tensor,
        ce_wm: nn.Module,
        gating_lambda: torch.Tensor,
    ) -> torch.Tensor:
        """forward 别名，调用 steer()。"""
        return self.steer(a_init, z_t, ce_wm, gating_lambda)
