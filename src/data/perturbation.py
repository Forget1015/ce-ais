"""对抗性摄动策略注册机制。

提供可扩展的摄动策略注册表，用于生成能量景观训练的负样本。
支持通过装饰器 @PerturbationRegistry.register(name) 扩展新策略。

内置策略:
- velocity_reversal: 速度矢量反转
- gripper_anomaly: 夹爪状态异常信号
- random_displacement: 随机大幅度位移偏移
"""

from typing import Callable, Dict, List, Optional

import torch


class PerturbationRegistry:
    """可扩展的摄动策略注册表。

    通过装饰器模式注册摄动函数，支持按名称检索和批量应用。

    Usage:
        @PerturbationRegistry.register("my_strategy")
        def my_strategy(action: torch.Tensor, **kwargs) -> torch.Tensor:
            ...
    """

    _registry: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """装饰器：注册摄动策略函数。

        Args:
            name: 策略名称。

        Returns:
            装饰器函数。
        """
        def decorator(fn: Callable) -> Callable:
            cls._registry[name] = fn
            return fn
        return decorator

    @classmethod
    def get(cls, name: str) -> Callable:
        """按名称检索摄动策略。

        Args:
            name: 策略名称。

        Returns:
            摄动函数。

        Raises:
            KeyError: 策略未注册。
        """
        if name not in cls._registry:
            raise KeyError(
                f"Perturbation strategy '{name}' not registered. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def list_strategies(cls) -> List[str]:
        """列出所有已注册的策略名称。"""
        return list(cls._registry.keys())

    @classmethod
    def apply(
        cls, name: str, action: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """按名称应用摄动策略。

        Args:
            name: 策略名称。
            action: 原始动作张量。
            **kwargs: 策略特定参数。

        Returns:
            摄动后的动作张量。
        """
        fn = cls.get(name)
        return fn(action, **kwargs)


# ---- 内置摄动策略 ----


@PerturbationRegistry.register("velocity_reversal")
def velocity_reversal(action: torch.Tensor, **kwargs) -> torch.Tensor:
    """速度矢量反转。

    反转动作中的 xyz 速度分量（前 3 维），保持旋转和夹爪不变。

    Args:
        action: [B, d_a] 或 [B, T, d_a] 动作张量，前 3 维为 xyz 速度。

    Returns:
        摄动后的动作张量。
    """
    perturbed = action.clone()
    perturbed[..., :3] *= -1
    return perturbed


@PerturbationRegistry.register("gripper_anomaly")
def gripper_anomaly(action: torch.Tensor, **kwargs) -> torch.Tensor:
    """夹爪状态异常信号。

    将动作最后一维（夹爪信号）替换为随机值。

    Args:
        action: [B, d_a] 或 [B, T, d_a] 动作张量，最后一维为夹爪。

    Returns:
        摄动后的动作张量。
    """
    perturbed = action.clone()
    perturbed[..., -1] = torch.rand_like(perturbed[..., -1])
    return perturbed


@PerturbationRegistry.register("random_displacement")
def random_displacement(
    action: torch.Tensor, scale: float = 0.5, **kwargs
) -> torch.Tensor:
    """随机大幅度位移偏移。

    对 xyz 位移分量（前 3 维）添加高斯噪声。

    Args:
        action: [B, d_a] 或 [B, T, d_a] 动作张量。
        scale: 噪声标准差。

    Returns:
        摄动后的动作张量。
    """
    perturbed = action.clone()
    perturbed[..., :3] += scale * torch.randn_like(perturbed[..., :3])
    return perturbed


@PerturbationRegistry.register("collision_violation")
def collision_violation(
    action: torch.Tensor, scale: float = 4.0, **kwargs
) -> torch.Tensor:
    """生成会导致碰撞的轨迹。

    将 xyz 位移放大并随机翻转方向，模拟超出工作空间的急剧运动。

    Args:
        action: [B, d_a] 或 [B, T, d_a] 动作张量。
        scale: xyz 缩放倍数。

    Returns:
        摄动后的动作张量。
    """
    perturbed = action.clone()
    sign_flip = torch.sign(torch.randn_like(perturbed[..., :3]))
    perturbed[..., :3] = perturbed[..., :3].abs() * scale * sign_flip
    perturbed[..., 3:6] += 0.5 * torch.randn_like(perturbed[..., 3:6])
    return perturbed


@PerturbationRegistry.register("temporal_shuffle")
def temporal_shuffle(action: torch.Tensor, **kwargs) -> torch.Tensor:
    """时序打乱：破坏动作序列的因果时序。

    3D 输入 [B, T, d_a] 对 T 维度随机置换；
    2D 输入 [B, d_a] 随机翻转部分维度符号。

    Args:
        action: [B, d_a] 或 [B, T, d_a] 动作张量。

    Returns:
        摄动后的动作张量。
    """
    perturbed = action.clone()
    if perturbed.dim() == 3:
        B, T, d_a = perturbed.shape
        for b in range(B):
            perm = torch.randperm(T, device=action.device)
            perturbed[b] = perturbed[b][perm]
    else:
        mask = torch.randint(0, 2, perturbed.shape, device=action.device) * 2 - 1
        perturbed = perturbed * mask.float()
    return perturbed


@PerturbationRegistry.register("joint_limit_violation")
def joint_limit_violation(
    action: torch.Tensor, prob: float = 0.5, **kwargs
) -> torch.Tensor:
    """超关节限位：将动作推向极限区域。

    以概率 prob 将每个维度替换为极端值（[0.8, 1.0] 或 [-1.0, -0.8]），
    夹爪维度强制随机二值化。

    Args:
        action: [B, d_a] 或 [B, T, d_a] 动作张量。
        prob: 每个元素被替换的概率。

    Returns:
        摄动后的动作张量。
    """
    perturbed = action.clone()
    mask = torch.rand_like(perturbed) < prob
    extreme_mag = 0.8 + 0.2 * torch.rand_like(perturbed)
    extreme_sign = torch.sign(torch.randn_like(perturbed))
    extreme_vals = extreme_mag * extreme_sign
    perturbed = torch.where(mask, extreme_vals, perturbed)
    perturbed[..., -1] = (torch.randint(0, 2, perturbed[..., -1].shape,
                                         device=action.device).float() * 2 - 1)
    return perturbed


@PerturbationRegistry.register("uniform_sampling")
def uniform_sampling(
    action: torch.Tensor, low: float = -1.0, high: float = 1.0, **kwargs
) -> torch.Tensor:
    """从 action space 均匀采样，覆盖整个空间。"""
    return torch.rand_like(action) * (high - low) + low


@PerturbationRegistry.register("multi_scale_gaussian")
def multi_scale_gaussian(
    action: torch.Tensor, **kwargs
) -> torch.Tensor:
    """从多个 σ 中随机选一个加高斯噪声。"""
    scales = [0.01, 0.05, 0.1, 0.3]
    idx = torch.randint(0, len(scales), (1,)).item()
    return action + scales[idx] * torch.randn_like(action)


@PerturbationRegistry.register("micro_perturbation")
def micro_perturbation(
    action: torch.Tensor, scale: float = 0.003, **kwargs
) -> torch.Tensor:
    """微小高斯扰动，匹配 CE-AIS steering 工作尺度。"""
    return action + scale * torch.randn_like(action)


@PerturbationRegistry.register("medium_perturbation")
def medium_perturbation(
    action: torch.Tensor, scale: float = 0.01, **kwargs
) -> torch.Tensor:
    """中等高斯扰动，匹配 CE-AIS n_steps=3 预期尺度。"""
    return action + scale * torch.randn_like(action)


@PerturbationRegistry.register("large_perturbation")
def large_perturbation(
    action: torch.Tensor, scale: float = 0.05, **kwargs
) -> torch.Tensor:
    """较大高斯扰动，介于 micro 和原始策略之间。"""
    return action + scale * torch.randn_like(action)


@PerturbationRegistry.register("hard_negative_mining")
def hard_negative_mining(
    action: torch.Tensor,
    energy_fn=None,
    z_context: Optional[torch.Tensor] = None,
    n_candidates: int = 10,
    **kwargs,
) -> torch.Tensor:
    """困难负样本挖掘：用当前能量模型找出能量低但物理非法的样本。

    生成 n_candidates 个候选摄动，选择能量最低（对模型而言最"可信"）的作为
    最具挑战性的负样本。若 energy_fn 不可用则回退到 random_displacement。

    Args:
        action: [B, d_a] 或 [B, T, d_a] 动作张量。
        energy_fn: 能量函数，接收 (z_seq, a_seq) 返回 [B]。
        z_context: [B, T, d_z] 或 [B, d_z] 对应的潜变量上下文。
        n_candidates: 候选负样本数量。

    Returns:
        摄动后的动作张量。
    """
    if energy_fn is None or z_context is None:
        return random_displacement(action, scale=0.5)

    base_strategies = ["velocity_reversal", "gripper_anomaly",
                       "random_displacement", "collision_violation",
                       "temporal_shuffle", "joint_limit_violation"]

    candidates = []
    for i in range(n_candidates):
        strat_name = base_strategies[i % len(base_strategies)]
        candidates.append(PerturbationRegistry.apply(strat_name, action))

    z = z_context
    if z.dim() == 2:
        z = z.unsqueeze(1)

    best = candidates[0]
    best_energy = None

    with torch.no_grad():
        for cand in candidates:
            a = cand
            if a.dim() == 2:
                a = a.unsqueeze(1)
            T = a.shape[1]
            z_exp = z.expand(-1, T, -1) if z.shape[1] != T else z
            e = energy_fn(z_exp, a)
            if best_energy is None or (e < best_energy).any():
                if best_energy is None:
                    best_energy = e
                    best = cand
                else:
                    # 逐样本选最低能量
                    mask = e < best_energy
                    if mask.any():
                        if cand.dim() == 2:
                            best[mask] = cand[mask]
                        else:
                            best[mask] = cand[mask]
                        best_energy[mask] = e[mask]

    return best
