"""退火朗之万动力学核心迭代逻辑。

实现退火朗之万动力学（Annealed Langevin Dynamics）的单步和多步迭代，
用于在动作空间中偏转候选动作。

核心公式:
    a_{k+1} = a_k - (ε_k / 2) · λ · [∇_a E_θ(z_t, a_k) + κ·(a_k - a_0)] + √ε_k · η_k

其中:
    ε_k = ε_0 · α^k  (退火步长)
    η_k ~ N(0, σ²·I)  (探索噪声)
    κ = kl_weight  (KL 散度约束强度)

梯度计算支持两种模式：
    - autograd: 使用 torch.autograd.grad（精确但慢，约 100ms/step）
    - finite_diff: 使用批量有限差分（近似但快，约 2ms/step）
"""

import torch


def _finite_diff_gradient(
    energy_fn,
    z_seq: torch.Tensor,
    action: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    """批量有限差分梯度估计。

    将 d_a 个扰动样本打包为一个批量前向调用，避免 autograd 反向传播开销。

    Args:
        energy_fn: (z_seq, a_seq) -> [B] 能量函数
        z_seq: [B, T, d_z]
        action: [B, T, d_a]
        eps: 有限差分步长

    Returns:
        grad: [B, T, d_a] 近似梯度
    """
    B, T, d_a = action.shape

    perturbations = torch.zeros(d_a, B, T, d_a, device=action.device, dtype=action.dtype)
    for d in range(d_a):
        perturbations[d, :, :, d] = eps

    a_base = action.unsqueeze(0)
    a_plus = a_base + perturbations
    a_minus = a_base - perturbations

    a_all = torch.cat([a_plus, a_minus], dim=0).reshape(2 * d_a * B, T, d_a)
    z_all = z_seq.unsqueeze(0).expand(2 * d_a, -1, -1, -1).reshape(2 * d_a * B, T, -1)

    with torch.no_grad():
        e_all = energy_fn(z_all, a_all)

    e_all = e_all.reshape(2 * d_a, B)
    e_plus = e_all[:d_a]
    e_minus = e_all[d_a:]

    grad_per_dim = (e_plus - e_minus) / (2 * eps)
    grad = grad_per_dim.permute(1, 0).unsqueeze(1).expand(-1, T, -1)

    return grad


def annealed_langevin_step(
    action: torch.Tensor,
    energy_grad: torch.Tensor,
    action_init: torch.Tensor,
    step_size: float,
    anneal_rate: float,
    step_idx: int,
    noise_scale: float,
    kl_weight: float,
    gating_lambda: torch.Tensor,
) -> torch.Tensor:
    """执行单步退火朗之万动力学更新。

    关键约束: 仅修改动作张量，不修改任何网络参数。

    Args:
        action: [B, T, d_a] 当前动作张量。
        energy_grad: [B, T, d_a] 能量函数对动作的梯度。
        action_init: [B, T, d_a] VLA 输出的初始动作（KL 锚点）。
        step_size: 初始步长 ε_0。
        anneal_rate: 退火率 α。
        step_idx: 当前迭代步索引 k。
        noise_scale: 探索噪声强度 σ。
        kl_weight: KL 散度权重 κ。
        gating_lambda: [B] 或 [B, 1, 1] 门控引导强度。

    Returns:
        更新后的动作张量 [B, T, d_a]。
    """
    eps_k = step_size * (anneal_rate ** step_idx)

    # KL 散度梯度: 约束不过度偏离 VLA 先验
    grad_kl = kl_weight * (action - action_init)

    # 确保 gating_lambda 可广播到 [B, T, d_a]
    if gating_lambda.dim() == 1:
        lam = gating_lambda.unsqueeze(-1).unsqueeze(-1)
    else:
        lam = gating_lambda

    # 探索噪声
    noise = noise_scale * torch.randn_like(action) * (eps_k ** 0.5)

    # 朗之万更新
    action_new = (
        action
        - (eps_k / 2) * lam * (energy_grad + grad_kl)
        + noise
    )

    return action_new


def run_langevin_dynamics(
    action_init: torch.Tensor,
    z_t: torch.Tensor,
    energy_fn,
    n_steps: int,
    step_size: float,
    anneal_rate: float,
    noise_scale: float,
    kl_weight: float,
    gating_lambda: torch.Tensor,
    grad_clip_norm: float = 1.0,
    action_clamp: float = 1.0,
    grad_mode: str = "finite_diff",
    fd_eps: float = 1e-3,
) -> torch.Tensor:
    """执行完整的退火朗之万动力学迭代。

    Args:
        action_init: [B, T, d_a] VLA 输出的初始动作。
        z_t: [B, d_z] 当前观测潜变量。
        energy_fn: 能量函数，接收 (z_seq, a_seq) 返回标量能量 [B]。
        n_steps: 迭代步数。
        step_size: 初始步长。
        anneal_rate: 退火率。
        noise_scale: 噪声强度。
        kl_weight: KL 权重。
        gating_lambda: [B] 门控强度。
        grad_clip_norm: 能量梯度的最大 L2 范数，防止梯度爆炸。
        action_clamp: 动作值的绝对值上界（匹配 Tanh 输出范围）。
        grad_mode: "autograd" 或 "finite_diff"。
        fd_eps: 有限差分步长（仅 finite_diff 模式）。

    Returns:
        校正后的动作张量 [B, T, d_a]。
    """
    if grad_mode == "finite_diff":
        a = action_init.clone().detach()

        for k in range(n_steps):
            T = a.shape[1]
            z_seq = z_t.unsqueeze(1).expand(-1, T, -1)

            grad_energy = _finite_diff_gradient(energy_fn, z_seq, a, eps=fd_eps)

            grad_norm = grad_energy.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            clip_coeff = (grad_clip_norm / grad_norm).clamp(max=1.0)
            grad_energy = grad_energy * clip_coeff

            a = annealed_langevin_step(
                action=a,
                energy_grad=grad_energy,
                action_init=action_init,
                step_size=step_size,
                anneal_rate=anneal_rate,
                step_idx=k,
                noise_scale=noise_scale,
                kl_weight=kl_weight,
                gating_lambda=gating_lambda,
            )
            a = a.clamp(-action_clamp, action_clamp)

        return a

    a = action_init.clone().detach().requires_grad_(True)

    for k in range(n_steps):
        T = a.shape[1]
        z_seq = z_t.unsqueeze(1).expand(-1, T, -1)

        energy = energy_fn(z_seq, a)
        grad_energy = torch.autograd.grad(
            energy.sum(), a, create_graph=False
        )[0]

        grad_norm = grad_energy.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        clip_coeff = (grad_clip_norm / grad_norm).clamp(max=1.0)
        grad_energy = grad_energy * clip_coeff

        a_new = annealed_langevin_step(
            action=a.detach(),
            energy_grad=grad_energy.detach(),
            action_init=action_init,
            step_size=step_size,
            anneal_rate=anneal_rate,
            step_idx=k,
            noise_scale=noise_scale,
            kl_weight=kl_weight,
            gating_lambda=gating_lambda,
        )
        a_new = a_new.clamp(-action_clamp, action_clamp)
        a = a_new.detach().requires_grad_(True)

    return a.detach()
