"""Mamba-3 因果能量世界模型（CE-WM）。

基于 Mamba-3 架构追踪长时序物理状态，输出标量能量值评估动作的物理合法性。

架构:
    输入投影层 → N 层 Mamba-3 堆叠 → 能量头

输入: 潜变量序列 z_{1:T} ∈ R^{T×d_z}, 候选动作序列 a_{1:T} ∈ R^{T×d_a}
输出: 标量能量值 E ∈ R

参数量目标: 100M - 300M
"""

import torch
import torch.nn as nn

from src.config.schema import CEWMConfig
from src.world_model.energy_head import EnergyHead
from src.world_model.mamba3_core import Mamba3Block


class CausalEnergyWorldModel(nn.Module):
    """
    Mamba-3 因果能量世界模型。

    Args:
        config: CEWMConfig 配置对象
    """

    def __init__(self, config: CEWMConfig):
        super().__init__()
        self.config = config

        # 输入投影层: (d_z + d_a) → d_model
        self.input_proj = nn.Linear(
            config.latent_dim + config.action_dim, config.d_model
        )

        # Mamba-3 时序引擎（堆叠 N 层）
        self.mamba_layers = nn.ModuleList(
            [
                Mamba3Block(
                    d_model=config.d_model,
                    d_state=config.d_state,
                    expand=config.expand_factor,
                    mimo_groups=config.mimo_groups,
                    dropout=config.dropout,
                    headdim=getattr(config, "headdim", 64),
                )
                for _ in range(config.n_layers)
            ]
        )

        # MLP 能量头
        self.energy_head = EnergyHead(config.d_model)

    def forward(
        self, z_seq: torch.Tensor, a_seq: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播：计算状态-动作序列的标量能量值。

        Args:
            z_seq: [B, T, d_z] 潜变量序列
            a_seq: [B, T, d_a] 候选动作序列

        Returns:
            energy: [B] 标量能量值
        """
        # 拼接潜变量和动作 → 投影到 d_model
        x = self.input_proj(torch.cat([z_seq, a_seq], dim=-1))  # [B, T, d_model]

        # 通过 N 层 Mamba-3
        for layer in self.mamba_layers:
            x = layer(x)

        # 取最后时间步的隐状态
        h_final = x[:, -1, :]  # [B, d_model]

        # 能量头输出标量
        energy = self.energy_head(h_final)  # [B]
        return energy

    def get_uncertainty(
        self,
        z_seq: torch.Tensor,
        a_seq: torch.Tensor,
        n_samples: int = 5,
    ) -> torch.Tensor:
        """
        MC-Dropout 认知不确定性估计。

        通过多次前向传播（启用 Dropout）计算能量值的方差，
        作为模型对当前输入的认知不确定性度量。

        Args:
            z_seq:     [B, T, d_z] 潜变量序列
            a_seq:     [B, T, d_a] 候选动作序列
            n_samples: MC-Dropout 采样次数

        Returns:
            uncertainty: [B] 认知不确定性（能量方差）
        """
        was_training = self.training

        # 启用 Dropout 进行随机采样
        self.train()
        energies = torch.stack(
            [self.forward(z_seq, a_seq) for _ in range(n_samples)]
        )  # [n_samples, B]

        # 恢复原始模式
        if not was_training:
            self.eval()

        # 方差作为不确定性
        return energies.var(dim=0)  # [B]
