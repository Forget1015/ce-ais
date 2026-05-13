"""能量梯度存在性属性测试。

Feature: ce-ais-framework
Property 10: 能量梯度存在且有限
"""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from src.config.schema import CEWMConfig
from src.world_model.ce_wm import CausalEnergyWorldModel


# ======================================================================
# Property 10: 能量梯度存在且有限
# ======================================================================


@settings(max_examples=100, deadline=None)
@given(
    batch_size=st.integers(min_value=1, max_value=4),
    seq_len=st.integers(min_value=1, max_value=8),
)
def test_energy_gradient_exists_and_finite(batch_size, seq_len):
    """对于任意合法输入，CE-WM 能量函数相对于动作张量的梯度应存在、有限且非全零。

    Feature: ce-ais-framework, Property 10: 能量梯度存在且有限
    """
    config = CEWMConfig(
        d_model=64,
        d_state=16,
        n_layers=2,
        expand_factor=2,
        mimo_groups=2,
        action_dim=7,
        latent_dim=32,
        dropout=0.0,  # 关闭 dropout 以确保确定性
    )
    model = CausalEnergyWorldModel(config)
    model.eval()

    z_seq = torch.randn(batch_size, seq_len, 32)
    a_seq = torch.randn(batch_size, seq_len, 7, requires_grad=True)

    energy = model(z_seq, a_seq)
    energy_sum = energy.sum()

    # 计算梯度
    grad = torch.autograd.grad(energy_sum, a_seq, create_graph=False)[0]

    # 梯度应存在
    assert grad is not None, "Gradient does not exist"

    # 梯度形状应与输入动作一致
    assert grad.shape == a_seq.shape, (
        f"Gradient shape {grad.shape} != action shape {a_seq.shape}"
    )

    # 梯度应有限
    assert torch.isfinite(grad).all(), (
        f"Gradient contains non-finite values: "
        f"nan={torch.isnan(grad).sum()}, inf={torch.isinf(grad).sum()}"
    )

    # 梯度应非全零
    assert grad.abs().sum() > 0, "Gradient is all zeros"
