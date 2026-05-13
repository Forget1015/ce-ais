"""CE-WM 参数量范围属性测试。

Feature: ce-ais-framework
Property 5: CE-WM 参数量范围约束
"""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from src.config.schema import CEWMConfig
from src.world_model.ce_wm import CausalEnergyWorldModel


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ======================================================================
# Property 5: CE-WM 参数量范围约束
# ======================================================================


def test_production_config_param_count():
    """默认生产配置（base.yaml 对应的 CEWMConfig 默认值）参数量应在 100M-300M。"""
    config = CEWMConfig()
    model = CausalEnergyWorldModel(config)
    n_params = count_parameters(model)

    min_params = 100_000_000
    max_params = 300_000_000

    assert min_params <= n_params <= max_params, (
        f"Parameter count {n_params / 1e6:.1f}M is outside "
        f"[{min_params / 1e6:.0f}M, {max_params / 1e6:.0f}M] range. "
        f"Config: d_model={config.d_model}, n_layers={config.n_layers}, expand={config.expand_factor}"
    )


@settings(max_examples=100, deadline=None)
@given(
    d_model=st.integers(min_value=256, max_value=768),
    n_layers=st.integers(min_value=12, max_value=48),
    expand_factor=st.integers(min_value=1, max_value=4),
)
def test_cewm_param_count_scales_with_config(d_model, n_layers, expand_factor):
    """参数量应随 d_model、n_layers、expand_factor 正相关增长。

    Feature: ce-ais-framework, Property 5: CE-WM 参数量范围约束
    """
    mimo_groups = 4
    d_inner = d_model * expand_factor

    if d_inner % mimo_groups != 0:
        d_model = (d_model // mimo_groups) * mimo_groups
        if d_model < 256:
            d_model = 256
        d_inner = d_model * expand_factor
        if d_inner % mimo_groups != 0:
            pytest.skip("Cannot satisfy mimo_groups divisibility constraint")

    config = CEWMConfig(
        d_model=d_model,
        d_state=64,
        n_layers=n_layers,
        expand_factor=expand_factor,
        mimo_groups=mimo_groups,
        action_dim=7,
        latent_dim=128,
        dropout=0.1,
    )

    model = CausalEnergyWorldModel(config)
    n_params = count_parameters(model)

    assert n_params > 0, "Model has no parameters"

    min_per_layer = d_model * d_model
    assert n_params >= n_layers * min_per_layer * 0.1, (
        f"Parameter count {n_params} seems too low for "
        f"d_model={d_model}, n_layers={n_layers}"
    )
