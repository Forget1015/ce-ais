"""编码器输入输出形状属性测试。

Feature: ce-ais-framework
Property 1: 组件输入输出形状不变量
"""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from src.config.schema import CEWMConfig, EncoderConfig, SteeringConfig


# ======================================================================
# Property 1: Encoder 输出形状 [B, d_z] 且 L2 范数为 1
# ======================================================================


@settings(max_examples=100, deadline=None)
@given(
    batch_size=st.integers(min_value=1, max_value=8),
    latent_dim=st.sampled_from([64, 128, 256]),
)
def test_encoder_output_shape_and_norm(batch_size, latent_dim):
    """Encoder 输出形状应为 [B, d_z] 且 L2 范数为 1。

    Feature: ce-ais-framework, Property 1: 组件输入输出形状不变量
    """
    from src.encoders.contrastive_encoder import ContrastiveEncoder

    config = EncoderConfig(
        backbone_type="resnet18",
        pose_dim=7,
        visual_dim=512,
        latent_dim=latent_dim,
        temperature=0.07,
        image_size=[64, 64],
    )
    encoder = ContrastiveEncoder(config)
    encoder.eval()

    rgb = torch.randn(batch_size, 3, 64, 64)
    depth = torch.randn(batch_size, 1, 64, 64)
    pose = torch.randn(batch_size, 7)

    with torch.no_grad():
        z = encoder(rgb, depth, pose)

    # 形状检查
    assert z.shape == (batch_size, latent_dim), (
        f"Expected shape ({batch_size}, {latent_dim}), got {z.shape}"
    )

    # L2 范数检查（归一化后每个样本的范数应接近 1）
    norms = z.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), (
        f"L2 norms not close to 1: {norms}"
    )


# ======================================================================
# Property 1: CE-WM 输出形状 [B]
# ======================================================================


@settings(max_examples=100, deadline=None)
@given(
    batch_size=st.integers(min_value=1, max_value=8),
    seq_len=st.integers(min_value=1, max_value=32),
)
def test_cewm_output_shape(batch_size, seq_len):
    """CE-WM 输出形状应为 [B]。

    Feature: ce-ais-framework, Property 1: 组件输入输出形状不变量
    """
    from src.world_model.ce_wm import CausalEnergyWorldModel

    config = CEWMConfig(
        d_model=64,
        d_state=16,
        n_layers=2,
        expand_factor=2,
        mimo_groups=2,
        action_dim=7,
        latent_dim=32,
        dropout=0.1,
    )
    model = CausalEnergyWorldModel(config)
    model.eval()

    z_seq = torch.randn(batch_size, seq_len, 32)
    a_seq = torch.randn(batch_size, seq_len, 7)

    with torch.no_grad():
        energy = model(z_seq, a_seq)

    assert energy.shape == (batch_size,), (
        f"Expected shape ({batch_size},), got {energy.shape}"
    )
    assert torch.isfinite(energy).all(), "Energy contains non-finite values"


# ======================================================================
# Property 1: EFE Steering 输出形状与输入候选动作形状一致
# ======================================================================


@settings(max_examples=100, deadline=None)
@given(
    batch_size=st.integers(min_value=1, max_value=4),
    seq_len=st.integers(min_value=1, max_value=8),
)
def test_efe_steering_output_shape(batch_size, seq_len):
    """EFE Steering 输出形状应与输入候选动作形状一致。

    Feature: ce-ais-framework, Property 1: 组件输入输出形状不变量
    """
    from src.steering.efe_steering import EFESteering
    from src.world_model.ce_wm import CausalEnergyWorldModel

    steering_config = SteeringConfig(
        n_steps=2,
        step_size=0.01,
        anneal_rate=0.5,
        noise_scale=0.001,
        kl_weight=10.0,
    )
    cewm_config = CEWMConfig(
        d_model=64, d_state=16, n_layers=2, expand_factor=2,
        mimo_groups=2, action_dim=7, latent_dim=32, dropout=0.1,
    )

    steering = EFESteering(steering_config)
    ce_wm = CausalEnergyWorldModel(cewm_config)
    ce_wm.eval()

    a_init = torch.randn(batch_size, seq_len, 7)
    z_t = torch.randn(batch_size, 32)
    gating_lambda = torch.ones(batch_size)

    a_star = steering.steer(a_init, z_t, ce_wm, gating_lambda)

    assert a_star.shape == a_init.shape, (
        f"Expected shape {a_init.shape}, got {a_star.shape}"
    )
    assert torch.isfinite(a_star).all(), "Steered action contains non-finite values"
