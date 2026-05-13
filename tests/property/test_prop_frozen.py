"""推理时参数冻结属性测试。

Feature: ce-ais-framework
Property 9: 推理时参数绝对冻结
"""

import copy

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from src.config.schema import CEWMConfig, EncoderConfig, SteeringConfig
from src.steering.efe_steering import EFESteering
from src.world_model.ce_wm import CausalEnergyWorldModel


# ======================================================================
# Property 9: 推理时参数绝对冻结
# ======================================================================


@settings(max_examples=100, deadline=None)
@given(
    batch_size=st.integers(min_value=1, max_value=4),
    seq_len=st.integers(min_value=1, max_value=4),
)
def test_cewm_params_frozen_during_inference(batch_size, seq_len):
    """推理前后 CE-WM 所有参数 requires_grad=False 且值不变。

    Feature: ce-ais-framework, Property 9: 推理时参数绝对冻结
    """
    config = CEWMConfig(
        d_model=64, d_state=16, n_layers=2, expand_factor=2,
        mimo_groups=2, action_dim=7, latent_dim=32, dropout=0.1,
    )
    model = CausalEnergyWorldModel(config)
    model.eval()

    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False

    # 保存参数快照
    params_before = {
        name: param.data.clone()
        for name, param in model.named_parameters()
    }

    # 执行推理
    z_seq = torch.randn(batch_size, seq_len, 32)
    a_seq = torch.randn(batch_size, seq_len, 7)

    with torch.no_grad():
        energy = model(z_seq, a_seq)
        _ = model.get_uncertainty(z_seq, a_seq, n_samples=3)

    # 验证所有参数 requires_grad=False
    for name, param in model.named_parameters():
        assert not param.requires_grad, (
            f"Parameter {name} has requires_grad=True after inference"
        )

    # 验证参数值不变
    for name, param in model.named_parameters():
        assert torch.equal(param.data, params_before[name]), (
            f"Parameter {name} changed during inference"
        )


@settings(max_examples=100, deadline=None)
@given(
    batch_size=st.integers(min_value=1, max_value=4),
)
def test_encoder_params_frozen_during_inference(batch_size):
    """推理前后 Encoder 所有参数 requires_grad=False 且值不变。

    Feature: ce-ais-framework, Property 9: 推理时参数绝对冻结
    """
    from src.encoders.contrastive_encoder import ContrastiveEncoder

    config = EncoderConfig(
        backbone_type="resnet18",
        latent_dim=32,
        image_size=[64, 64],
    )
    encoder = ContrastiveEncoder(config)
    encoder.eval()

    for param in encoder.parameters():
        param.requires_grad = False

    params_before = {
        name: param.data.clone()
        for name, param in encoder.named_parameters()
    }

    rgb = torch.randn(batch_size, 3, 64, 64)
    depth = torch.randn(batch_size, 1, 64, 64)
    pose = torch.randn(batch_size, 7)

    with torch.no_grad():
        z = encoder(rgb, depth, pose)

    for name, param in encoder.named_parameters():
        assert not param.requires_grad, (
            f"Parameter {name} has requires_grad=True after inference"
        )
        assert torch.equal(param.data, params_before[name]), (
            f"Parameter {name} changed during inference"
        )


@settings(max_examples=100, deadline=None)
@given(
    batch_size=st.integers(min_value=1, max_value=4),
    seq_len=st.integers(min_value=1, max_value=4),
)
def test_steering_does_not_modify_model_params(batch_size, seq_len):
    """EFE 偏转过程不应修改任何模型参数。

    Feature: ce-ais-framework, Property 9: 推理时参数绝对冻结
    """
    cewm_config = CEWMConfig(
        d_model=64, d_state=16, n_layers=2, expand_factor=2,
        mimo_groups=2, action_dim=7, latent_dim=32, dropout=0.0,
    )
    steering_config = SteeringConfig(
        n_steps=2, step_size=0.01, anneal_rate=0.5,
        noise_scale=0.001, kl_weight=10.0,
    )

    ce_wm = CausalEnergyWorldModel(cewm_config)
    ce_wm.eval()
    for param in ce_wm.parameters():
        param.requires_grad = False

    steering = EFESteering(steering_config)

    params_before = {
        name: param.data.clone()
        for name, param in ce_wm.named_parameters()
    }

    a_init = torch.randn(batch_size, seq_len, 7)
    z_t = torch.randn(batch_size, 32)
    gating_lambda = torch.ones(batch_size)

    a_star = steering.steer(a_init, z_t, ce_wm, gating_lambda)

    for name, param in ce_wm.named_parameters():
        assert torch.equal(param.data, params_before[name]), (
            f"CE-WM parameter {name} changed during steering"
        )
