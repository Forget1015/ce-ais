"""双流推理集成测试。

使用 Mock VLA + 真实小规模 CE-WM 验证端到端推理。
验证参数冻结、能量评估、偏转输出正确性。
"""

import pytest
import torch
import torch.nn as nn

from src.config.schema import (
    BilateralGatingConfig,
    CEWMConfig,
    EncoderConfig,
    SteeringConfig,
)
from src.dual_stream.topology import DualStreamTopology
from src.dual_stream.vla_adapter import VLAAdapter
from src.encoders.contrastive_encoder import ContrastiveEncoder
from src.steering.bilateral_gating import BilateralGating
from src.steering.efe_steering import EFESteering
from src.world_model.ce_wm import CausalEnergyWorldModel


# ------------------------------------------------------------------
# Mock VLA
# ------------------------------------------------------------------


class MockVLAAdapter(VLAAdapter):
    """Mock VLA 适配器，输出固定形状的随机动作。"""

    def __init__(self, action_dim=7, chunk_size=4):
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self._model = nn.Linear(1, 1)  # 占位模型

    def predict(self, observation, instruction):
        B = observation["rgb"].shape[0]
        return torch.randn(B, self.chunk_size, self.action_dim)

    def parameters(self):
        return self._model.parameters()


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def dual_stream_components():
    """创建双流推理所需的所有组件。"""
    encoder_config = EncoderConfig(
        backbone_type="resnet18",
        latent_dim=32,
        image_size=[64, 64],
    )
    cewm_config = CEWMConfig(
        d_model=64, d_state=16, n_layers=2, expand_factor=2,
        mimo_groups=2, action_dim=7, latent_dim=32, dropout=0.1,
    )
    steering_config = SteeringConfig(
        n_steps=2, step_size=0.01, anneal_rate=0.5,
        noise_scale=0.001, kl_weight=10.0,
    )
    gating_config = BilateralGatingConfig(
        lambda_max=1.0, sensitivity=0.1,
        window_size=10, mc_samples=2,
    )

    vla = MockVLAAdapter(action_dim=7, chunk_size=4)
    encoder = ContrastiveEncoder(encoder_config)
    ce_wm = CausalEnergyWorldModel(cewm_config)
    steering = EFESteering(steering_config)
    gating = BilateralGating(gating_config)

    return {
        "vla": vla,
        "encoder": encoder,
        "ce_wm": ce_wm,
        "steering": steering,
        "gating": gating,
    }


@pytest.fixture
def dual_stream(dual_stream_components):
    """创建 DualStreamTopology 实例。"""
    return DualStreamTopology(
        vla_adapter=dual_stream_components["vla"],
        encoder=dual_stream_components["encoder"],
        ce_wm=dual_stream_components["ce_wm"],
        steering=dual_stream_components["steering"],
        gating=dual_stream_components["gating"],
        mc_samples=2,
    )


@pytest.fixture
def mock_obs():
    """Mock 观测数据。"""
    return {
        "rgb": torch.randn(1, 3, 64, 64),
        "depth": torch.randn(1, 1, 64, 64),
        "pose": torch.randn(1, 7),
    }


# ------------------------------------------------------------------
# 测试
# ------------------------------------------------------------------


class TestDualStreamIntegration:
    """双流推理集成测试。"""

    def test_all_params_frozen(self, dual_stream):
        """所有模型参数应被冻结（requires_grad=False）。"""
        for name, param in dual_stream.encoder.named_parameters():
            assert not param.requires_grad, (
                f"Encoder param {name} not frozen"
            )

        for name, param in dual_stream.ce_wm.named_parameters():
            assert not param.requires_grad, (
                f"CE-WM param {name} not frozen"
            )

        for param in dual_stream.vla.parameters():
            assert not param.requires_grad, "VLA param not frozen"

    def test_step_produces_valid_output(self, dual_stream, mock_obs):
        """单步推理应产生有效的动作输出和诊断信息。"""
        action, info = dual_stream.step(mock_obs, "pick up the cup")

        # 动作形状检查
        assert action.dim() == 3, f"Expected 3D action, got {action.dim()}D"
        assert action.shape[0] == 1  # batch size
        assert action.shape[2] == 7  # action dim

        # 动作应有限
        assert torch.isfinite(action).all(), "Action contains non-finite values"

        # 诊断信息应包含关键字段
        assert "energy_before" in info
        assert "energy_after" in info
        assert "uncertainty" in info
        assert "gating_lambda" in info

    def test_energy_evaluation(self, dual_stream, mock_obs):
        """能量评估应产生有限标量值。"""
        _, info = dual_stream.step(mock_obs, "pick up the cup")

        energy_before = info["energy_before"]
        energy_after = info["energy_after"]

        assert torch.isfinite(energy_before).all(), (
            "Energy before is not finite"
        )
        assert torch.isfinite(energy_after).all(), (
            "Energy after is not finite"
        )

    def test_params_unchanged_after_step(
        self, dual_stream_components, mock_obs
    ):
        """推理步骤后所有模型参数应保持不变。"""
        ds = DualStreamTopology(
            vla_adapter=dual_stream_components["vla"],
            encoder=dual_stream_components["encoder"],
            ce_wm=dual_stream_components["ce_wm"],
            steering=dual_stream_components["steering"],
            gating=dual_stream_components["gating"],
            mc_samples=2,
        )

        # 保存参数快照
        encoder_params = {
            n: p.data.clone()
            for n, p in ds.encoder.named_parameters()
        }
        cewm_params = {
            n: p.data.clone()
            for n, p in ds.ce_wm.named_parameters()
        }

        # 执行推理
        ds.step(mock_obs, "pick up the cup")

        # 验证参数不变
        for n, p in ds.encoder.named_parameters():
            assert torch.equal(p.data, encoder_params[n]), (
                f"Encoder param {n} changed"
            )
        for n, p in ds.ce_wm.named_parameters():
            assert torch.equal(p.data, cewm_params[n]), (
                f"CE-WM param {n} changed"
            )

    def test_safe_step_fallback(self, dual_stream, mock_obs):
        """safe_step 应在正常情况下返回偏转结果。"""
        action, info = dual_stream.safe_step(mock_obs, "pick up the cup")

        assert action.dim() == 3
        assert torch.isfinite(action).all()
        assert info.get("status") == "steered" or "fallback" in info

    def test_multiple_steps_consistent(self, dual_stream, mock_obs):
        """多次推理步骤应产生一致的输出格式。"""
        for i in range(3):
            action, info = dual_stream.step(mock_obs, "pick up the cup")

            assert action.dim() == 3
            assert action.shape[2] == 7
            assert torch.isfinite(action).all()
            assert "energy_before" in info
