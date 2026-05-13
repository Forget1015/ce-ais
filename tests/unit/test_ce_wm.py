"""单元测试：Mamba-3 核心层、能量头和 CE-WM 完整模型。"""

import torch
import pytest

from src.config.schema import CEWMConfig
from src.world_model.mamba3_core import Mamba3Block
from src.world_model.energy_head import EnergyHead
from src.world_model.ce_wm import CausalEnergyWorldModel


# ---------------------------------------------------------------------------
# Mamba3Block 测试 (Task 5.1)
# ---------------------------------------------------------------------------

class TestMamba3Block:
    """Mamba-3 核心层单元测试。"""

    def test_output_shape(self):
        block = Mamba3Block(d_model=64, d_state=16, expand=2, mimo_groups=2)
        x = torch.randn(2, 10, 64)
        y = block(x)
        assert y.shape == (2, 10, 64)

    def test_residual_connection(self):
        block = Mamba3Block(d_model=64, d_state=16, expand=2, mimo_groups=2)
        block.eval()
        x = torch.randn(2, 5, 64)
        y = block(x)
        assert not torch.allclose(x, y, atol=1e-6)
        assert y.abs().sum() > 0

    def test_configurable_params(self):
        configs = [
            dict(d_model=128, d_state=32, expand=1, mimo_groups=2),
            dict(d_model=256, d_state=64, expand=2, mimo_groups=4),
            dict(d_model=64, d_state=16, expand=4, mimo_groups=1),
        ]
        for cfg in configs:
            block = Mamba3Block(**cfg)
            x = torch.randn(1, 3, cfg["d_model"])
            y = block(x)
            assert y.shape == x.shape, f"Failed for config {cfg}"

    def test_complex_state_updates(self):
        block = Mamba3Block(d_model=64, d_state=16, expand=2, mimo_groups=2)
        block.eval()
        x1 = torch.randn(1, 5, 64)
        x2 = torch.randn(1, 5, 64)
        y1 = block(x1)
        y2 = block(x2)
        assert not torch.allclose(y1, y2, atol=1e-5)

    def test_mimo_groups_assertion(self):
        """d_inner 不能被 mimo_groups 整除时应报错。"""
        with pytest.raises(AssertionError):
            Mamba3Block(d_model=64, d_state=16, expand=2, mimo_groups=3)

    def test_dropout_effect(self):
        block = Mamba3Block(d_model=64, d_state=16, expand=2, mimo_groups=2, dropout=0.5)
        block.train()
        x = torch.randn(2, 5, 64)
        y1 = block(x)
        y2 = block(x)
        assert not torch.allclose(y1, y2, atol=1e-6)

    def test_no_nan_inf(self):
        block = Mamba3Block(d_model=64, d_state=16, expand=2, mimo_groups=2)
        block.eval()
        x = torch.randn(4, 8, 64)
        y = block(x)
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()

    def test_batch_size_one(self):
        block = Mamba3Block(d_model=64, d_state=16, expand=2, mimo_groups=2)
        x = torch.randn(1, 3, 64)
        y = block(x)
        assert y.shape == (1, 3, 64)

    def test_sequence_length_one(self):
        block = Mamba3Block(d_model=64, d_state=16, expand=2, mimo_groups=2)
        x = torch.randn(2, 1, 64)
        y = block(x)
        assert y.shape == (2, 1, 64)

    def test_headdim_param(self):
        """headdim 参数保留兼容性。"""
        block = Mamba3Block(d_model=128, d_state=32, expand=2, mimo_groups=4, headdim=32)
        x = torch.randn(1, 5, 128)
        y = block(x)
        assert y.shape == (1, 5, 128)


# ---------------------------------------------------------------------------
# EnergyHead 测试 (Task 5.2)
# ---------------------------------------------------------------------------

class TestEnergyHead:
    """MLP 能量头单元测试。"""

    def test_output_shape(self):
        head = EnergyHead(d_model=64)
        h = torch.randn(4, 64)
        energy = head(h)
        assert energy.shape == (4,)

    def test_scalar_output(self):
        head = EnergyHead(d_model=128)
        h = torch.randn(1, 128)
        energy = head(h)
        assert energy.shape == (1,)
        assert energy.dim() == 1

    def test_no_nan(self):
        head = EnergyHead(d_model=64)
        h = torch.randn(8, 64)
        energy = head(h)
        assert not torch.isnan(energy).any()


# ---------------------------------------------------------------------------
# CausalEnergyWorldModel 测试 (Task 5.2)
# ---------------------------------------------------------------------------

class TestCausalEnergyWorldModel:
    """CE-WM 完整模型单元测试。"""

    @pytest.fixture
    def small_config(self):
        return CEWMConfig(
            d_model=64,
            d_state=16,
            n_layers=2,
            expand_factor=2,
            mimo_groups=2,
            action_dim=7,
            latent_dim=32,
            dropout=0.1,
        )

    def test_forward_output_shape(self, small_config):
        model = CausalEnergyWorldModel(small_config)
        z_seq = torch.randn(2, 5, small_config.latent_dim)
        a_seq = torch.randn(2, 5, small_config.action_dim)
        energy = model(z_seq, a_seq)
        assert energy.shape == (2,)

    def test_forward_no_nan(self, small_config):
        model = CausalEnergyWorldModel(small_config)
        model.eval()
        z_seq = torch.randn(4, 8, small_config.latent_dim)
        a_seq = torch.randn(4, 8, small_config.action_dim)
        energy = model(z_seq, a_seq)
        assert not torch.isnan(energy).any()
        assert not torch.isinf(energy).any()

    def test_different_inputs_different_energy(self, small_config):
        model = CausalEnergyWorldModel(small_config)
        model.eval()
        z = torch.randn(1, 5, small_config.latent_dim)
        a1 = torch.randn(1, 5, small_config.action_dim)
        a2 = torch.randn(1, 5, small_config.action_dim) * 10
        e1 = model(z, a1)
        e2 = model(z, a2)
        assert not torch.allclose(e1, e2, atol=1e-6)

    def test_get_uncertainty(self, small_config):
        model = CausalEnergyWorldModel(small_config)
        z_seq = torch.randn(2, 5, small_config.latent_dim)
        a_seq = torch.randn(2, 5, small_config.action_dim)
        uncertainty = model.get_uncertainty(z_seq, a_seq, n_samples=3)
        assert uncertainty.shape == (2,)
        assert (uncertainty >= 0).all()

    def test_get_uncertainty_restores_mode(self, small_config):
        model = CausalEnergyWorldModel(small_config)
        model.eval()
        z = torch.randn(1, 3, small_config.latent_dim)
        a = torch.randn(1, 3, small_config.action_dim)
        model.get_uncertainty(z, a, n_samples=2)
        assert not model.training

        model.train()
        model.get_uncertainty(z, a, n_samples=2)
        assert model.training

    def test_energy_gradient_exists(self, small_config):
        model = CausalEnergyWorldModel(small_config)
        model.eval()
        z_seq = torch.randn(2, 5, small_config.latent_dim)
        a_seq = torch.randn(2, 5, small_config.action_dim, requires_grad=True)
        energy = model(z_seq, a_seq)
        energy.sum().backward()
        assert a_seq.grad is not None
        assert not torch.isnan(a_seq.grad).any()
        assert not torch.isinf(a_seq.grad).any()

    def test_configurable_layers(self):
        for n_layers in [1, 3, 6]:
            config = CEWMConfig(
                d_model=64, d_state=16, n_layers=n_layers,
                expand_factor=2, mimo_groups=2, action_dim=7,
                latent_dim=32, dropout=0.1,
            )
            model = CausalEnergyWorldModel(config)
            assert len(model.mamba_layers) == n_layers
            z = torch.randn(1, 3, 32)
            a = torch.randn(1, 3, 7)
            energy = model(z, a)
            assert energy.shape == (1,)

    def test_batch_size_one(self, small_config):
        model = CausalEnergyWorldModel(small_config)
        z = torch.randn(1, 3, small_config.latent_dim)
        a = torch.randn(1, 3, small_config.action_dim)
        energy = model(z, a)
        assert energy.shape == (1,)

    def test_sequence_length_one(self, small_config):
        model = CausalEnergyWorldModel(small_config)
        z = torch.randn(2, 1, small_config.latent_dim)
        a = torch.randn(2, 1, small_config.action_dim)
        energy = model(z, a)
        assert energy.shape == (2,)

    def test_production_config_forward(self):
        """生产配置（默认 CEWMConfig）能正常前向传播。"""
        config = CEWMConfig()
        model = CausalEnergyWorldModel(config)
        z = torch.randn(1, 5, config.latent_dim)
        a = torch.randn(1, 5, config.action_dim)
        energy = model(z, a)
        assert energy.shape == (1,)
        assert not torch.isnan(energy).any()
