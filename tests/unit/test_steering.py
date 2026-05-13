"""EFE 偏转模块单元测试。

测试:
- 朗之万动力学单步更新
- EFESteering 完整偏转流程
- 输出形状一致性
- 参数冻结约束
- 退火温度调度
"""

import torch
import pytest

from src.config.schema import SteeringConfig, CEWMConfig
from src.steering.langevin import annealed_langevin_step, run_langevin_dynamics
from src.steering.efe_steering import EFESteering
from src.world_model.ce_wm import CausalEnergyWorldModel


@pytest.fixture
def steering_config():
    return SteeringConfig(
        n_steps=3,
        step_size=0.01,
        anneal_rate=0.5,
        noise_scale=0.001,
        kl_weight=10.0,
    )


@pytest.fixture
def small_cewm():
    """小规模 CE-WM 用于测试。"""
    config = CEWMConfig(
        d_model=32,
        d_state=8,
        n_layers=1,
        expand_factor=2,
        mimo_groups=2,
        action_dim=7,
        latent_dim=16,
        dropout=0.1,
    )
    model = CausalEnergyWorldModel(config)
    model.eval()
    return model


class TestAnnealedLangevinStep:
    """退火朗之万动力学单步更新测试。"""

    def test_output_shape_matches_input(self):
        """输出形状应与输入一致。"""
        B, T, d_a = 4, 5, 7
        action = torch.randn(B, T, d_a)
        energy_grad = torch.randn(B, T, d_a)
        action_init = torch.randn(B, T, d_a)
        gating_lambda = torch.ones(B)

        result = annealed_langevin_step(
            action=action,
            energy_grad=energy_grad,
            action_init=action_init,
            step_size=0.01,
            anneal_rate=0.5,
            step_idx=0,
            noise_scale=0.001,
            kl_weight=10.0,
            gating_lambda=gating_lambda,
        )

        assert result.shape == (B, T, d_a)

    def test_zero_gradient_minimal_change(self):
        """零梯度时，更新应主要由 KL 项和噪声驱动。"""
        B, T, d_a = 2, 3, 7
        action = torch.randn(B, T, d_a)
        action_init = action.clone()
        energy_grad = torch.zeros(B, T, d_a)
        gating_lambda = torch.ones(B)

        result = annealed_langevin_step(
            action=action,
            energy_grad=energy_grad,
            action_init=action_init,
            step_size=0.01,
            anneal_rate=0.5,
            step_idx=0,
            noise_scale=0.0,  # 无噪声
            kl_weight=10.0,
            gating_lambda=gating_lambda,
        )

        # action == action_init 时 KL 梯度为 0，所以结果应接近原始
        assert torch.allclose(result, action, atol=1e-6)

    def test_annealing_reduces_step_size(self):
        """退火应随迭代步数减小有效步长。"""
        B, T, d_a = 2, 3, 7
        action = torch.randn(B, T, d_a)
        energy_grad = torch.ones(B, T, d_a)
        action_init = torch.zeros(B, T, d_a)
        gating_lambda = torch.ones(B)

        # 步骤 0 vs 步骤 5 的更新幅度
        result_step0 = annealed_langevin_step(
            action=action, energy_grad=energy_grad,
            action_init=action_init, step_size=0.1,
            anneal_rate=0.5, step_idx=0, noise_scale=0.0,
            kl_weight=0.0, gating_lambda=gating_lambda,
        )
        result_step5 = annealed_langevin_step(
            action=action, energy_grad=energy_grad,
            action_init=action_init, step_size=0.1,
            anneal_rate=0.5, step_idx=5, noise_scale=0.0,
            kl_weight=0.0, gating_lambda=gating_lambda,
        )

        diff0 = (result_step0 - action).abs().mean()
        diff5 = (result_step5 - action).abs().mean()
        assert diff0 > diff5, "Later steps should have smaller updates"

    def test_gating_lambda_zero_no_update(self):
        """门控强度为 0 时，不应有梯度驱动的更新。"""
        B, T, d_a = 2, 3, 7
        action = torch.randn(B, T, d_a)
        energy_grad = torch.ones(B, T, d_a)
        action_init = torch.zeros(B, T, d_a)
        gating_lambda = torch.zeros(B)

        result = annealed_langevin_step(
            action=action, energy_grad=energy_grad,
            action_init=action_init, step_size=0.1,
            anneal_rate=0.5, step_idx=0, noise_scale=0.0,
            kl_weight=10.0, gating_lambda=gating_lambda,
        )

        # λ=0 且无噪声时，结果应等于原始动作
        assert torch.allclose(result, action, atol=1e-6)


class TestEFESteering:
    """EFESteering 完整偏转流程测试。"""

    def test_steer_output_shape(self, steering_config, small_cewm):
        """偏转输出形状应与输入一致。"""
        steering = EFESteering(steering_config)
        B, T, d_a, d_z = 2, 3, 7, 16

        a_init = torch.randn(B, T, d_a)
        z_t = torch.randn(B, d_z)
        gating_lambda = torch.ones(B)

        a_star = steering.steer(a_init, z_t, small_cewm, gating_lambda)

        assert a_star.shape == (B, T, d_a)

    def test_steer_does_not_modify_model_params(self, steering_config, small_cewm):
        """偏转过程不应修改模型参数。"""
        steering = EFESteering(steering_config)
        B, T, d_a, d_z = 2, 3, 7, 16

        # 记录偏转前的参数
        params_before = {
            name: p.clone() for name, p in small_cewm.named_parameters()
        }

        a_init = torch.randn(B, T, d_a)
        z_t = torch.randn(B, d_z)
        gating_lambda = torch.ones(B)

        steering.steer(a_init, z_t, small_cewm, gating_lambda)

        # 验证参数未变
        for name, p in small_cewm.named_parameters():
            assert torch.equal(p, params_before[name]), (
                f"Parameter {name} was modified during steering"
            )

    def test_steer_output_is_finite(self, steering_config, small_cewm):
        """偏转输出应为有限值。"""
        steering = EFESteering(steering_config)
        B, T, d_a, d_z = 2, 3, 7, 16

        a_init = torch.randn(B, T, d_a)
        z_t = torch.randn(B, d_z)
        gating_lambda = torch.ones(B)

        a_star = steering.steer(a_init, z_t, small_cewm, gating_lambda)

        assert torch.isfinite(a_star).all(), "Steering output contains NaN/Inf"

    def test_steer_modifies_action(self, steering_config, small_cewm):
        """偏转应实际修改动作（非恒等变换）。"""
        # 使用较大步长确保有可见变化
        config = SteeringConfig(
            n_steps=3,
            step_size=0.1,
            anneal_rate=0.5,
            noise_scale=0.0,
            kl_weight=0.1,
        )
        steering = EFESteering(config)
        B, T, d_a, d_z = 2, 3, 7, 16

        a_init = torch.randn(B, T, d_a)
        z_t = torch.randn(B, d_z)
        gating_lambda = torch.ones(B)

        a_star = steering.steer(a_init, z_t, small_cewm, gating_lambda)

        # 偏转后动作应与原始不同
        assert not torch.allclose(a_star, a_init, atol=1e-6), (
            "Steering did not modify the action"
        )

    def test_forward_calls_steer(self, steering_config, small_cewm):
        """forward() 应等价于 steer()。"""
        steering = EFESteering(steering_config)
        B, T, d_a, d_z = 2, 3, 7, 16

        a_init = torch.randn(B, T, d_a)
        z_t = torch.randn(B, d_z)
        gating_lambda = torch.ones(B)

        torch.manual_seed(42)
        result_steer = steering.steer(a_init, z_t, small_cewm, gating_lambda)

        torch.manual_seed(42)
        result_forward = steering.forward(a_init, z_t, small_cewm, gating_lambda)

        assert torch.allclose(result_steer, result_forward, atol=1e-5)


class TestRunLangevinDynamics:
    """run_langevin_dynamics 完整迭代测试。"""

    def test_output_shape(self, small_cewm):
        """输出形状应与输入一致。"""
        B, T, d_a, d_z = 2, 3, 7, 16
        a_init = torch.randn(B, T, d_a)
        z_t = torch.randn(B, d_z)
        gating_lambda = torch.ones(B)

        result = run_langevin_dynamics(
            action_init=a_init,
            z_t=z_t,
            energy_fn=small_cewm,
            n_steps=3,
            step_size=0.01,
            anneal_rate=0.5,
            noise_scale=0.001,
            kl_weight=10.0,
            gating_lambda=gating_lambda,
        )

        assert result.shape == (B, T, d_a)

    def test_output_detached(self, small_cewm):
        """输出应已 detach，不需要梯度。"""
        B, T, d_a, d_z = 2, 3, 7, 16
        a_init = torch.randn(B, T, d_a)
        z_t = torch.randn(B, d_z)
        gating_lambda = torch.ones(B)

        result = run_langevin_dynamics(
            action_init=a_init,
            z_t=z_t,
            energy_fn=small_cewm,
            n_steps=2,
            step_size=0.01,
            anneal_rate=0.5,
            noise_scale=0.001,
            kl_weight=10.0,
            gating_lambda=gating_lambda,
        )

        assert not result.requires_grad
