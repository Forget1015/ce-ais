"""共享测试 fixtures。

提供 mock 观测数据、mock 配置和小规模模型实例，
供所有测试模块统一使用。
"""

import sys
from pathlib import Path

import pytest
import torch

# 确保项目根目录在 sys.path 中
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config.schema import (
    BilateralGatingConfig,
    CEAISConfig,
    CEWMConfig,
    EncoderConfig,
    EvaluationConfig,
    SteeringConfig,
    TrainingConfig,
)
from src.data_structures import (
    ActionChunk,
    EnergyEvaluation,
    EvalMetrics,
    Observation,
    SteeringResult,
    TrajectoryTuple,
)


# ------------------------------------------------------------------
# 配置 fixtures
# ------------------------------------------------------------------


@pytest.fixture
def encoder_config():
    """小规模编码器配置（用于快速测试）。"""
    return EncoderConfig(
        backbone_type="resnet18",
        pose_dim=7,
        visual_dim=512,
        latent_dim=32,
        temperature=0.07,
        image_size=[64, 64],
    )


@pytest.fixture
def cewm_config():
    """小规模 CE-WM 配置（用于快速测试）。"""
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


@pytest.fixture
def steering_config():
    """偏转模块配置。"""
    return SteeringConfig(
        n_steps=3,
        step_size=0.01,
        anneal_rate=0.5,
        noise_scale=0.001,
        kl_weight=10.0,
    )


@pytest.fixture
def gating_config():
    """门控模块配置。"""
    return BilateralGatingConfig(
        lambda_max=1.0,
        sensitivity=0.1,
        window_size=10,
        uncertainty_method="mc_dropout",
        mc_samples=3,
    )


@pytest.fixture
def training_config():
    """训练配置。"""
    return TrainingConfig(
        encoder_epochs=2,
        ce_wm_epochs=2,
        batch_size=4,
        learning_rate=1e-3,
        weight_decay=1e-5,
        amp=False,
        checkpoint_interval=1,
        neg_sample_ratio=2,
    )


@pytest.fixture
def eval_config():
    """评估配置字典。"""
    return {
        "protocol": "ABC_to_D",
        "max_chain_length": 5,
        "ood_perturbations": {
            "physics": {
                "mass_scale": [0.5, 2.0],
                "friction_scale": [0.3, 1.5],
            },
            "visual": {
                "brightness_jitter": 0.5,
                "gaussian_noise_std": 0.1,
                "camera_pose_offset": 0.05,
            },
        },
    }


@pytest.fixture
def full_config(encoder_config, cewm_config, steering_config, gating_config, training_config):
    """完整 CE-AIS 配置。"""
    config = CEAISConfig()
    config.encoder = encoder_config
    config.ce_wm = cewm_config
    config.steering = steering_config
    config.bilateral_gating = gating_config
    config.training = training_config
    return config


# ------------------------------------------------------------------
# 数据 fixtures
# ------------------------------------------------------------------


@pytest.fixture
def batch_size():
    """默认测试批大小。"""
    return 2


@pytest.fixture
def seq_len():
    """默认测试序列长度。"""
    return 4


@pytest.fixture
def mock_observation(batch_size):
    """Mock 多模态观测数据。"""
    return Observation(
        rgb=torch.randn(batch_size, 3, 64, 64),
        depth=torch.randn(batch_size, 1, 64, 64),
        pose=torch.randn(batch_size, 7),
    )


@pytest.fixture
def mock_observation_dict(batch_size):
    """Mock 观测数据（字典格式）。"""
    return {
        "rgb": torch.randn(batch_size, 3, 64, 64),
        "depth": torch.randn(batch_size, 1, 64, 64),
        "pose": torch.randn(batch_size, 7),
    }


@pytest.fixture
def mock_action_chunk(batch_size, seq_len):
    """Mock 动作分块。"""
    return ActionChunk(actions=torch.randn(batch_size, seq_len, 7))


@pytest.fixture
def mock_latent(batch_size):
    """Mock 潜变量。"""
    z = torch.randn(batch_size, 32)
    return torch.nn.functional.normalize(z, dim=-1)


@pytest.fixture
def mock_latent_seq(batch_size, seq_len):
    """Mock 潜变量序列。"""
    return torch.randn(batch_size, seq_len, 32)


@pytest.fixture
def mock_action_seq(batch_size, seq_len):
    """Mock 动作序列。"""
    return torch.randn(batch_size, seq_len, 7)


@pytest.fixture
def mock_trajectory_tuple(mock_latent_seq, mock_action_seq):
    """Mock 轨迹元组（正样本）。"""
    return TrajectoryTuple(
        z_seq=mock_latent_seq,
        a_seq=mock_action_seq,
        is_positive=True,
    )


@pytest.fixture
def mock_energy_evaluation(batch_size):
    """Mock 能量评估结果。"""
    return EnergyEvaluation(
        energy=torch.randn(batch_size),
        uncertainty=torch.rand(batch_size),
        gating_lambda=torch.rand(batch_size),
    )


@pytest.fixture
def mock_steering_result(batch_size, seq_len):
    """Mock 偏转结果。"""
    return SteeringResult(
        action_before=torch.randn(batch_size, seq_len, 7),
        action_after=torch.randn(batch_size, seq_len, 7),
        energy_before=1.5,
        energy_after=0.8,
        uncertainty=0.3,
        gating_lambda=0.9,
        n_steps=5,
    )


@pytest.fixture
def mock_eval_metrics():
    """Mock 评估指标。"""
    return EvalMetrics(
        chain_success_rate={1: 0.9, 2: 0.7, 3: 0.5},
        single_task_rate={"open_drawer": 0.85, "close_drawer": 0.90},
        avg_steps=150.0,
        transient_recovery_time=25.0,
        trajectory_jerk=0.05,
        latency_ms=12.5,
    )


# ------------------------------------------------------------------
# 模型 fixtures
# ------------------------------------------------------------------


@pytest.fixture
def small_encoder(encoder_config):
    """小规模编码器实例。"""
    from src.encoders.contrastive_encoder import ContrastiveEncoder

    return ContrastiveEncoder(encoder_config)


@pytest.fixture
def small_cewm(cewm_config):
    """小规模 CE-WM 实例。"""
    from src.world_model.ce_wm import CausalEnergyWorldModel

    return CausalEnergyWorldModel(cewm_config)


@pytest.fixture
def steering_module(steering_config):
    """偏转模块实例。"""
    from src.steering.efe_steering import EFESteering

    return EFESteering(steering_config)


@pytest.fixture
def gating_module(gating_config):
    """门控模块实例。"""
    from src.steering.bilateral_gating import BilateralGating

    return BilateralGating(gating_config)
