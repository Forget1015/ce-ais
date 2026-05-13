"""配置数据类（dataclass）定义，用于类型校验。"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ProjectConfig:
    """项目全局设置。"""
    name: str = "ce-ais"
    seed: int = 42
    device: str = "cuda:0"


@dataclass
class EncoderConfig:
    """对比预训练编码器配置。"""
    backbone_type: str = "resnet18"
    pose_dim: int = 7
    visual_dim: int = 512
    latent_dim: int = 128
    temperature: float = 0.07
    image_size: List[int] = field(default_factory=lambda: [200, 200])


@dataclass
class CEWMConfig:
    """Mamba-3 因果能量世界模型配置。"""
    d_model: int = 640
    d_state: int = 64
    n_layers: int = 32
    expand_factor: int = 3
    mimo_groups: int = 4
    action_dim: int = 7
    latent_dim: int = 128
    dropout: float = 0.1
    headdim: int = 64


@dataclass
class SteeringConfig:
    """EFE 引导偏转配置。"""
    n_steps: int = 5
    step_size: float = 0.01
    anneal_rate: float = 0.5
    noise_scale: float = 0.001
    kl_weight: float = 10.0
    grad_mode: str = "finite_diff"


@dataclass
class BilateralGatingConfig:
    """双向不确定性门控配置。"""
    lambda_max: float = 1.0
    sensitivity: float = 0.1
    window_size: int = 50
    uncertainty_method: str = "mc_dropout"
    mc_samples: int = 5


@dataclass
class TrainingConfig:
    """训练超参数配置。"""
    encoder_epochs: int = 100
    ce_wm_epochs: int = 200
    batch_size: int = 256
    encoder_batch_size: Optional[int] = None
    cewm_batch_size: Optional[int] = None
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    amp: bool = True
    checkpoint_interval: int = 10
    neg_sample_ratio: int = 5


@dataclass
class OODPhysicsConfig:
    """物理 OOD 摄动配置。"""
    mass_scale: List[float] = field(default_factory=lambda: [0.5, 2.0])
    friction_scale: List[float] = field(default_factory=lambda: [0.3, 1.5])


@dataclass
class OODVisualConfig:
    """视觉 OOD 摄动配置。"""
    brightness_jitter: float = 0.5
    gaussian_noise_std: float = 0.1
    camera_pose_offset: float = 0.05


@dataclass
class OODPerturbationsConfig:
    """OOD 摄动配置。"""
    physics: OODPhysicsConfig = field(default_factory=OODPhysicsConfig)
    visual: OODVisualConfig = field(default_factory=OODVisualConfig)


@dataclass
class EvaluationConfig:
    """评估参数配置。"""
    protocol: str = "ABC_to_D"
    max_chain_length: int = 5
    ood_perturbations: OODPerturbationsConfig = field(
        default_factory=OODPerturbationsConfig
    )


@dataclass
class LoggingConfig:
    """日志与实验追踪配置。"""
    tensorboard: bool = True
    wandb: bool = False
    log_interval: int = 10


@dataclass
class CEAISConfig:
    """CE-AIS 完整配置。"""
    project: ProjectConfig = field(default_factory=ProjectConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    ce_wm: CEWMConfig = field(default_factory=CEWMConfig)
    steering: SteeringConfig = field(default_factory=SteeringConfig)
    bilateral_gating: BilateralGatingConfig = field(
        default_factory=BilateralGatingConfig
    )
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
