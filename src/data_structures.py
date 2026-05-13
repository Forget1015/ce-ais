"""CE-AIS 核心数据结构定义。

集中定义设计文档中的所有核心数据类，供各模块统一引用。
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch


@dataclass
class Observation:
    """多模态观测数据。"""

    rgb: torch.Tensor       # [B, 3, H, W] RGB 图像
    depth: torch.Tensor     # [B, 1, H, W] 深度图
    pose: torch.Tensor      # [B, d_p] 末端执行器姿态 (xyz + quaternion)


@dataclass
class ActionChunk:
    """动作分块。"""

    actions: torch.Tensor   # [B, T, d_a] T 步连续动作序列
    # d_a = 7: (dx, dy, dz, droll, dpitch, dyaw, gripper)


@dataclass
class EnergyEvaluation:
    """能量评估结果。"""

    energy: torch.Tensor        # [B] 标量能量值
    uncertainty: torch.Tensor   # [B] 认知不确定性
    gating_lambda: torch.Tensor # [B] 门控引导强度


@dataclass
class TrajectoryTuple:
    """状态-动作轨迹元组（用于能量景观训练）。"""

    z_seq: torch.Tensor     # [B, T, d_z] 潜变量序列
    a_seq: torch.Tensor     # [B, T, d_a] 动作序列
    is_positive: bool       # 正样本（专家）/ 负样本（对抗摄动）


@dataclass
class SteeringResult:
    """偏转结果。"""

    action_before: torch.Tensor   # [B, T, d_a] 偏转前动作
    action_after: torch.Tensor    # [B, T, d_a] 偏转后动作
    energy_before: float          # 偏转前能量
    energy_after: float           # 偏转后能量
    uncertainty: float            # 认知不确定性
    gating_lambda: float          # 门控强度
    n_steps: int                  # 实际迭代步数


@dataclass
class EvalMetrics:
    """评估指标。"""

    chain_success_rate: Dict[int, float] = field(default_factory=dict)
    single_task_rate: Dict[str, float] = field(default_factory=dict)
    avg_steps: float = 0.0
    transient_recovery_time: float = 0.0
    trajectory_jerk: float = 0.0
    latency_ms: float = 0.0
