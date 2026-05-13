"""物理指标计算属性测试。

Feature: ce-ais-framework
Property 12: 物理指标计算正确性
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from src.evaluation.metrics import MetricsModule


# ======================================================================
# Property 12: 物理指标计算正确性 — Jerk
# ======================================================================


@settings(max_examples=100)
@given(
    n_steps=st.integers(min_value=4, max_value=200),
    n_dims=st.sampled_from([3, 6, 7]),
)
def test_trajectory_jerk_is_finite(n_steps, n_dims):
    """Jerk = 三阶数值差分，结果应有限。

    Feature: ce-ais-framework, Property 12: 物理指标计算正确性
    """
    # 生成随机轨迹
    positions = np.random.randn(n_steps, n_dims).astype(np.float64)

    jerk = MetricsModule.compute_trajectory_jerk(positions, dt=1.0)

    assert np.isfinite(jerk), f"Jerk is not finite: {jerk}"
    assert jerk >= 0, f"Jerk should be non-negative: {jerk}"


@settings(max_examples=100)
@given(
    n_steps=st.integers(min_value=4, max_value=100),
)
def test_trajectory_jerk_zero_for_linear(n_steps):
    """线性轨迹的 Jerk 应为 0（或接近 0）。

    Feature: ce-ais-framework, Property 12: 物理指标计算正确性
    """
    # 线性轨迹: 速度恒定 → 加速度为 0 → Jerk 为 0
    t = np.linspace(0, 1, n_steps).reshape(-1, 1)
    positions = np.hstack([t, 2 * t, 3 * t])  # [T, 3]

    jerk = MetricsModule.compute_trajectory_jerk(positions, dt=1.0)

    assert np.isclose(jerk, 0.0, atol=1e-8), (
        f"Jerk for linear trajectory should be ~0, got {jerk}"
    )


def test_trajectory_jerk_short_sequence():
    """长度 < 4 的轨迹 Jerk 应返回 0。

    Feature: ce-ais-framework, Property 12: 物理指标计算正确性
    """
    positions = np.random.randn(3, 3)
    jerk = MetricsModule.compute_trajectory_jerk(positions)
    assert jerk == 0.0


# ======================================================================
# Property 12: 物理指标计算正确性 — 瞬态恢复时间
# ======================================================================


@settings(max_examples=100)
@given(
    pre_length=st.integers(min_value=0, max_value=20),
    recovery_delay=st.integers(min_value=0, max_value=50),
    window_size=st.integers(min_value=2, max_value=20),
)
def test_transient_recovery_time_non_negative(
    pre_length, recovery_delay, window_size
):
    """瞬态恢复时间应为非负整数。

    Feature: ce-ais-framework, Property 12: 物理指标计算正确性
    """
    # 构造成功历史: 干扰前全成功，干扰后先失败再恢复
    history = [True] * pre_length
    history += [False] * recovery_delay
    history += [True] * (window_size + 10)

    perturbation_step = pre_length

    recovery_time = MetricsModule.compute_transient_recovery_time(
        success_history=history,
        perturbation_step=perturbation_step,
        window_size=window_size,
        threshold=0.5,
    )

    assert isinstance(recovery_time, int), (
        f"Recovery time should be int, got {type(recovery_time)}"
    )
    assert recovery_time >= 0, (
        f"Recovery time should be non-negative, got {recovery_time}"
    )


@settings(max_examples=100)
@given(
    window_size=st.integers(min_value=2, max_value=10),
)
def test_transient_recovery_immediate_if_no_perturbation(window_size):
    """如果干扰后立即恢复，恢复时间应为 0。

    Feature: ce-ais-framework, Property 12: 物理指标计算正确性
    """
    # 全部成功
    history = [True] * (window_size + 20)
    perturbation_step = 5

    recovery_time = MetricsModule.compute_transient_recovery_time(
        success_history=history,
        perturbation_step=perturbation_step,
        window_size=window_size,
        threshold=0.5,
    )

    assert recovery_time == 0, (
        f"Recovery time should be 0 for all-success history, got {recovery_time}"
    )
