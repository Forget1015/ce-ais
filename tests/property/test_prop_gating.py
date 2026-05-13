"""高斯门控函数行为属性测试。

Feature: ce-ais-framework
Property 11: 高斯门控函数行为
"""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from src.config.schema import BilateralGatingConfig
from src.steering.bilateral_gating import BilateralGating


# ======================================================================
# Property 11: 高斯门控函数行为
# ======================================================================


@settings(max_examples=100)
@given(
    lambda_max=st.floats(min_value=0.1, max_value=5.0),
    sensitivity=st.floats(min_value=0.01, max_value=1.0),
    uncertainty_val=st.floats(min_value=-10.0, max_value=10.0),
)
def test_gating_lambda_bounded(lambda_max, sensitivity, uncertainty_val):
    """0 ≤ λ ≤ λ_max。

    Feature: ce-ais-framework, Property 11: 高斯门控函数行为
    """
    config = BilateralGatingConfig(
        lambda_max=lambda_max,
        sensitivity=sensitivity,
        window_size=10,
    )
    gating = BilateralGating(config)

    uncertainty = torch.tensor([uncertainty_val])
    lam = gating.compute_lambda(uncertainty)

    assert (lam >= 0).all(), f"Lambda {lam.item()} < 0"
    assert (lam <= lambda_max + 1e-6).all(), (
        f"Lambda {lam.item()} > lambda_max {lambda_max}"
    )


@settings(max_examples=100)
@given(
    lambda_max=st.floats(min_value=0.5, max_value=3.0),
    sensitivity=st.floats(min_value=0.01, max_value=0.5),
)
def test_gating_lambda_max_at_mean(lambda_max, sensitivity):
    """u_t = μ_u 时 λ = λ_max。

    Feature: ce-ais-framework, Property 11: 高斯门控函数行为
    """
    config = BilateralGatingConfig(
        lambda_max=lambda_max,
        sensitivity=sensitivity,
        window_size=10,
    )
    gating = BilateralGating(config)

    # 先填充历史以建立稳定均值
    mu_u = 0.5
    for _ in range(10):
        gating.compute_lambda(torch.tensor([mu_u]))

    # 当 u_t = μ_u 时，λ 应等于 λ_max
    lam = gating.compute_lambda(torch.tensor([mu_u]))

    assert torch.allclose(lam, torch.tensor([lambda_max]), atol=1e-4), (
        f"Lambda {lam.item()} != lambda_max {lambda_max} when u_t = mu_u"
    )


@settings(max_examples=100)
@given(
    lambda_max=st.floats(min_value=0.5, max_value=3.0),
    sensitivity=st.floats(min_value=0.05, max_value=0.5),
    delta=st.floats(min_value=0.1, max_value=5.0),
)
def test_gating_lambda_decreases_with_distance(lambda_max, sensitivity, delta):
    """|u_t - μ_u| 增大时 λ 单调递减。

    Feature: ce-ais-framework, Property 11: 高斯门控函数行为
    """
    config = BilateralGatingConfig(
        lambda_max=lambda_max,
        sensitivity=sensitivity,
        window_size=100,
    )
    gating = BilateralGating(config)

    # 建立稳定均值
    mu_u = 1.0
    for _ in range(100):
        gating.compute_lambda(torch.tensor([mu_u]))

    # 在均值处计算 λ
    lam_at_mean = gating.compute_lambda(torch.tensor([mu_u]))

    # 重新建立均值（因为 compute_lambda 会更新历史）
    gating.reset()
    for _ in range(100):
        gating.compute_lambda(torch.tensor([mu_u]))

    # 在偏离均值处计算 λ
    lam_far = gating.compute_lambda(torch.tensor([mu_u + delta]))

    assert lam_far.item() <= lam_at_mean.item() + 1e-4, (
        f"Lambda at distance {delta} ({lam_far.item():.4f}) should be <= "
        f"lambda at mean ({lam_at_mean.item():.4f})"
    )
