"""摄动策略属性测试。

Feature: ce-ais-framework
Property 6: 对抗摄动改变动作
Property 7: 正负样本批次比例
Property 8: 摄动策略注册与检索
"""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from src.data.perturbation import PerturbationRegistry


# ======================================================================
# Property 6: 对抗摄动改变动作 — 应用摄动后动作应与原始不同
# ======================================================================


@settings(max_examples=100)
@given(
    batch_size=st.integers(min_value=1, max_value=8),
    action_dim=st.integers(min_value=3, max_value=10),
)
def test_velocity_reversal_changes_action(batch_size, action_dim):
    """velocity_reversal 摄动后动作应与原始不同。

    Feature: ce-ais-framework, Property 6: 对抗摄动改变动作
    """
    # 确保前 3 维非零
    action = torch.randn(batch_size, action_dim)
    action[:, :3] = action[:, :3].clamp(min=0.1)  # 确保非零

    perturbed = PerturbationRegistry.apply("velocity_reversal", action)

    assert not torch.equal(action, perturbed), (
        "Perturbed action should differ from original"
    )
    assert perturbed.shape == action.shape, (
        f"Shape mismatch: {perturbed.shape} != {action.shape}"
    )


@settings(max_examples=100)
@given(
    batch_size=st.integers(min_value=1, max_value=8),
    action_dim=st.integers(min_value=3, max_value=10),
)
def test_random_displacement_changes_action(batch_size, action_dim):
    """random_displacement 摄动后动作应与原始不同。

    Feature: ce-ais-framework, Property 6: 对抗摄动改变动作
    """
    action = torch.randn(batch_size, action_dim)

    perturbed = PerturbationRegistry.apply(
        "random_displacement", action, scale=0.5
    )

    assert not torch.equal(action, perturbed), (
        "Perturbed action should differ from original"
    )
    assert perturbed.shape == action.shape


@settings(max_examples=100)
@given(
    batch_size=st.integers(min_value=1, max_value=8),
    seq_len=st.integers(min_value=1, max_value=8),
    action_dim=st.integers(min_value=3, max_value=10),
)
def test_gripper_anomaly_changes_action(batch_size, seq_len, action_dim):
    """gripper_anomaly 摄动后动作应与原始不同（3D 输入）。

    Feature: ce-ais-framework, Property 6: 对抗摄动改变动作
    """
    action = torch.randn(batch_size, seq_len, action_dim)

    perturbed = PerturbationRegistry.apply("gripper_anomaly", action)

    # 最后一维应被修改
    assert perturbed.shape == action.shape
    # 由于随机替换，几乎不可能完全相同
    assert not torch.equal(action[..., -1], perturbed[..., -1]), (
        "Gripper dimension should be modified"
    )


# ======================================================================
# Property 8: 摄动策略注册与检索
# ======================================================================


@settings(max_examples=100)
@given(
    strategy_name=st.sampled_from(
        ["velocity_reversal", "gripper_anomaly", "random_displacement"]
    ),
    batch_size=st.integers(min_value=1, max_value=8),
    action_dim=st.integers(min_value=3, max_value=10),
)
def test_perturbation_registry_retrieval(strategy_name, batch_size, action_dim):
    """注册后应能正确检索摄动策略，返回同形状张量。

    Feature: ce-ais-framework, Property 8: 摄动策略注册与检索
    """
    # 策略应可检索
    fn = PerturbationRegistry.get(strategy_name)
    assert callable(fn), f"Strategy '{strategy_name}' is not callable"

    # 应用策略应返回同形状张量
    action = torch.randn(batch_size, action_dim)
    perturbed = PerturbationRegistry.apply(strategy_name, action)

    assert perturbed.shape == action.shape, (
        f"Shape mismatch for '{strategy_name}': "
        f"{perturbed.shape} != {action.shape}"
    )
    assert isinstance(perturbed, torch.Tensor)


def test_perturbation_registry_lists_all_strategies():
    """注册表应列出所有内置策略。

    Feature: ce-ais-framework, Property 8: 摄动策略注册与检索
    """
    strategies = PerturbationRegistry.list_strategies()
    assert "velocity_reversal" in strategies
    assert "gripper_anomaly" in strategies
    assert "random_displacement" in strategies


def test_perturbation_registry_unknown_raises():
    """检索未注册策略应抛出 KeyError。

    Feature: ce-ais-framework, Property 8: 摄动策略注册与检索
    """
    with pytest.raises(KeyError):
        PerturbationRegistry.get("nonexistent_strategy")


# ======================================================================
# Property 7: 正负样本批次比例 — 负样本数量应等于正样本数量乘以 K
# ======================================================================


@settings(max_examples=100)
@given(
    batch_size=st.integers(min_value=1, max_value=16),
    neg_ratio=st.integers(min_value=1, max_value=10),
    action_dim=st.integers(min_value=3, max_value=10),
)
def test_negative_sample_ratio(batch_size, neg_ratio, action_dim):
    """负样本数量应等于正样本数量乘以 K。

    Feature: ce-ais-framework, Property 7: 正负样本批次比例
    """
    # 模拟正样本
    a_pos = torch.randn(batch_size, action_dim)

    # 为每个正样本生成 K 个负样本
    neg_samples = []
    strategies = PerturbationRegistry.list_strategies()

    for i in range(neg_ratio):
        strategy = strategies[i % len(strategies)]
        a_neg = PerturbationRegistry.apply(strategy, a_pos)
        neg_samples.append(a_neg)

    # 堆叠为 [B, K, d_a]
    a_neg_batch = torch.stack(neg_samples, dim=1)

    # 验证比例
    assert a_neg_batch.shape[0] == batch_size, (
        f"Batch size mismatch: {a_neg_batch.shape[0]} != {batch_size}"
    )
    assert a_neg_batch.shape[1] == neg_ratio, (
        f"Negative ratio mismatch: {a_neg_batch.shape[1]} != {neg_ratio}"
    )
    assert a_neg_batch.shape[2] == action_dim, (
        f"Action dim mismatch: {a_neg_batch.shape[2]} != {action_dim}"
    )

    # 负样本总数 = B * K
    total_neg = a_neg_batch.shape[0] * a_neg_batch.shape[1]
    total_pos = batch_size
    assert total_neg == total_pos * neg_ratio, (
        f"Total negatives ({total_neg}) != "
        f"total positives ({total_pos}) * K ({neg_ratio})"
    )
