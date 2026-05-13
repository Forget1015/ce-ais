"""对比损失数学性质属性测试。

Feature: ce-ais-framework
Property 2: 对比损失数学性质
"""

import pytest
import torch
import torch.nn.functional as F
from hypothesis import given, settings
from hypothesis import strategies as st

from src.training.losses import InfoNCELoss, NCELoss


# ======================================================================
# Property 2: InfoNCE / NCE 损失值应为有限正数
# ======================================================================


@settings(max_examples=100)
@given(
    batch_size=st.integers(min_value=2, max_value=16),
    latent_dim=st.sampled_from([32, 64, 128]),
    temperature=st.floats(min_value=0.01, max_value=1.0),
)
def test_infonce_loss_finite_positive(batch_size, latent_dim, temperature):
    """InfoNCE 损失值应为有限正数。

    Feature: ce-ais-framework, Property 2: 对比损失数学性质
    """
    loss_fn = InfoNCELoss(temperature=temperature)

    z_anchor = F.normalize(torch.randn(batch_size, latent_dim), dim=-1)
    z_positive = F.normalize(torch.randn(batch_size, latent_dim), dim=-1)

    loss = loss_fn(z_anchor, z_positive)

    assert torch.isfinite(loss), f"InfoNCE loss is not finite: {loss.item()}"
    assert loss.item() > 0, f"InfoNCE loss is not positive: {loss.item()}"


@settings(max_examples=100)
@given(
    batch_size=st.integers(min_value=2, max_value=16),
    n_neg=st.integers(min_value=1, max_value=10),
    temperature=st.floats(min_value=0.1, max_value=2.0),
)
def test_nce_loss_finite_positive(batch_size, n_neg, temperature):
    """NCE 损失值应为有限正数。

    Feature: ce-ais-framework, Property 2: 对比损失数学性质
    """
    loss_fn = NCELoss(temperature=temperature)

    energy_pos = torch.randn(batch_size)
    energy_neg = torch.randn(batch_size, n_neg)

    loss = loss_fn(energy_pos, energy_neg)

    assert torch.isfinite(loss), f"NCE loss is not finite: {loss.item()}"
    assert loss.item() > 0, f"NCE loss is not positive: {loss.item()}"


# ======================================================================
# Property 2: 损失单调递减性
# ======================================================================


@settings(max_examples=100)
@given(
    batch_size=st.integers(min_value=2, max_value=8),
    latent_dim=st.sampled_from([32, 64, 128]),
)
def test_infonce_loss_decreases_with_similarity(batch_size, latent_dim):
    """当正样本相似度趋近1且负样本相似度趋近-1时，InfoNCE 损失应单调递减。

    Feature: ce-ais-framework, Property 2: 对比损失数学性质
    """
    loss_fn = InfoNCELoss(temperature=0.07)

    # 场景1: 随机嵌入（低相似度）
    z_anchor = F.normalize(torch.randn(batch_size, latent_dim), dim=-1)
    z_random = F.normalize(torch.randn(batch_size, latent_dim), dim=-1)
    loss_random = loss_fn(z_anchor, z_random)

    # 场景2: 正样本 = 锚点 + 小噪声（高相似度）
    z_similar = F.normalize(z_anchor + 0.01 * torch.randn_like(z_anchor), dim=-1)
    loss_similar = loss_fn(z_anchor, z_similar)

    # 高相似度时损失应更低
    assert loss_similar.item() <= loss_random.item() + 1e-3, (
        f"Loss with similar pairs ({loss_similar.item():.4f}) should be <= "
        f"loss with random pairs ({loss_random.item():.4f})"
    )


@settings(max_examples=100)
@given(
    batch_size=st.integers(min_value=2, max_value=8),
    n_neg=st.integers(min_value=2, max_value=8),
)
def test_nce_loss_decreases_with_energy_gap(batch_size, n_neg):
    """当正样本能量低、负样本能量高时，NCE 损失应更低。

    Feature: ce-ais-framework, Property 2: 对比损失数学性质
    """
    loss_fn = NCELoss(temperature=1.0)

    # 场景1: 小能量差距
    energy_pos_small = torch.zeros(batch_size)
    energy_neg_small = torch.ones(batch_size, n_neg) * 0.5
    loss_small_gap = loss_fn(energy_pos_small, energy_neg_small)

    # 场景2: 大能量差距（正样本低，负样本高）
    energy_pos_large = torch.zeros(batch_size) - 5.0
    energy_neg_large = torch.ones(batch_size, n_neg) * 5.0
    loss_large_gap = loss_fn(energy_pos_large, energy_neg_large)

    # 大能量差距时损失应更低
    assert loss_large_gap.item() <= loss_small_gap.item() + 1e-3, (
        f"Loss with large gap ({loss_large_gap.item():.4f}) should be <= "
        f"loss with small gap ({loss_small_gap.item():.4f})"
    )
