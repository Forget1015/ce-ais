"""损失函数单元测试：InfoNCELoss + NCELoss。"""

import pytest
import torch

from src.training.losses import InfoNCELoss, NCELoss


class TestInfoNCELoss:
    def test_output_is_finite_positive(self):
        loss_fn = InfoNCELoss(temperature=0.07)
        z_a = torch.randn(8, 128)
        z_p = torch.randn(8, 128)
        loss = loss_fn(z_a, z_p)
        assert loss.item() > 0
        assert torch.isfinite(loss)

    def test_identical_pairs_low_loss(self):
        """正样本对完全相同时，损失应较低。"""
        loss_fn = InfoNCELoss(temperature=0.07)
        z = torch.randn(8, 128)
        z = torch.nn.functional.normalize(z, dim=-1)
        loss = loss_fn(z, z)
        # 对角线相似度为 1/τ，远大于其他项，损失应接近 0
        assert loss.item() < 1.0

    def test_loss_decreases_with_better_alignment(self):
        """正样本对越对齐，损失越低。"""
        loss_fn = InfoNCELoss(temperature=0.5)
        z = torch.nn.functional.normalize(torch.randn(16, 64), dim=-1)

        # 完全对齐
        loss_aligned = loss_fn(z, z)
        # 随机负样本
        z_random = torch.nn.functional.normalize(torch.randn(16, 64), dim=-1)
        loss_random = loss_fn(z, z_random)

        assert loss_aligned.item() < loss_random.item()

    def test_invalid_temperature_raises(self):
        with pytest.raises(ValueError, match="Temperature must be positive"):
            InfoNCELoss(temperature=0.0)
        with pytest.raises(ValueError, match="Temperature must be positive"):
            InfoNCELoss(temperature=-1.0)

    def test_gradient_flows(self):
        loss_fn = InfoNCELoss(temperature=0.07)
        z_a = torch.randn(4, 64, requires_grad=True)
        z_p = torch.randn(4, 64, requires_grad=True)
        loss = loss_fn(z_a, z_p)
        loss.backward()
        assert z_a.grad is not None
        assert z_p.grad is not None


class TestNCELoss:
    def test_output_is_finite_positive(self):
        loss_fn = NCELoss(temperature=1.0)
        e_pos = torch.randn(8)
        e_neg = torch.randn(8, 5)
        loss = loss_fn(e_pos, e_neg)
        assert loss.item() > 0
        assert torch.isfinite(loss)

    def test_low_pos_high_neg_gives_low_loss(self):
        """正样本低能量、负样本高能量时，损失应较低。"""
        loss_fn = NCELoss(temperature=1.0)
        e_pos = torch.full((8,), -5.0)  # 低能量
        e_neg = torch.full((8, 5), 5.0)  # 高能量
        loss_good = loss_fn(e_pos, e_neg)

        # 反转：正样本高能量、负样本低能量
        loss_bad = loss_fn(-e_pos, -e_neg)

        assert loss_good.item() < loss_bad.item()

    def test_swapped_energies_increase_loss(self):
        """正负样本能量互换时，损失应增大。"""
        loss_fn = NCELoss(temperature=1.0)
        e_pos = torch.tensor([0.1, 0.2, 0.3])
        e_neg = torch.tensor([[2.0, 3.0], [2.5, 3.5], [2.0, 4.0]])
        loss_normal = loss_fn(e_pos, e_neg)

        # 互换：用负样本能量的均值作为正样本
        e_pos_swap = e_neg.mean(dim=1)
        e_neg_swap = e_pos.unsqueeze(1).expand_as(e_neg)
        loss_swapped = loss_fn(e_pos_swap, e_neg_swap)

        assert loss_normal.item() < loss_swapped.item()

    def test_invalid_temperature_raises(self):
        with pytest.raises(ValueError, match="Temperature must be positive"):
            NCELoss(temperature=0.0)

    def test_gradient_flows(self):
        loss_fn = NCELoss(temperature=1.0)
        e_pos = torch.randn(4, requires_grad=True)
        e_neg = torch.randn(4, 3, requires_grad=True)
        loss = loss_fn(e_pos, e_neg)
        loss.backward()
        assert e_pos.grad is not None
        assert e_neg.grad is not None
