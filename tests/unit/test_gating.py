"""双向不确定性门控模块单元测试。

测试:
- 高斯门控函数输出范围
- 均值处最大值
- 远离均值时衰减
- 历史窗口更新
- 重置功能
"""

import torch
import pytest

from src.config.schema import BilateralGatingConfig
from src.steering.bilateral_gating import BilateralGating


@pytest.fixture
def gating_config():
    return BilateralGatingConfig(
        lambda_max=1.0,
        sensitivity=0.1,
        window_size=50,
    )


@pytest.fixture
def gating(gating_config):
    return BilateralGating(gating_config)


class TestBilateralGating:
    """双向不确定性门控测试。"""

    def test_output_range(self, gating):
        """输出应在 [0, λ_max] 范围内。"""
        uncertainty = torch.tensor([0.1, 0.5, 1.0, 2.0, 5.0])
        lam = gating.compute_lambda(uncertainty)

        assert (lam >= 0).all(), "Lambda should be non-negative"
        assert (lam <= gating.lambda_max + 1e-6).all(), (
            "Lambda should not exceed lambda_max"
        )

    def test_max_at_mean(self, gating):
        """当不确定性等于历史均值时，λ 应等于 λ_max。"""
        # 先填充历史，使均值稳定
        mu = 1.0
        for _ in range(50):
            gating.compute_lambda(torch.tensor([mu]))

        # 输入等于均值
        lam = gating.compute_lambda(torch.tensor([mu]))
        assert torch.allclose(
            lam, torch.tensor([gating.lambda_max]), atol=0.05
        ), f"Lambda at mean should be ~lambda_max, got {lam.item()}"

    def test_decay_away_from_mean(self, gating):
        """远离均值时 λ 应衰减。"""
        mu = 1.0
        for _ in range(50):
            gating.compute_lambda(torch.tensor([mu]))

        # 在均值处
        lam_at_mean = gating.compute_lambda(torch.tensor([mu]))

        # 重新填充历史（因为上面的调用改变了均值）
        gating.reset()
        for _ in range(50):
            gating.compute_lambda(torch.tensor([mu]))

        # 远离均值
        lam_far = gating.compute_lambda(torch.tensor([mu + 1.0]))

        assert lam_at_mean.item() > lam_far.item(), (
            "Lambda should decrease when uncertainty is far from mean"
        )

    def test_high_uncertainty_low_lambda(self, gating):
        """极高不确定性时 λ 应趋近 0。"""
        mu = 1.0
        for _ in range(50):
            gating.compute_lambda(torch.tensor([mu]))

        # 极高不确定性
        lam = gating.compute_lambda(torch.tensor([mu + 10.0]))
        assert lam.item() < 0.01, (
            f"Lambda should be near 0 for extreme uncertainty, got {lam.item()}"
        )

    def test_batch_processing(self, gating):
        """应支持批量处理。"""
        uncertainty = torch.tensor([0.5, 1.0, 1.5, 2.0])
        lam = gating.compute_lambda(uncertainty)

        assert lam.shape == (4,)
        assert (lam >= 0).all()

    def test_history_window_size(self, gating):
        """历史窗口应不超过配置的最大长度。"""
        for i in range(100):
            gating.compute_lambda(torch.tensor([float(i)]))

        assert len(gating.history) <= gating.window_size

    def test_reset_clears_history(self, gating):
        """reset() 应清空历史缓冲。"""
        for _ in range(10):
            gating.compute_lambda(torch.tensor([1.0]))

        assert len(gating.history) > 0
        gating.reset()
        assert len(gating.history) == 0

    def test_get_mean_uncertainty_empty(self, gating):
        """空历史时均值应为 0。"""
        assert gating.get_mean_uncertainty() == 0.0

    def test_get_mean_uncertainty(self, gating):
        """均值计算应正确。"""
        values = [1.0, 2.0, 3.0]
        for v in values:
            gating.compute_lambda(torch.tensor([v]))

        expected_mean = sum(values) / len(values)
        assert abs(gating.get_mean_uncertainty() - expected_mean) < 1e-5

    def test_different_lambda_max(self):
        """不同 λ_max 配置应正确反映。"""
        config = BilateralGatingConfig(
            lambda_max=2.5,
            sensitivity=0.1,
            window_size=10,
        )
        gating = BilateralGating(config)

        mu = 1.0
        for _ in range(10):
            gating.compute_lambda(torch.tensor([mu]))

        lam = gating.compute_lambda(torch.tensor([mu]))
        assert torch.allclose(
            lam, torch.tensor([2.5]), atol=0.1
        )

    def test_different_sensitivity(self):
        """更大的灵敏度参数应使门控更宽容。"""
        config_narrow = BilateralGatingConfig(
            lambda_max=1.0, sensitivity=0.01, window_size=10
        )
        config_wide = BilateralGatingConfig(
            lambda_max=1.0, sensitivity=1.0, window_size=10
        )

        gating_narrow = BilateralGating(config_narrow)
        gating_wide = BilateralGating(config_wide)

        mu = 1.0
        for _ in range(10):
            gating_narrow.compute_lambda(torch.tensor([mu]))
            gating_wide.compute_lambda(torch.tensor([mu]))

        # 偏离均值 0.5
        offset = mu + 0.5
        lam_narrow = gating_narrow.compute_lambda(torch.tensor([offset]))
        lam_wide = gating_wide.compute_lambda(torch.tensor([offset]))

        # 窄灵敏度应衰减更快
        assert lam_narrow.item() < lam_wide.item(), (
            "Narrower sensitivity should decay faster"
        )

    def test_gaussian_formula_correctness(self):
        """验证高斯门控公式的数学正确性。"""
        config = BilateralGatingConfig(
            lambda_max=1.0, sensitivity=0.1, window_size=10
        )
        gating = BilateralGating(config)

        mu = 2.0
        for _ in range(10):
            gating.compute_lambda(torch.tensor([mu]))

        u_t = torch.tensor([2.5])
        lam = gating.compute_lambda(u_t)

        # 手动计算期望值
        # 注意: 调用 compute_lambda 会更新历史，所以 mu 会略有变化
        # 使用近似验证
        import math
        # 历史中有 10 个 mu=2.0 和 1 个 2.5 的均值
        actual_mu = (10 * 2.0 + 2.5) / 11
        expected = 1.0 * math.exp(
            -(2.5 - actual_mu) ** 2 / (2 * 0.1 ** 2)
        )

        assert abs(lam.item() - expected) < 0.01, (
            f"Expected {expected}, got {lam.item()}"
        )
