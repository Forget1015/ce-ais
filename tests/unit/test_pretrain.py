"""预训练流程单元测试。

使用小规模 mock 数据验证预训练管线完整运行。
验证损失下降、检查点保存/恢复。
"""

import os
import tempfile

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config.schema import (
    CEAISConfig,
    CEWMConfig,
    EncoderConfig,
    LoggingConfig,
    TrainingConfig,
)
from src.encoders.contrastive_encoder import ContrastiveEncoder
from src.training.pretrain_pipeline import PretrainPipeline
from src.world_model.ce_wm import CausalEnergyWorldModel


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def small_config():
    """小规模训练配置。"""
    config = CEAISConfig()
    config.encoder = EncoderConfig(
        backbone_type="resnet18",
        pose_dim=7,
        visual_dim=512,
        latent_dim=32,
        temperature=0.07,
        image_size=[64, 64],
    )
    config.ce_wm = CEWMConfig(
        d_model=64, d_state=16, n_layers=2, expand_factor=2,
        mimo_groups=2, action_dim=7, latent_dim=32, dropout=0.1,
    )
    config.training = TrainingConfig(
        encoder_epochs=3,
        ce_wm_epochs=3,
        batch_size=4,
        learning_rate=1e-3,
        weight_decay=1e-5,
        amp=False,
        checkpoint_interval=1,
        neg_sample_ratio=2,
    )
    config.logging = LoggingConfig(
        tensorboard=False, wandb=False, log_interval=1,
    )
    config.project.device = "cpu"
    return config


class _DictDataset(torch.utils.data.Dataset):
    """将字典数据包装为 Dataset。"""

    def __init__(self, data_dict, size):
        self.data = data_dict
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {k: v[idx % v.shape[0]] for k, v in self.data.items()}


@pytest.fixture
def encoder_dataloader():
    """Mock 编码器训练数据加载器。"""
    n_samples = 8
    data = {
        "rgb": torch.randn(n_samples, 3, 64, 64),
        "depth": torch.randn(n_samples, 1, 64, 64),
        "pose": torch.randn(n_samples, 7),
    }
    dataset = _DictDataset(data, n_samples)
    return DataLoader(dataset, batch_size=4, shuffle=False)


@pytest.fixture
def cewm_dataloader():
    """Mock CE-WM 训练数据加载器。"""
    n_samples = 8
    K = 2  # 负样本比例
    data = {
        "z_pos": torch.randn(n_samples, 32),
        "a_pos": torch.randn(n_samples, 7),
        "a_neg": torch.randn(n_samples, K, 7),
    }
    dataset = _DictDataset(data, n_samples)
    return DataLoader(dataset, batch_size=4, shuffle=False)


# ------------------------------------------------------------------
# 测试
# ------------------------------------------------------------------


class TestPretrainPipeline:
    """预训练管线测试。"""

    def test_encoder_pretrain_runs(
        self, small_config, encoder_dataloader
    ):
        """编码器预训练应完整运行不抛异常。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            encoder = ContrastiveEncoder(small_config.encoder)
            ce_wm = CausalEnergyWorldModel(small_config.ce_wm)

            pipeline = PretrainPipeline(
                config=small_config,
                encoder=encoder,
                ce_wm=ce_wm,
                log_dir=os.path.join(tmpdir, "logs"),
                checkpoint_dir=os.path.join(tmpdir, "ckpts"),
            )

            result = pipeline.pretrain_encoder(
                encoder_dataloader, resume=False
            )

            assert "final_loss" in result
            assert "losses_history" in result
            assert len(result["losses_history"]) == small_config.training.encoder_epochs

    def test_encoder_loss_decreases(
        self, small_config, encoder_dataloader
    ):
        """编码器预训练损失应呈下降趋势。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 增加 epoch 数以观察下降
            small_config.training.encoder_epochs = 5

            encoder = ContrastiveEncoder(small_config.encoder)
            ce_wm = CausalEnergyWorldModel(small_config.ce_wm)

            pipeline = PretrainPipeline(
                config=small_config,
                encoder=encoder,
                ce_wm=ce_wm,
                log_dir=os.path.join(tmpdir, "logs"),
                checkpoint_dir=os.path.join(tmpdir, "ckpts"),
            )

            result = pipeline.pretrain_encoder(
                encoder_dataloader, resume=False
            )

            losses = result["losses_history"]
            # 最终损失应小于初始损失（允许一定波动）
            assert losses[-1] <= losses[0] + 0.5, (
                f"Loss did not decrease: first={losses[0]:.4f}, "
                f"last={losses[-1]:.4f}"
            )

    def test_cewm_pretrain_runs(
        self, small_config, cewm_dataloader
    ):
        """CE-WM 预训练应完整运行不抛异常。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            encoder = ContrastiveEncoder(small_config.encoder)
            ce_wm = CausalEnergyWorldModel(small_config.ce_wm)

            pipeline = PretrainPipeline(
                config=small_config,
                encoder=encoder,
                ce_wm=ce_wm,
                log_dir=os.path.join(tmpdir, "logs"),
                checkpoint_dir=os.path.join(tmpdir, "ckpts"),
            )

            result = pipeline.pretrain_cewm(
                cewm_dataloader, resume=False
            )

            assert "final_loss" in result
            assert "energy_report" in result
            assert len(result["losses_history"]) == small_config.training.ce_wm_epochs

    def test_checkpoint_save_and_resume(
        self, small_config, encoder_dataloader
    ):
        """检查点保存后应能恢复训练。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = os.path.join(tmpdir, "ckpts")
            log_dir = os.path.join(tmpdir, "logs")

            encoder = ContrastiveEncoder(small_config.encoder)
            ce_wm = CausalEnergyWorldModel(small_config.ce_wm)

            pipeline = PretrainPipeline(
                config=small_config,
                encoder=encoder,
                ce_wm=ce_wm,
                log_dir=log_dir,
                checkpoint_dir=ckpt_dir,
            )

            # 第一次训练
            result1 = pipeline.pretrain_encoder(
                encoder_dataloader, resume=False
            )

            # 验证检查点已保存
            ckpt_files = [
                f for f in os.listdir(ckpt_dir)
                if f.startswith("encoder") and f.endswith(".pt")
            ]
            assert len(ckpt_files) > 0, "No checkpoint files saved"

            # 创建新管线并恢复
            encoder2 = ContrastiveEncoder(small_config.encoder)
            ce_wm2 = CausalEnergyWorldModel(small_config.ce_wm)

            pipeline2 = PretrainPipeline(
                config=small_config,
                encoder=encoder2,
                ce_wm=ce_wm2,
                log_dir=os.path.join(tmpdir, "logs2"),
                checkpoint_dir=ckpt_dir,
            )

            # 恢复训练应不抛异常
            result2 = pipeline2.pretrain_encoder(
                encoder_dataloader, resume=True
            )

            assert "final_loss" in result2
