"""CheckpointManager 单元测试。"""

import os

import pytest
import torch
import torch.nn as nn

from src.utils.checkpoint import CheckpointManager


class SimpleModel(nn.Module):
    """用于测试的简单模型。"""

    def __init__(self, in_dim=4, out_dim=2):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def ckpt_dir(tmp_path):
    return str(tmp_path / "checkpoints")


@pytest.fixture
def model():
    torch.manual_seed(42)
    return SimpleModel()


class TestCheckpointSaveLoad:
    """检查点保存与加载测试。"""

    def test_save_creates_file(self, ckpt_dir, model):
        mgr = CheckpointManager(ckpt_dir)
        path = mgr.save(epoch=1, model=model)
        assert os.path.exists(path)
        assert "epoch0001" in path

    def test_load_restores_model(self, ckpt_dir):
        torch.manual_seed(0)
        model_a = SimpleModel()
        mgr = CheckpointManager(ckpt_dir)
        path = mgr.save(epoch=5, model=model_a, global_step=100)

        torch.manual_seed(99)
        model_b = SimpleModel()
        # 确保参数不同
        assert not torch.equal(
            model_a.linear.weight, model_b.linear.weight
        )

        info = mgr.load(path, model_b, map_location="cpu")
        assert info["epoch"] == 5
        assert info["global_step"] == 100
        assert torch.equal(model_a.linear.weight, model_b.linear.weight)

    def test_save_load_with_optimizer(self, ckpt_dir, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        # 执行一步以改变优化器状态
        x = torch.randn(2, 4)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        mgr = CheckpointManager(ckpt_dir)
        path = mgr.save(epoch=3, model=model, optimizer=optimizer, global_step=50)

        model_new = SimpleModel()
        opt_new = torch.optim.Adam(model_new.parameters(), lr=1e-3)
        info = mgr.load(path, model_new, optimizer=opt_new, map_location="cpu")

        assert info["epoch"] == 3
        assert info["global_step"] == 50
        # 优化器状态应恢复
        for key in optimizer.state_dict()["param_groups"][0]:
            assert (
                optimizer.state_dict()["param_groups"][0][key]
                == opt_new.state_dict()["param_groups"][0][key]
            )

    def test_save_load_with_extra(self, ckpt_dir, model):
        mgr = CheckpointManager(ckpt_dir)
        extra = {"best_loss": 0.01, "config": {"lr": 1e-4}}
        path = mgr.save(epoch=10, model=model, extra=extra)

        model_new = SimpleModel()
        info = mgr.load(path, model_new, map_location="cpu")
        assert info["extra"]["best_loss"] == 0.01
        assert info["extra"]["config"]["lr"] == 1e-4


class TestCheckpointFindLatest:
    """自动检测最近检查点测试。"""

    def test_find_latest_returns_none_when_empty(self, ckpt_dir):
        mgr = CheckpointManager(ckpt_dir)
        assert mgr.find_latest() is None

    def test_find_latest_returns_most_recent(self, ckpt_dir, model):
        mgr = CheckpointManager(ckpt_dir, max_keep=10)
        mgr.save(epoch=1, model=model)
        mgr.save(epoch=5, model=model)
        mgr.save(epoch=3, model=model)

        latest = mgr.find_latest()
        assert latest is not None
        assert "epoch0005" in latest


class TestCheckpointRotation:
    """检查点轮转机制测试。"""

    def test_rotation_keeps_max_keep(self, ckpt_dir, model):
        mgr = CheckpointManager(ckpt_dir, max_keep=3)
        for i in range(1, 7):
            mgr.save(epoch=i, model=model)

        remaining = mgr.list_checkpoints()
        assert len(remaining) == 3
        # 应保留最近 3 个
        assert "epoch0004" in remaining[0]
        assert "epoch0005" in remaining[1]
        assert "epoch0006" in remaining[2]

    def test_rotation_max_keep_1(self, ckpt_dir, model):
        mgr = CheckpointManager(ckpt_dir, max_keep=1)
        mgr.save(epoch=1, model=model)
        mgr.save(epoch=2, model=model)
        mgr.save(epoch=3, model=model)

        remaining = mgr.list_checkpoints()
        assert len(remaining) == 1
        assert "epoch0003" in remaining[0]


class TestCheckpointResumeOrStart:
    """自动恢复训练测试。"""

    def test_resume_from_scratch(self, ckpt_dir, model):
        mgr = CheckpointManager(ckpt_dir)
        info = mgr.resume_or_start(model, map_location="cpu")
        assert info["epoch"] == 0
        assert info["global_step"] == 0

    def test_resume_from_checkpoint(self, ckpt_dir):
        torch.manual_seed(0)
        model_a = SimpleModel()
        mgr = CheckpointManager(ckpt_dir)
        mgr.save(epoch=10, model=model_a, global_step=500)

        torch.manual_seed(99)
        model_b = SimpleModel()
        info = mgr.resume_or_start(model_b, map_location="cpu")
        assert info["epoch"] == 10
        assert info["global_step"] == 500
        assert torch.equal(model_a.linear.weight, model_b.linear.weight)
