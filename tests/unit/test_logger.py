"""Logger 单元测试。"""

import json
import os

import pytest

from src.utils.logger import Logger


@pytest.fixture
def log_dir(tmp_path):
    """创建临时日志目录。"""
    return str(tmp_path / "logs")


class TestLoggerTraining:
    """训练日志记录测试。"""

    def test_log_training_creates_jsonl(self, log_dir):
        logger = Logger(log_dir=log_dir)
        logger.log_training(epoch=1, step=10, loss=0.5, learning_rate=1e-4)
        logger.close()

        path = os.path.join(log_dir, "train_metrics.jsonl")
        assert os.path.exists(path)
        with open(path, "r") as f:
            record = json.loads(f.readline())
        assert record["type"] == "training"
        assert record["epoch"] == 1
        assert record["step"] == 10
        assert record["loss"] == 0.5
        assert record["learning_rate"] == 1e-4

    def test_log_training_with_extra(self, log_dir):
        logger = Logger(log_dir=log_dir)
        logger.log_training(
            epoch=2,
            step=20,
            loss=0.3,
            learning_rate=5e-5,
            samples_per_sec=128.0,
            extra={"grad_norm": 1.5},
        )
        logger.close()

        path = os.path.join(log_dir, "train_metrics.jsonl")
        with open(path, "r") as f:
            record = json.loads(f.readline())
        assert record["samples_per_sec"] == 128.0
        assert record["grad_norm"] == 1.5

    def test_log_training_multiple_entries(self, log_dir):
        logger = Logger(log_dir=log_dir)
        for i in range(5):
            logger.log_training(epoch=1, step=i, loss=1.0 - i * 0.1, learning_rate=1e-4)
        logger.close()

        path = os.path.join(log_dir, "train_metrics.jsonl")
        with open(path, "r") as f:
            lines = f.readlines()
        assert len(lines) == 5


class TestLoggerEvaluation:
    """评估日志记录测试。"""

    def test_log_evaluation_success(self, log_dir):
        logger = Logger(log_dir=log_dir)
        logger.log_evaluation(task_name="pick_up", success=True)
        logger.close()

        path = os.path.join(log_dir, "eval_metrics.jsonl")
        with open(path, "r") as f:
            record = json.loads(f.readline())
        assert record["task_name"] == "pick_up"
        assert record["success"] is True

    def test_log_evaluation_with_trajectory(self, log_dir):
        logger = Logger(log_dir=log_dir)
        trajectory = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        energies = [1.0, 0.8, 0.5]
        logger.log_evaluation(
            task_name="slide",
            success=False,
            action_trajectory=trajectory,
            energy_sequence=energies,
        )
        logger.close()

        path = os.path.join(log_dir, "eval_metrics.jsonl")
        with open(path, "r") as f:
            record = json.loads(f.readline())
        assert record["action_trajectory"] == trajectory
        assert record["energy_sequence"] == energies
        assert record["success"] is False


class TestLoggerError:
    """错误日志记录测试。"""

    def test_log_error_records_traceback(self, log_dir):
        logger = Logger(log_dir=log_dir)
        try:
            raise ValueError("test error")
        except ValueError as e:
            logger.log_error(
                error_type="test_error",
                context={"epoch": 5, "step": 100},
                exception=e,
            )
        logger.close()

        path = os.path.join(log_dir, "errors.jsonl")
        assert os.path.exists(path)
        with open(path, "r") as f:
            record = json.loads(f.readline())
        assert record["error_type"] == "test_error"
        assert record["message"] == "test error"
        assert "ValueError" in record["traceback"]
        assert record["context"]["epoch"] == 5
        assert record["context"]["step"] == 100

    def test_log_error_with_config_snapshot(self, log_dir):
        logger = Logger(log_dir=log_dir)
        try:
            raise RuntimeError("oom")
        except RuntimeError as e:
            logger.log_error(
                error_type="oom",
                context={"epoch": 1, "step": 50, "config": {"batch_size": 64}},
                exception=e,
            )
        logger.close()

        path = os.path.join(log_dir, "errors.jsonl")
        with open(path, "r") as f:
            record = json.loads(f.readline())
        assert record["context"]["config_snapshot"] == {"batch_size": 64}


class TestLoggerInit:
    """Logger 初始化测试。"""

    def test_creates_log_dir(self, tmp_path):
        log_dir = str(tmp_path / "new_logs")
        logger = Logger(log_dir=log_dir)
        logger.close()
        assert os.path.isdir(log_dir)

    def test_tensorboard_disabled_by_default(self, log_dir):
        logger = Logger(log_dir=log_dir)
        assert logger._tb_writer is None
        logger.close()

    def test_wandb_disabled_by_default(self, log_dir):
        logger = Logger(log_dir=log_dir)
        assert logger._wandb_run is None
        logger.close()
