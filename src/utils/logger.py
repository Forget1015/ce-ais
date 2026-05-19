"""结构化日志记录系统。

支持:
- 训练日志记录（损失、学习率、显存占用、训练速度）
- 评估日志记录（任务成功/失败、动作轨迹、能量值序列）
- 可选的 TensorBoard / Weights & Biases 集成
- 错误日志记录（完整堆栈信息、模型状态、GPU 内存信息）
"""

import json
import logging
import os
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional


def _get_gpu_memory_mb() -> float:
    """获取当前 GPU 显存占用（MB），无 GPU 时返回 0.0。"""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e6
    except Exception:
        pass
    return 0.0


class Logger:
    """CE-AIS 结构化日志记录器。

    提供训练、评估和错误日志的统一记录接口，
    并支持可选的 TensorBoard 和 Weights & Biases 集成。
    """

    def __init__(
        self,
        log_dir: str,
        tensorboard: bool = False,
        wandb: bool = False,
        log_interval: int = 10,
        project_name: str = "ce-ais",
    ):
        """初始化日志记录器。

        Args:
            log_dir: 日志输出目录。
            tensorboard: 是否启用 TensorBoard 集成。
            wandb: 是否启用 Weights & Biases 集成。
            log_interval: 日志记录间隔（步数）。
            project_name: 项目名称（用于 wandb）。
        """
        self.log_dir = log_dir
        self.log_interval = log_interval
        os.makedirs(log_dir, exist_ok=True)

        # Python 标准日志
        self._logger = logging.getLogger("ce-ais")
        self._logger.setLevel(logging.DEBUG)
        if not self._logger.handlers:
            fh = logging.FileHandler(
                os.path.join(log_dir, "train.log"), encoding="utf-8"
            )
            fh.setLevel(logging.DEBUG)
            fmt = logging.Formatter(
                "%(asctime)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            fh.setFormatter(fmt)
            self._logger.addHandler(fh)

        # TensorBoard 集成
        self._tb_writer = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tb_writer = SummaryWriter(
                    log_dir=os.path.join(log_dir, "tensorboard")
                )
                self._logger.info("TensorBoard 日志已启用")
            except ImportError:
                self._logger.warning(
                    "tensorboard 未安装，跳过 TensorBoard 集成"
                )

        # Weights & Biases 集成
        self._wandb_run = None
        if wandb:
            try:
                import wandb as _wandb

                self._wandb_run = _wandb.init(
                    project=project_name,
                    dir=log_dir,
                    reinit=True,
                )
                self._logger.info("Weights & Biases 日志已启用")
            except ImportError:
                self._logger.warning("wandb 未安装，跳过 W&B 集成")

        # 结构化日志文件路径
        self._train_log_path = os.path.join(log_dir, "train_metrics.jsonl")
        self._eval_log_path = os.path.join(log_dir, "eval_metrics.jsonl")
        self._error_log_path = os.path.join(log_dir, "errors.jsonl")

    # ------------------------------------------------------------------
    # 训练日志
    # ------------------------------------------------------------------

    def log_training(
        self,
        epoch: int,
        step: int,
        loss: float,
        learning_rate: float,
        gpu_memory_mb: Optional[float] = None,
        samples_per_sec: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """记录训练指标。

        Args:
            epoch: 当前训练轮次。
            step: 全局训练步数。
            loss: 当前损失值。
            learning_rate: 当前学习率。
            gpu_memory_mb: GPU 显存占用（MB），为 None 时自动检测。
            samples_per_sec: 训练速度（样本/秒）。
            extra: 额外指标字典。
        """
        if gpu_memory_mb is None:
            gpu_memory_mb = _get_gpu_memory_mb()

        record: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "type": "training",
            "epoch": epoch,
            "step": step,
            "loss": loss,
            "learning_rate": learning_rate,
            "gpu_memory_mb": gpu_memory_mb,
        }
        if samples_per_sec is not None:
            record["samples_per_sec"] = samples_per_sec
        if extra:
            record.update(extra)

        # 写入 JSONL
        self._append_jsonl(self._train_log_path, record)

        # Python 日志
        self._logger.info(
            "epoch=%d step=%d loss=%.6f lr=%.2e mem=%.1fMB",
            epoch,
            step,
            loss,
            learning_rate,
            gpu_memory_mb,
        )

        # TensorBoard
        if self._tb_writer is not None:
            self._tb_writer.add_scalar("train/loss", loss, step)
            self._tb_writer.add_scalar("train/learning_rate", learning_rate, step)
            self._tb_writer.add_scalar("train/gpu_memory_mb", gpu_memory_mb, step)
            if samples_per_sec is not None:
                self._tb_writer.add_scalar(
                    "train/samples_per_sec", samples_per_sec, step
                )

        # W&B
        if self._wandb_run is not None:
            try:
                import wandb as _wandb

                _wandb.log(
                    {
                        "train/loss": loss,
                        "train/learning_rate": learning_rate,
                        "train/gpu_memory_mb": gpu_memory_mb,
                        "train/samples_per_sec": samples_per_sec,
                        "epoch": epoch,
                    },
                    step=step,
                )
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 能量景观日志（CE-WM 训练专用）
    # ------------------------------------------------------------------

    def log_epoch_energy(
        self,
        epoch: int,
        pos_mean: float,
        neg_mean: float,
        margin: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """记录单 epoch 末的能量分布统计。

        Args:
            epoch: 当前 epoch（0-indexed）。
            pos_mean: 正样本（专家轨迹）平均能量。
            neg_mean: 负样本（扰动动作）平均能量。
            margin: neg_mean - pos_mean，越大越好。
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "type": "energy_stats",
            "epoch": epoch,
            "pos_energy_mean": pos_mean,
            "neg_energy_mean": neg_mean,
            "energy_margin": margin,
        }
        if extra:
            record.update(extra)
        self._append_jsonl(self._train_log_path, record)

        self._logger.info(
            "energy_epoch=%d pos=%.4f neg=%.4f margin=%.4f",
            epoch, pos_mean, neg_mean, margin,
        )

        if self._tb_writer is not None:
            self._tb_writer.add_scalar("energy/pos_mean", pos_mean, epoch)
            self._tb_writer.add_scalar("energy/neg_mean", neg_mean, epoch)
            self._tb_writer.add_scalar("energy/margin", margin, epoch)
            if extra:
                for key, value in extra.items():
                    if isinstance(value, (int, float)):
                        self._tb_writer.add_scalar(f"energy/{key}", value, epoch)

        if self._wandb_run is not None:
            try:
                import wandb as _wandb
                payload = {
                    "energy/pos_mean": pos_mean,
                    "energy/neg_mean": neg_mean,
                    "energy/margin": margin,
                    "epoch": epoch,
                }
                if extra:
                    payload.update({f"energy/{k}": v for k, v in extra.items()})
                _wandb.log(payload)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 评估日志
    # ------------------------------------------------------------------

    def log_evaluation(
        self,
        task_name: str,
        success: bool,
        action_trajectory: Optional[List[List[float]]] = None,
        energy_sequence: Optional[List[float]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """记录评估指标。

        Args:
            task_name: 任务名称。
            success: 任务是否成功。
            action_trajectory: 动作轨迹列表。
            energy_sequence: 能量值序列。
            extra: 额外指标字典。
        """
        record: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "type": "evaluation",
            "task_name": task_name,
            "success": success,
        }
        if action_trajectory is not None:
            record["action_trajectory"] = action_trajectory
        if energy_sequence is not None:
            record["energy_sequence"] = energy_sequence
        if extra:
            record.update(extra)

        self._append_jsonl(self._eval_log_path, record)

        status = "SUCCESS" if success else "FAIL"
        self._logger.info("eval task=%s status=%s", task_name, status)

        # TensorBoard（评估以标量形式记录成功率不太合适，跳过）
        # W&B
        if self._wandb_run is not None:
            try:
                import wandb as _wandb

                _wandb.log(
                    {
                        f"eval/{task_name}/success": int(success),
                    }
                )
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 错误日志
    # ------------------------------------------------------------------

    def log_error(
        self,
        error_type: str,
        context: Dict[str, Any],
        exception: Exception,
    ) -> None:
        """记录完整错误信息。

        Args:
            error_type: 错误类型标识（如 "oom", "nan_loss", "steering_failure"）。
            context: 当前模型状态（epoch, step, config 等）。
            exception: 异常对象。
        """
        error_record: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "message": str(exception),
            "traceback": traceback.format_exc(),
            "context": {
                "epoch": context.get("epoch"),
                "step": context.get("step"),
                "gpu_memory_mb": _get_gpu_memory_mb(),
                "config_snapshot": context.get("config"),
            },
        }

        self._append_jsonl(self._error_log_path, error_record)

        self._logger.error(
            "ERROR type=%s msg=%s", error_type, str(exception)
        )

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def _append_jsonl(self, path: str, record: Dict[str, Any]) -> None:
        """向 JSONL 文件追加一条记录。"""
        with open(path, "a", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

    def close(self) -> None:
        """关闭所有日志后端。"""
        if self._tb_writer is not None:
            self._tb_writer.close()
        if self._wandb_run is not None:
            try:
                import wandb as _wandb

                _wandb.finish()
            except Exception:
                pass
