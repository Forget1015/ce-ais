"""模型检查点保存与加载工具。

支持:
- 保存完整训练状态（模型参数、优化器状态、epoch、全局步数等）
- 从检查点恢复训练
- 自动检测最近检查点
- 检查点轮转机制（保留最近 N 个检查点）
"""

import glob
import os
import re
from typing import Any, Dict, Optional

import torch


class CheckpointManager:
    """检查点管理器。

    管理模型检查点的保存、加载和轮转，确保训练可中断恢复。
    """

    def __init__(
        self,
        checkpoint_dir: str,
        max_keep: int = 5,
        prefix: str = "checkpoint",
    ):
        """初始化检查点管理器。

        Args:
            checkpoint_dir: 检查点保存目录。
            max_keep: 最多保留的检查点数量（轮转机制）。
            prefix: 检查点文件名前缀。
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_keep = max_keep
        self.prefix = prefix
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        global_step: int = 0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """保存检查点。

        Args:
            epoch: 当前训练轮次。
            model: PyTorch 模型。
            optimizer: 优化器（可选）。
            scheduler: 学习率调度器（可选）。
            global_step: 全局训练步数。
            extra: 额外需要保存的状态字典。

        Returns:
            保存的检查点文件路径。
        """
        state: Dict[str, Any] = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
        }
        if optimizer is not None:
            state["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            state["scheduler_state_dict"] = scheduler.state_dict()
        if extra:
            state["extra"] = extra

        filename = f"{self.prefix}_epoch{epoch:04d}.pt"
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(state, filepath)

        # 检查点轮转：删除超出 max_keep 的旧检查点
        self._rotate()

        return filepath

    def load(
        self,
        filepath: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        map_location: Optional[str] = None,
    ) -> Dict[str, Any]:
        """从指定路径加载检查点。

        Args:
            filepath: 检查点文件路径。
            model: 要恢复参数的模型。
            optimizer: 要恢复状态的优化器（可选）。
            scheduler: 要恢复状态的学习率调度器（可选）。
            map_location: 设备映射（如 "cpu"）。

        Returns:
            包含 epoch、global_step 和 extra 的状态字典。
        """
        state = torch.load(filepath, map_location=map_location, weights_only=False)

        model.load_state_dict(state["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])

        if scheduler is not None and "scheduler_state_dict" in state:
            scheduler.load_state_dict(state["scheduler_state_dict"])

        return {
            "epoch": state.get("epoch", 0),
            "global_step": state.get("global_step", 0),
            "extra": state.get("extra", {}),
        }

    def find_latest(self) -> Optional[str]:
        """自动检测最近的检查点文件。

        Returns:
            最近检查点的文件路径，若无检查点则返回 None。
        """
        pattern = os.path.join(
            self.checkpoint_dir, f"{self.prefix}_epoch*.pt"
        )
        checkpoints = sorted(glob.glob(pattern))
        if not checkpoints:
            return None
        return checkpoints[-1]

    def list_checkpoints(self) -> list:
        """列出所有检查点文件路径（按 epoch 升序排列）。

        Returns:
            检查点文件路径列表。
        """
        pattern = os.path.join(
            self.checkpoint_dir, f"{self.prefix}_epoch*.pt"
        )
        return sorted(glob.glob(pattern))

    def _rotate(self) -> None:
        """检查点轮转：保留最近 max_keep 个检查点，删除多余的旧检查点。"""
        checkpoints = self.list_checkpoints()
        while len(checkpoints) > self.max_keep:
            oldest = checkpoints.pop(0)
            try:
                os.remove(oldest)
            except OSError:
                pass

    def resume_or_start(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        map_location: Optional[str] = None,
    ) -> Dict[str, Any]:
        """自动检测最近检查点并恢复训练，若无检查点则从头开始。

        Args:
            model: 要恢复参数的模型。
            optimizer: 要恢复状态的优化器（可选）。
            scheduler: 要恢复状态的学习率调度器（可选）。
            map_location: 设备映射。

        Returns:
            包含 epoch、global_step 和 extra 的状态字典。
            若无检查点，返回 {"epoch": 0, "global_step": 0, "extra": {}}。
        """
        latest = self.find_latest()
        if latest is None:
            return {"epoch": 0, "global_step": 0, "extra": {}}
        return self.load(
            latest,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=map_location,
        )
