"""评估指标模块。

实现多任务链式成功率、单任务成功率、瞬态恢复时间、
轨迹力学平滑度（Jerk）和计算延时帕累托边界数据记录。

支持将指标数据保存为 CSV/JSON 格式。
"""

import csv
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("ce-ais")


@dataclass
class EvalMetrics:
    """评估指标数据类。"""

    chain_success_rate: Dict[int, float] = field(default_factory=dict)
    single_task_rate: Dict[str, float] = field(default_factory=dict)
    avg_steps: float = 0.0
    transient_recovery_time: float = 0.0
    trajectory_jerk: float = 0.0
    latency_ms: float = 0.0


class MetricsModule:
    """评估指标计算与记录模块。

    Args:
        output_dir: 指标输出目录。
    """

    def __init__(self, output_dir: str = "logs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 累积数据
        self._chain_results: Dict[int, List[bool]] = {}
        self._task_results: Dict[str, List[bool]] = {}
        self._step_counts: List[int] = []
        self._latency_records: List[Dict[str, float]] = []
        self._trajectory_data: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # 数据记录
    # ------------------------------------------------------------------

    def record_chain_result(
        self, chain_length: int, success: bool
    ) -> None:
        """记录一次链式评估结果。"""
        if chain_length not in self._chain_results:
            self._chain_results[chain_length] = []
        self._chain_results[chain_length].append(success)

    def record_task_result(
        self, task_name: str, success: bool, steps: int = 0
    ) -> None:
        """记录一次单任务评估结果。"""
        if task_name not in self._task_results:
            self._task_results[task_name] = []
        self._task_results[task_name].append(success)
        if steps > 0:
            self._step_counts.append(steps)

    def record_latency(
        self, latency_ms: float, success_rate: float
    ) -> None:
        """记录延时-成功率帕累托数据点。"""
        self._latency_records.append(
            {"latency_ms": latency_ms, "success_rate": success_rate}
        )

    def record_trajectory(self, positions: np.ndarray) -> None:
        """记录末端执行器轨迹用于 Jerk 计算。

        Args:
            positions: [T, 3] 末端执行器 xyz 位置序列。
        """
        self._trajectory_data.append(positions)

    # ------------------------------------------------------------------
    # 指标计算
    # ------------------------------------------------------------------

    def compute_chain_success_rate(self) -> Dict[int, float]:
        """计算多任务链式成功率。"""
        rates = {}
        for length, results in sorted(self._chain_results.items()):
            if results:
                rates[length] = sum(results) / len(results)
            else:
                rates[length] = 0.0
        return rates

    def compute_single_task_rate(self) -> Dict[str, float]:
        """计算单任务成功率。"""
        rates = {}
        for task, results in self._task_results.items():
            if results:
                rates[task] = sum(results) / len(results)
            else:
                rates[task] = 0.0
        return rates

    @staticmethod
    def compute_trajectory_jerk(positions: np.ndarray, dt: float = 1.0) -> float:
        """计算轨迹 Jerk（加速度变化率 = 三阶数值差分）。

        Args:
            positions: [T, D] 位置序列（D 通常为 3）。
            dt: 时间步长。

        Returns:
            平均 Jerk 值（L2 范数的均值）。
        """
        if len(positions) < 4:
            return 0.0

        positions = np.asarray(positions, dtype=np.float64)

        # 一阶差分: 速度
        velocity = np.diff(positions, axis=0) / dt
        # 二阶差分: 加速度
        acceleration = np.diff(velocity, axis=0) / dt
        # 三阶差分: Jerk
        jerk = np.diff(acceleration, axis=0) / dt

        # 每个时间步的 Jerk L2 范数
        jerk_norms = np.linalg.norm(jerk, axis=-1)
        return float(np.mean(jerk_norms))

    @staticmethod
    def compute_transient_recovery_time(
        success_history: List[bool],
        perturbation_step: int,
        window_size: int = 10,
        threshold: float = 0.5,
    ) -> int:
        """计算瞬态恢复时间。

        从干扰注入时刻开始，计算胜率恢复到阈值所需的步数。

        Args:
            success_history: 按时间排列的成功/失败记录。
            perturbation_step: 干扰注入的时间步索引。
            window_size: 滑动窗口大小。
            threshold: 恢复阈值。

        Returns:
            恢复所需步数（非负整数）。若未恢复则返回剩余长度。
        """
        if perturbation_step >= len(success_history):
            return 0

        post_perturbation = success_history[perturbation_step:]

        for i in range(len(post_perturbation) - window_size + 1):
            window = post_perturbation[i : i + window_size]
            win_rate = sum(window) / len(window)
            if win_rate >= threshold:
                return i

        return len(post_perturbation)

    # ------------------------------------------------------------------
    # 汇总
    # ------------------------------------------------------------------

    def compute_all(self) -> EvalMetrics:
        """计算所有指标并返回 EvalMetrics。"""
        chain_rates = self.compute_chain_success_rate()
        task_rates = self.compute_single_task_rate()
        avg_steps = (
            float(np.mean(self._step_counts)) if self._step_counts else 0.0
        )

        # 汇总轨迹 Jerk
        jerk_values = []
        for traj in self._trajectory_data:
            j = self.compute_trajectory_jerk(traj)
            jerk_values.append(j)
        avg_jerk = float(np.mean(jerk_values)) if jerk_values else 0.0

        # 平均延时
        avg_latency = 0.0
        if self._latency_records:
            avg_latency = float(
                np.mean([r["latency_ms"] for r in self._latency_records])
            )

        return EvalMetrics(
            chain_success_rate=chain_rates,
            single_task_rate=task_rates,
            avg_steps=avg_steps,
            transient_recovery_time=0.0,  # 需要外部提供 perturbation_step
            trajectory_jerk=avg_jerk,
            latency_ms=avg_latency,
        )

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def save_json(self, filename: str = "eval_metrics.json") -> str:
        """将指标保存为 JSON 文件。"""
        metrics = self.compute_all()
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(metrics), f, indent=2, ensure_ascii=False)
        logger.info("Metrics saved to %s", path)
        return path

    def save_csv(self, filename: str = "eval_metrics.csv") -> str:
        """将指标保存为 CSV 文件。"""
        metrics = self.compute_all()
        path = os.path.join(self.output_dir, filename)

        rows = []
        # 链式成功率
        for length, rate in metrics.chain_success_rate.items():
            rows.append(
                {"metric": f"chain_success_{length}", "value": rate}
            )
        # 单任务成功率
        for task, rate in metrics.single_task_rate.items():
            rows.append(
                {"metric": f"task_rate_{task}", "value": rate}
            )
        # 标量指标
        rows.append({"metric": "avg_steps", "value": metrics.avg_steps})
        rows.append(
            {"metric": "trajectory_jerk", "value": metrics.trajectory_jerk}
        )
        rows.append({"metric": "latency_ms", "value": metrics.latency_ms})

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["metric", "value"])
            writer.writeheader()
            writer.writerows(rows)

        logger.info("Metrics CSV saved to %s", path)
        return path

    def save_latency_pareto(
        self, filename: str = "latency_pareto.csv"
    ) -> str:
        """保存延时帕累托边界数据。"""
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["latency_ms", "success_rate"]
            )
            writer.writeheader()
            writer.writerows(self._latency_records)
        logger.info("Latency Pareto data saved to %s", path)
        return path
