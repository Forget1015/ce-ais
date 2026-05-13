"""可视化工具模块。

自动生成评估指标的可视化图表（折线图、散点图）至 logs/ 目录。
"""

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("ce-ais")


def _get_matplotlib():
    """延迟导入 matplotlib，使用 Agg 后端避免 GUI 依赖。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_chain_success_rate(
    chain_rates: Dict[int, float],
    output_dir: str = "logs",
    filename: str = "chain_success_rate.png",
) -> str:
    """绘制多任务链式成功率折线图。

    Args:
        chain_rates: {chain_length: success_rate} 字典。
        output_dir: 输出目录。
        filename: 输出文件名。

    Returns:
        图表文件路径。
    """
    plt = _get_matplotlib()
    os.makedirs(output_dir, exist_ok=True)

    lengths = sorted(chain_rates.keys())
    rates = [chain_rates[l] for l in lengths]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lengths, rates, marker="o", linewidth=2, markersize=8)
    ax.set_xlabel("Chain Length", fontsize=12)
    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_title("Multi-Task Chain Success Rate", fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chain success rate plot saved to %s", path)
    return path


def plot_latency_pareto(
    latency_data: List[Dict[str, float]],
    output_dir: str = "logs",
    filename: str = "latency_pareto.png",
) -> str:
    """绘制延时-成功率帕累托散点图。

    Args:
        latency_data: [{"latency_ms": float, "success_rate": float}, ...] 列表。
        output_dir: 输出目录。
        filename: 输出文件名。

    Returns:
        图表文件路径。
    """
    plt = _get_matplotlib()
    os.makedirs(output_dir, exist_ok=True)

    latencies = [d["latency_ms"] for d in latency_data]
    rates = [d["success_rate"] for d in latency_data]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(latencies, rates, alpha=0.7, s=50)
    ax.set_xlabel("Latency (ms)", fontsize=12)
    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_title("Latency vs Success Rate (Pareto)", fontsize=14)
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Latency Pareto plot saved to %s", path)
    return path


def plot_recovery_curve(
    success_history: List[bool],
    perturbation_step: int,
    output_dir: str = "logs",
    filename: str = "recovery_curve.png",
    window_size: int = 10,
) -> str:
    """绘制瞬态恢复曲线。

    Args:
        success_history: 按时间排列的成功/失败记录。
        perturbation_step: 干扰注入的时间步索引。
        output_dir: 输出目录。
        filename: 输出文件名。
        window_size: 滑动窗口大小。

    Returns:
        图表文件路径。
    """
    plt = _get_matplotlib()
    os.makedirs(output_dir, exist_ok=True)

    # 计算滑动窗口胜率
    win_rates = []
    for i in range(len(success_history) - window_size + 1):
        window = success_history[i : i + window_size]
        win_rates.append(sum(window) / len(window))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(len(win_rates)), win_rates, linewidth=1.5)
    ax.axvline(
        x=perturbation_step,
        color="red",
        linestyle="--",
        label="OOD Injection",
    )
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Win Rate (sliding window)", fontsize=12)
    ax.set_title("Transient Recovery Curve", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Recovery curve saved to %s", path)
    return path


def plot_comparison_table(
    results: Dict[str, Dict[str, float]],
    output_dir: str = "logs",
    filename: str = "comparison_table.png",
) -> str:
    """绘制方法对比表格图。

    Args:
        results: {method_name: {metric_name: value}} 嵌套字典。
        output_dir: 输出目录。
        filename: 输出文件名。

    Returns:
        图表文件路径。
    """
    plt = _get_matplotlib()
    os.makedirs(output_dir, exist_ok=True)

    methods = list(results.keys())
    if not methods:
        return ""

    metrics = list(results[methods[0]].keys())
    cell_text = []
    for method in methods:
        row = [f"{results[method].get(m, 0.0):.4f}" for m in metrics]
        cell_text.append(row)

    fig, ax = plt.subplots(figsize=(max(8, len(metrics) * 1.5), len(methods) * 0.6 + 2))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        rowLabels=methods,
        colLabels=metrics,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax.set_title("Method Comparison", fontsize=14, pad=20)

    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Comparison table saved to %s", path)
    return path
