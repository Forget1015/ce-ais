#!/usr/bin/env python3
"""CE-AIS 论文出图脚本。

从 results/ 目录读取实验数据，生成可直接 \\includegraphics 的矢量 PDF 图
和 LaTeX 表格。

输出:
- figures/main_table.tex:       主实验对比表（表 3）
- figures/ood_table.tex:        OOD 实验对比表（表 4）
- figures/recovery_curve.pdf:   U 型反弹恢复曲线（图 5）
- figures/pareto.pdf:           Pareto 帕累托散点图（图 6）
- figures/ablation_table.tex:   消融实验表

Usage:
    uv run python scripts/plot_results.py
    uv run python scripts/plot_results.py --results-dir results/ --output-dir figures/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description="CE-AIS Paper Plots & Tables")
    parser.add_argument("--results-dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "results"))
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "figures"))
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


METHOD_DISPLAY = {
    "ce_ais": "CE-AIS (Ours)",
    "frozen_openvla": "Frozen OpenVLA",
    "pdf": "PDF",
    "tt_vla": "TT-VLA",
    "ada_world_policy": "AdaWorldPolicy",
}

METHOD_COLORS = {
    "ce_ais": "#E63946",
    "frozen_openvla": "#457B9D",
    "pdf": "#2A9D8F",
    "tt_vla": "#E9C46A",
    "ada_world_policy": "#F4A261",
}

METHOD_MARKERS = {
    "ce_ais": "D",
    "frozen_openvla": "o",
    "pdf": "s",
    "tt_vla": "^",
    "ada_world_policy": "v",
}


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })
    return plt


def generate_main_table(results_dir, output_dir):
    """生成主实验对比表 (LaTeX)。"""
    path = os.path.join(results_dir, "main_experiment.json")
    if not os.path.exists(path):
        print(f"  [SKIP] {path} not found")
        return

    with open(path) as f:
        data = json.load(f)

    results = data["results"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Main experiment: CE-AIS vs.\ baselines on CALVIN ABC$\to$D.}",
        r"\label{tab:main}",
        r"\begin{tabular}{lcccccr}",
        r"\toprule",
        r"Method & L1 & L2 & L3 & L4 & L5 & Lat.(ms) \\",
        r"\midrule",
    ]

    for method, res in results.items():
        display = METHOD_DISPLAY.get(method, method)
        if method == "ce_ais":
            display = r"\textbf{" + display + "}"
        cols = []
        for i in range(1, 6):
            val = res.get(f"chain_{i}", 0)
            s = f"{val:.1%}"
            # Bold best value
            cols.append(s)
        lat = res.get("avg_latency_ms", 0)
        line = f"{display} & {' & '.join(cols)} & {lat:.1f} \\\\"
        lines.append(line)

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    out_path = os.path.join(output_dir, "main_table.tex")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  [OK] {out_path}")


def generate_ood_table(results_dir, output_dir):
    """生成 OOD 实验对比表 (LaTeX)。"""
    path = os.path.join(results_dir, "ood_experiment.json")
    if not os.path.exists(path):
        print(f"  [SKIP] {path} not found")
        return

    with open(path) as f:
        data = json.load(f)

    results = data["results"]
    ood_types = list(results.keys())
    methods = data["config"]["methods"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{OOD robustness: L1 success rate under distribution shifts.}",
        r"\label{tab:ood}",
        r"\begin{tabular}{l" + "c" * len(ood_types) + "}",
        r"\toprule",
        r"Method & " + " & ".join(ood_types) + r" \\",
        r"\midrule",
    ]

    for method in methods:
        display = METHOD_DISPLAY.get(method, method)
        if method == "ce_ais":
            display = r"\textbf{" + display + "}"
        cols = []
        for ood in ood_types:
            val = results.get(ood, {}).get(method, {}).get("chain_1", 0)
            cols.append(f"{val:.1%}")
        lines.append(f"{display} & {' & '.join(cols)} \\\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    out_path = os.path.join(output_dir, "ood_table.tex")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  [OK] {out_path}")


def generate_recovery_curve(results_dir, output_dir):
    """生成 U 型反弹恢复曲线 (PDF)。"""
    path = os.path.join(results_dir, "recovery_curve.json")
    if not os.path.exists(path):
        print(f"  [SKIP] {path} not found")
        return

    plt = setup_matplotlib()

    with open(path) as f:
        data = json.load(f)

    curves = data["curves"]
    inject_step = data["config"]["inject_step"]

    fig, ax = plt.subplots(figsize=(6, 4))

    for method, curve_data in curves.items():
        rates = curve_data.get("smoothed_rates", curve_data["raw_rates"])
        color = METHOD_COLORS.get(method, "#999999")
        display = METHOD_DISPLAY.get(method, method)
        lw = 2.5 if method == "ce_ais" else 1.5
        ax.plot(range(len(rates)), rates, label=display, color=color,
                linewidth=lw, alpha=0.9)

    # 注入点标记
    ax.axvline(x=inject_step, color="red", linestyle="--", alpha=0.5,
               label="OOD injection")

    ax.set_xlabel("Step")
    ax.set_ylabel("Instantaneous Success Rate")
    ax.set_title("Recovery Curve After OOD Injection")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim(0, len(rates) - 1)
    ax.set_ylim(-0.05, 1.05)

    out_path = os.path.join(output_dir, "recovery_curve.pdf")
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"  [OK] {out_path}")


def generate_pareto(results_dir, output_dir):
    """生成 Pareto 帕累托散点图 (PDF)。"""
    path = os.path.join(results_dir, "pareto_data.json")
    if not os.path.exists(path):
        print(f"  [SKIP] {path} not found")
        return

    plt = setup_matplotlib()

    with open(path) as f:
        data = json.load(f)

    points = data["pareto_points"]

    fig, ax = plt.subplots(figsize=(6, 4.5))

    # 分离 baseline 点 vs CE-AIS 点
    baseline_points = [p for p in points if p["n_steps"] is None]
    ceais_points = [p for p in points if p["n_steps"] is not None]

    for p in baseline_points:
        method = p["method"]
        color = METHOD_COLORS.get(method, "#999999")
        marker = METHOD_MARKERS.get(method, "o")
        display = METHOD_DISPLAY.get(method, method)
        ax.scatter(p["latency_ms"], p["success_rate"] * 100,
                   c=color, marker=marker, s=80, zorder=5,
                   label=display, edgecolors="black", linewidths=0.5)

    if ceais_points:
        lats = [p["latency_ms"] for p in ceais_points]
        srs = [p["success_rate"] * 100 for p in ceais_points]
        color = METHOD_COLORS.get("ce_ais", "#E63946")

        sorted_pairs = sorted(zip(lats, srs))
        ax.plot([x[0] for x in sorted_pairs], [x[1] for x in sorted_pairs],
                color=color, linewidth=1.5, alpha=0.6, zorder=4)

        for p in ceais_points:
            ax.scatter(p["latency_ms"], p["success_rate"] * 100,
                       c=color, marker="D", s=80, zorder=5,
                       edgecolors="black", linewidths=0.5)
            ax.annotate(f"n={p['n_steps']}",
                        (p["latency_ms"], p["success_rate"] * 100),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=7, color=color)

        ax.scatter([], [], c=color, marker="D", s=80,
                   label="CE-AIS (Ours)", edgecolors="black", linewidths=0.5)

    ax.set_xlabel("Inference Latency (ms / step)")
    ax.set_ylabel("L1 Chain Success Rate (%)")
    ax.set_title("Pareto Front: Latency vs. Success Rate")
    ax.legend(loc="lower right", framealpha=0.9)

    out_path = os.path.join(output_dir, "pareto.pdf")
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"  [OK] {out_path}")


def generate_ablation_table(results_dir, output_dir):
    """生成消融实验表 (LaTeX)。"""
    path = os.path.join(results_dir, "ablations.json")
    if not os.path.exists(path):
        path = os.path.join("logs", "ablations.json")
    if not os.path.exists(path):
        print(f"  [SKIP] ablations.json not found")
        return

    with open(path) as f:
        data = json.load(f)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation study on CE-AIS components.}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lccr}",
        r"\toprule",
        r"Configuration & L1 & L3 & Lat.(ms) \\",
        r"\midrule",
    ]

    # Full model
    bl = data.get("baseline", {})
    bl_csr = bl.get("chain_success_rate", {})
    lines.append(
        r"\textbf{CE-AIS (full)} & "
        f"{bl_csr.get('1', bl_csr.get(1, 0)):.1%} & "
        f"{bl_csr.get('3', bl_csr.get(3, 0)):.1%} & "
        f"{bl.get('latency_ms', 0):.1f} \\\\"
    )

    lines.append(r"\midrule")

    ablation_display = {
        "no_gating": "w/o Bilateral Gating",
        "mse_energy": "MSE energy (replace NCE)",
        "mamba1_backbone": "Mamba-1 backbone",
    }

    for name, res in data.get("ablations", {}).items():
        display = ablation_display.get(name, name)
        csr = res.get("chain_success_rate", {})
        lines.append(
            f"{display} & "
            f"{csr.get('1', csr.get(1, 0)):.1%} & "
            f"{csr.get('3', csr.get(3, 0)):.1%} & "
            f"{res.get('latency_ms', 0):.1f} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    out_path = os.path.join(output_dir, "ablation_table.tex")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  [OK] {out_path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("CE-AIS Paper Figure & Table Generation")
    print(f"Results: {args.results_dir}")
    print(f"Output:  {args.output_dir}")
    print("=" * 60)

    print("\n[1] Main experiment table...")
    generate_main_table(args.results_dir, args.output_dir)

    print("\n[2] OOD experiment table...")
    generate_ood_table(args.results_dir, args.output_dir)

    print("\n[3] Recovery curve (U-shaped rebound)...")
    generate_recovery_curve(args.results_dir, args.output_dir)

    print("\n[4] Pareto curve (latency vs success)...")
    generate_pareto(args.results_dir, args.output_dir)

    print("\n[5] Ablation table...")
    generate_ablation_table(args.results_dir, args.output_dir)

    print("\n" + "=" * 60)
    print("Done. Generated figures and tables:")
    for f in sorted(os.listdir(args.output_dir)):
        fpath = os.path.join(args.output_dir, f)
        size = os.path.getsize(fpath)
        print(f"  {f} ({size} bytes)")
    print("=" * 60)


if __name__ == "__main__":
    main()
