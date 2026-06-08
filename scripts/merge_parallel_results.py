"""合并多 worker 并行评估结果。"""
import argparse
import json
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-base", type=str, required=True)
    parser.add_argument("--n-workers", type=int, required=True)
    args = parser.parse_args()

    base = Path(args.output_base)
    all_method_results = {}

    for i in range(args.n_workers):
        worker_file = base / f"worker_{i}" / "main_experiment.json"
        if not worker_file.exists():
            print(f"WARNING: {worker_file} not found, skipping worker {i}")
            continue

        with open(worker_file) as f:
            data = json.load(f)

        for method_name, method_data in data.get("methods", {}).items():
            if method_name not in all_method_results:
                all_method_results[method_name] = {
                    "per_chain_completed_tasks": [],
                    "latencies": [],
                }
            completed = method_data.get("per_chain_completed_tasks", [])
            all_method_results[method_name]["per_chain_completed_tasks"].extend(completed)
            if "avg_latency_ms" in method_data:
                all_method_results[method_name]["latencies"].append(method_data["avg_latency_ms"])
            if "ce_ais_diagnostics" in method_data:
                diag = all_method_results[method_name].setdefault("ce_ais_diagnostics_all", [])
                diag.append(method_data["ce_ais_diagnostics"])

    # 计算合并指标
    merged = {"total_chains": 0, "methods": {}}

    for method_name, mdata in all_method_results.items():
        completed_tasks = mdata["per_chain_completed_tasks"]
        n = len(completed_tasks)
        merged["total_chains"] = max(merged["total_chains"], n)

        chain_successes = {}
        for length in range(1, 6):
            chain_successes[f"chain_{length}"] = sum(1 for c in completed_tasks if c >= length) / n if n else 0

        avg_completed = float(np.mean(completed_tasks)) if completed_tasks else 0
        avg_latency = float(np.mean(mdata["latencies"])) if mdata["latencies"] else 0

        dist = {}
        for i in range(6):
            dist[str(i)] = sum(1 for c in completed_tasks if c == i)

        method_summary = {
            **chain_successes,
            "avg_completed_tasks": avg_completed,
            "avg_latency_ms": avg_latency,
            "n_chains": n,
            "completed_tasks_distribution": dist,
            "per_chain_completed_tasks": completed_tasks,
        }

        # 合并 CE-AIS diagnostics
        if "ce_ais_diagnostics_all" in mdata:
            diags = mdata["ce_ais_diagnostics_all"]
            merged_diag = {}
            for key in diags[0]:
                vals = [d[key] for d in diags if key in d and isinstance(d[key], (int, float))]
                if vals:
                    merged_diag[key] = float(np.mean(vals))
            method_summary["ce_ais_diagnostics"] = merged_diag

        merged["methods"][method_name] = method_summary

    # 打印结果
    print("=" * 70)
    print(f"{'Method':<20} {'L1':>7} {'L2':>7} {'L3':>7} {'L4':>7} {'L5':>7} {'AvgLen':>7} {'Lat(ms)':>8}")
    print("-" * 70)
    for method_name, ms in merged["methods"].items():
        print(f"{method_name:<20} "
              f"{ms['chain_1']*100:>6.1f}% "
              f"{ms['chain_2']*100:>6.1f}% "
              f"{ms['chain_3']*100:>6.1f}% "
              f"{ms['chain_4']*100:>6.1f}% "
              f"{ms['chain_5']*100:>6.1f}% "
              f"{ms['avg_completed_tasks']:>7.2f} "
              f"{ms['avg_latency_ms']:>7.1f}")
    print("=" * 70)

    # 保存
    output_file = base / "merged_results.json"
    with open(output_file, "w") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()
