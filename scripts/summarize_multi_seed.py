#!/usr/bin/env python3
"""汇总 multi-seed evaluation 结果。"""
import json, re, sys, os
from pathlib import Path
import numpy as np

result_dir = Path("results/multi_seed_eval")

results = []
for log_file in sorted(result_dir.glob("seed_*.log")):
    seed = int(re.search(r"seed_(\d+)", log_file.name).group(1))
    content = log_file.read_text()

    # Check if complete
    if "1000/1000" not in content:
        chains_done = len(re.findall(r"chain \d+/1000 task 1/5 (success|failed)", content))
        print(f"  seed={seed}: INCOMPLETE ({chains_done}/1000 chains)")
        continue

    # Parse per-level success
    levels = {}
    for length in range(1, 6):
        pattern = rf"L{length}[= ]+(\d+\.?\d*)%"
        m = re.search(pattern, content)
        if m:
            levels[f"L{length}"] = float(m.group(1))

    # If not found in summary, count from chain results
    if not levels:
        successes = {i: 0 for i in range(1, 6)}
        total = 0
        for m in re.finditer(r"chain (\d+)/1000.*?(?:task (\d+)/5 (success|failed))", content, re.DOTALL):
            pass  # fallback method below

    # Try parsing from the final summary line
    summary_match = re.search(
        r"frozen_flower\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%",
        content
    )
    if summary_match:
        levels = {f"L{i+1}": float(summary_match.group(i+1)) for i in range(5)}

    if levels:
        avg_len = sum(levels.get(f"L{i}", 0) for i in range(1, 6)) / 100.0
        results.append({"seed": seed, **levels, "avg_len": avg_len})
        print(f"  seed={seed}: L1={levels.get('L1',0):.1f}% avg={avg_len:.2f}")

if not results:
    print("No completed results found yet.")
    sys.exit(0)

# Statistics
avg_lens = [r["avg_len"] for r in results]
l1s = [r.get("L1", 0) for r in results]

print(f"\n{'='*60}")
print(f"SUMMARY: {len(results)} seeds completed")
print(f"{'='*60}")
print(f"Avg Length:  mean={np.mean(avg_lens):.3f} ± {np.std(avg_lens):.3f}  "
      f"[min={np.min(avg_lens):.3f}, max={np.max(avg_lens):.3f}]")
print(f"L1 Success:  mean={np.mean(l1s):.1f}% ± {np.std(l1s):.1f}%  "
      f"[min={np.min(l1s):.1f}%, max={np.max(l1s):.1f}%]")

# Per-level stats
for lvl in range(1, 6):
    key = f"L{lvl}"
    vals = [r.get(key, 0) for r in results]
    print(f"  {key}: {np.mean(vals):.1f}% ± {np.std(vals):.1f}%")

# Best seed
best = max(results, key=lambda r: r["avg_len"])
print(f"\nBest seed: {best['seed']} (avg_len={best['avg_len']:.3f})")

# Save summary
summary = {
    "num_seeds": len(results),
    "avg_len_mean": float(np.mean(avg_lens)),
    "avg_len_std": float(np.std(avg_lens)),
    "avg_len_min": float(np.min(avg_lens)),
    "avg_len_max": float(np.max(avg_lens)),
    "l1_mean": float(np.mean(l1s)),
    "best_seed": best["seed"],
    "best_avg_len": best["avg_len"],
    "per_seed": results,
}
with open(result_dir / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved to {result_dir / 'summary.json'}")
