#!/usr/bin/env python3
"""CE-AIS Pareto 帕累托曲线实验脚本。

横轴 latency，纵轴 success rate，比较 5 种方法 + CE-AIS 不同 n_steps 配置。

输出: results/pareto_data.json

Usage:
    PYOPENGL_PLATFORM=egl uv run python scripts/evaluate_pareto.py \
        --data-dir data/task_ABC_D --n-chains 50
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description="CE-AIS Pareto Curve Experiment")
    parser.add_argument("--data-dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "data", "task_ABC_D"))
    parser.add_argument("--n-chains", type=int, default=50)
    parser.add_argument("--chain-length", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str,
                        default=os.path.join(PROJECT_ROOT, "configs", "base.yaml"))
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "results"))
    parser.add_argument("--vla-type", type=str, default="proxy",
                        choices=["openvla", "proxy"])
    parser.add_argument("--ce-ais-n-steps", type=int, nargs="*",
                        default=[1, 3, 5, 10, 20],
                        help="CE-AIS Langevin steps configs to sweep")
    return parser.parse_args()


def load_frame_state(data_dir, frame_id):
    frame_path = os.path.join(data_dir, "validation", f"episode_{frame_id:07d}.npz")
    if not os.path.exists(frame_path):
        return None, None
    data = np.load(frame_path)
    return data["robot_obs"].astype(np.float64), data["scene_obs"].astype(np.float64)


def sample_task_chains(available_tasks, n_chains, chain_length, rng):
    chains = []
    task_list = list(available_tasks)
    for _ in range(n_chains):
        chain = []
        for j in range(chain_length):
            candidates = [t for t in task_list if not chain or t != chain[-1]]
            chain.append(rng.choice(candidates))
        chains.append(chain)
    return chains


def measure_latency_and_success(policy_fn, wrapper, chains, data_dir, args, rng):
    """测量平均推理延迟和 chain_1 成功率。"""
    val_dir = os.path.join(data_dir, "validation")
    val_files = sorted(f for f in os.listdir(val_dir) if f.endswith(".npz"))
    val_frame_ids = [int(f.split("_")[1].split(".")[0]) for f in val_files]

    l1_successes = []
    latencies_per_step = []

    for ci, chain in enumerate(chains):
        frame_id = rng.choice(val_frame_ids)
        robot_obs, scene_obs = load_frame_state(data_dir, frame_id)
        wrapper.reset(task=chain[0], robot_obs=robot_obs, scene_obs=scene_obs)

        t_start = time.time()
        result = wrapper.run_chain_evaluation(
            policy_fn=policy_fn, task_chain=chain,
            max_steps_per_task=args.max_steps,
        )
        elapsed = time.time() - t_start
        total_steps = sum(r["steps"] for r in result["task_results"])
        if total_steps > 0:
            latencies_per_step.append(elapsed * 1000 / total_steps)

        l1_successes.append(result["completed_tasks"] >= 1)

    success_rate = sum(l1_successes) / len(l1_successes) if l1_successes else 0.0
    avg_latency = float(np.mean(latencies_per_step)) if latencies_per_step else 0.0

    return success_rate, avg_latency


def main():
    args = parse_args()
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)

    from src.config.config_manager import ConfigManager
    cm = ConfigManager(config_path=args.config)
    config_dict = cm.config
    device = config_dict.get("project", {}).get("device", "cuda:0")
    if not torch.cuda.is_available():
        device = "cpu"

    print("=" * 70)
    print("CE-AIS Pareto Curve (Latency vs Success Rate)")
    print(f"CE-AIS n_steps sweep: {args.ce_ais_n_steps}")
    print(f"Chains: {args.n_chains}")
    print("=" * 70)

    from src.evaluation.calvin_integration import CALVINWrapper

    vla_config = {"type": args.vla_type, "device": device,
                  "action_dim": 7, "chunk_size": 1}

    pareto_points = []

    # --- Baselines ---
    baseline_methods = ["frozen_openvla", "pdf", "tt_vla", "ada_world_policy"]

    for method_name in baseline_methods:
        print(f"\n[Pareto] {method_name}...")
        method_rng = np.random.RandomState(args.seed)

        env_config = {
            "use_real_env": True, "scene": "calvin_scene_D",
            "cameras": "static_and_gripper", "use_egl": True,
            "seed": args.seed, "max_chain_length": args.chain_length,
        }
        wrapper = CALVINWrapper(config=env_config)
        chains = sample_task_chains(wrapper.available_tasks, args.n_chains,
                                    args.chain_length, method_rng)

        from src.evaluation.baseline_framework import BASELINE_REGISTRY
        bl_cls = BASELINE_REGISTRY[method_name]
        bl = bl_cls({**vla_config, "vla_type": args.vla_type})
        bl.setup()

        def policy_fn(obs_dict, instruction, _bl=bl):
            obs_on_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                             for k, v in obs_dict.items()}
            return _bl.predict(obs_on_device, str(instruction)).squeeze(1).cpu()

        success_rate, avg_latency = measure_latency_and_success(
            policy_fn, wrapper, chains, args.data_dir, args, method_rng)

        pareto_points.append({
            "method": method_name,
            "n_steps": None,
            "success_rate": success_rate,
            "latency_ms": avg_latency,
        })
        print(f"  Success: {success_rate:.1%}, Latency: {avg_latency:.1f}ms")

        bl.teardown()
        wrapper.close()

    # --- CE-AIS with different n_steps ---
    for n_steps in args.ce_ais_n_steps:
        method_name = f"ce_ais_n{n_steps}"
        print(f"\n[Pareto] CE-AIS (n_steps={n_steps})...")
        method_rng = np.random.RandomState(args.seed)

        env_config = {
            "use_real_env": True, "scene": "calvin_scene_D",
            "cameras": "static_and_gripper", "use_egl": True,
            "seed": args.seed, "max_chain_length": args.chain_length,
        }
        wrapper = CALVINWrapper(config=env_config)
        chains = sample_task_chains(wrapper.available_tasks, args.n_chains,
                                    args.chain_length, method_rng)

        # 覆盖 n_steps
        config_override = dict(config_dict)
        if "steering" not in config_override:
            config_override["steering"] = {}
        config_override["steering"]["n_steps"] = n_steps

        from scripts.run_paper_experiments import build_ce_ais_policy
        topology = build_ce_ais_policy(config_override, vla_config, device)

        def policy_fn(obs_dict, instruction):
            obs_on_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                             for k, v in obs_dict.items()}
            action, _ = topology.safe_step(obs_on_device, str(instruction))
            return action.squeeze(1).cpu()

        success_rate, avg_latency = measure_latency_and_success(
            policy_fn, wrapper, chains, args.data_dir, args, method_rng)

        pareto_points.append({
            "method": "ce_ais",
            "n_steps": n_steps,
            "success_rate": success_rate,
            "latency_ms": avg_latency,
        })
        print(f"  Success: {success_rate:.1%}, Latency: {avg_latency:.1f}ms")
        wrapper.close()

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    output = {
        "config": {
            "n_chains": args.n_chains,
            "chain_length": args.chain_length,
            "seed": args.seed,
            "ce_ais_n_steps": args.ce_ais_n_steps,
        },
        "pareto_points": pareto_points,
    }
    output_path = os.path.join(args.output_dir, "pareto_data.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nPareto data saved: {output_path}")

    # 打印散点表
    print("\n" + "=" * 50)
    print(f"{'Method':<25} {'Success':>10} {'Latency':>10}")
    print("-" * 50)
    for p in sorted(pareto_points, key=lambda x: x["latency_ms"]):
        label = p["method"]
        if p["n_steps"] is not None:
            label += f" (n={p['n_steps']})"
        print(f"{label:<25} {p['success_rate']:>9.1%} {p['latency_ms']:>9.1f}ms")
    print("=" * 50)


if __name__ == "__main__":
    main()
