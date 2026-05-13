#!/usr/bin/env python3
"""CE-AIS OOD 干扰实验脚本。

在 3 类 OOD 干扰（视觉/物理/相机）下，每类 50 个 episode，
对比 CE-AIS 与 baseline 的成功率。

输出: results/ood_experiment.json

Usage:
    PYOPENGL_PLATFORM=egl uv run python scripts/evaluate_ood.py \
        --data-dir data/task_ABC_D --n-episodes 50
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
    parser = argparse.ArgumentParser(description="CE-AIS OOD Perturbation Experiment")
    parser.add_argument("--data-dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "data", "task_ABC_D"))
    parser.add_argument("--n-episodes", type=int, default=50,
                        help="Episodes per OOD type")
    parser.add_argument("--chain-length", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str,
                        default=os.path.join(PROJECT_ROOT, "configs", "base.yaml"))
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "results"))
    parser.add_argument("--vla-type", type=str, default="proxy",
                        choices=["openvla", "proxy"])
    parser.add_argument("--methods", type=str, nargs="*",
                        default=["ce_ais", "frozen_openvla", "pdf"])
    return parser.parse_args()


OOD_SCENARIOS = {
    "physics": {
        "description": "物理动力学突变 (mass x2.0, friction x0.3)",
        "inject_args": {"perturbation_type": "physics", "mass_scale": 2.0, "friction_scale": 0.3},
    },
    "visual": {
        "description": "视觉灾难 (brightness=0.5, noise=0.1)",
        "inject_args": {"perturbation_type": "visual", "brightness": 0.5, "noise_std": 0.1},
    },
    "camera": {
        "description": "相机姿态偏转 (offset=0.05)",
        "inject_args": {"perturbation_type": "camera", "camera_offset": 0.05},
    },
}


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


def run_ood_evaluation(method_name, policy_fn, wrapper, chains,
                       data_dir, ood_type, ood_cfg, args, rng):
    """在特定 OOD 干扰下评估方法。"""
    val_dir = os.path.join(data_dir, "validation")
    val_files = sorted(f for f in os.listdir(val_dir) if f.endswith(".npz"))
    val_frame_ids = [int(f.split("_")[1].split(".")[0]) for f in val_files]

    chain_successes = {i: [] for i in range(1, args.chain_length + 1)}
    all_results = []

    for ci, chain in enumerate(chains):
        frame_id = rng.choice(val_frame_ids)
        robot_obs, scene_obs = load_frame_state(data_dir, frame_id)
        wrapper.reset(task=chain[0], robot_obs=robot_obs, scene_obs=scene_obs)

        inject_args = dict(ood_cfg["inject_args"])
        p_type = inject_args.pop("perturbation_type")
        wrapper.inject_ood(p_type, **inject_args)

        result = wrapper.run_chain_evaluation(
            policy_fn=policy_fn,
            task_chain=chain,
            max_steps_per_task=args.max_steps,
        )
        all_results.append(result)
        completed = result["completed_tasks"]
        for length in range(1, args.chain_length + 1):
            chain_successes[length].append(completed >= length)

    results_summary = {}
    for length in range(1, args.chain_length + 1):
        s = chain_successes[length]
        results_summary[f"chain_{length}"] = sum(s) / len(s) if s else 0.0
    results_summary["avg_completed_tasks"] = float(
        np.mean([r["completed_tasks"] for r in all_results]))
    return results_summary


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
    print("CE-AIS OOD Perturbation Experiments")
    print(f"OOD Types: {list(OOD_SCENARIOS.keys())}")
    print(f"Episodes per type: {args.n_episodes}")
    print("=" * 70)

    from src.evaluation.calvin_integration import CALVINWrapper

    vla_config = {
        "type": args.vla_type,
        "device": device,
        "action_dim": 7,
        "chunk_size": 1,
    }

    all_results = {}

    for ood_type, ood_cfg in OOD_SCENARIOS.items():
        print(f"\n{'='*60}")
        print(f"OOD Scenario: {ood_type} — {ood_cfg['description']}")
        print(f"{'='*60}")

        all_results[ood_type] = {}

        for method_name in args.methods:
            print(f"\n  [{method_name}] Evaluating under {ood_type} OOD...")
            method_rng = np.random.RandomState(args.seed)

            env_config = {
                "use_real_env": True, "scene": "calvin_scene_D",
                "cameras": "static_and_gripper", "use_egl": True,
                "seed": args.seed, "max_chain_length": args.chain_length,
            }
            wrapper = CALVINWrapper(config=env_config)

            chains = sample_task_chains(wrapper.available_tasks, args.n_episodes,
                                        args.chain_length, method_rng)

            if method_name == "ce_ais":
                from scripts.run_paper_experiments import build_ce_ais_policy
                topology = build_ce_ais_policy(config_dict, vla_config, device)

                def policy_fn(obs_dict, instruction):
                    obs_on_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                                     for k, v in obs_dict.items()}
                    action, _ = topology.safe_step(obs_on_device, str(instruction))
                    return action.squeeze(1).cpu()
            else:
                from src.evaluation.baseline_framework import BASELINE_REGISTRY
                bl_cls = BASELINE_REGISTRY[method_name]
                bl = bl_cls({**vla_config, "vla_type": args.vla_type})
                bl.setup()

                def policy_fn(obs_dict, instruction, _bl=bl):
                    obs_on_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                                     for k, v in obs_dict.items()}
                    return _bl.predict(obs_on_device, str(instruction)).squeeze(1).cpu()

            results = run_ood_evaluation(
                method_name, policy_fn, wrapper, chains,
                args.data_dir, ood_type, ood_cfg, args, method_rng)

            all_results[ood_type][method_name] = results
            print(f"    L1={results['chain_1']:.1%}, "
                  f"avg_completed={results['avg_completed_tasks']:.2f}")

            if method_name != "ce_ais" and hasattr(bl, "teardown"):
                bl.teardown()
            wrapper.close()

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    output = {
        "config": {"n_episodes": args.n_episodes, "seed": args.seed,
                    "methods": args.methods, "ood_scenarios": list(OOD_SCENARIOS.keys())},
        "results": all_results,
    }
    output_path = os.path.join(args.output_dir, "ood_experiment.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nOOD results saved: {output_path}")

    # 汇总表
    print("\n" + "=" * 70)
    print(f"{'Method':<20}", end="")
    for ood in OOD_SCENARIOS:
        print(f" {ood:>12}", end="")
    print()
    print("-" * 56)
    for method in args.methods:
        print(f"{method:<20}", end="")
        for ood in OOD_SCENARIOS:
            r = all_results.get(ood, {}).get(method, {})
            print(f" {r.get('chain_1', 0):>11.1%}", end="")
        print()
    print("=" * 70)


if __name__ == "__main__":
    main()
