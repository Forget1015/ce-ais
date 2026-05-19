#!/usr/bin/env python3
"""CE-AIS OOD 干扰实验脚本。

在 3 类 OOD 干扰（视觉/物理/相机）下，每类 50 个 episode，
对比 CE-AIS 与 baseline 的成功率。

输出: results/ood_experiment.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.eval_common import (
    add_common_eval_args,
    build_eval_vla_config,
    configure_egl,
    eval_metadata,
    resolve_eval_device,
    to_device_obs,
)


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
    parser.add_argument("--methods", type=str, nargs="*",
                        default=["ce_ais", "frozen_vla", "pdf"])
    parser.add_argument("--ood-types", type=str, nargs="*",
                        default=list(OOD_SCENARIOS.keys()),
                        choices=list(OOD_SCENARIOS.keys()))
    parser.add_argument("--severity-sweep", action="store_true",
                        help="Evaluate mild/medium/severe settings for selected OOD types")
    parser.add_argument("--severities", type=str, nargs="*",
                        default=["mild", "medium", "severe"],
                        choices=["mild", "medium", "severe"])
    parser.add_argument("--progress-interval", type=int, default=10,
                        help="Print progress every N episodes; set 0 to disable")
    add_common_eval_args(parser, PROJECT_ROOT)
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


OOD_SEVERITY_SCENARIOS = {
    "physics": {
        "mild": {"description": "物理轻度偏移 (mass x1.2, friction x0.8)",
                 "inject_args": {"perturbation_type": "physics", "mass_scale": 1.2, "friction_scale": 0.8}},
        "medium": {"description": "物理中度偏移 (mass x1.5, friction x0.5)",
                   "inject_args": {"perturbation_type": "physics", "mass_scale": 1.5, "friction_scale": 0.5}},
        "severe": OOD_SCENARIOS["physics"],
    },
    "visual": {
        "mild": {"description": "视觉轻度干扰 (brightness=0.8, noise=0.02)",
                 "inject_args": {"perturbation_type": "visual", "brightness": 0.8, "noise_std": 0.02}},
        "medium": {"description": "视觉中度干扰 (brightness=0.6, noise=0.05)",
                   "inject_args": {"perturbation_type": "visual", "brightness": 0.6, "noise_std": 0.05}},
        "severe": OOD_SCENARIOS["visual"],
    },
    "camera": {
        "mild": {"description": "相机轻度偏转 (offset=0.01)",
                 "inject_args": {"perturbation_type": "camera", "camera_offset": 0.01}},
        "medium": {"description": "相机中度偏转 (offset=0.03)",
                   "inject_args": {"perturbation_type": "camera", "camera_offset": 0.03}},
        "severe": OOD_SCENARIOS["camera"],
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
        for _ in range(chain_length):
            candidates = [t for t in task_list if not chain or t != chain[-1]]
            chain.append(rng.choice(candidates))
        chains.append(chain)
    return chains


def make_ood_injector(wrapper, ood_cfg):
    def inject():
        inject_args = dict(ood_cfg["inject_args"])
        p_type = inject_args.pop("perturbation_type")
        wrapper.inject_ood(p_type, **inject_args)
    return inject


def run_ood_evaluation(method_name, policy_fn, wrapper, chains,
                       data_dir, ood_cfg, args, rng, policy_reset_fn=None):
    """在特定 OOD 干扰下评估方法。"""
    val_dir = os.path.join(data_dir, "validation")
    val_files = sorted(f for f in os.listdir(val_dir) if f.endswith(".npz"))
    val_frame_ids = [int(f.split("_")[1].split(".")[0]) for f in val_files]

    chain_successes = {i: [] for i in range(1, args.chain_length + 1)}
    all_results = []
    completed_tasks = []

    for episode_idx, chain in enumerate(chains, start=1):
        frame_id = rng.choice(val_frame_ids)
        robot_obs, scene_obs = load_frame_state(data_dir, frame_id)
        result = wrapper.run_chain_evaluation(
            policy_fn=policy_fn,
            task_chain=chain,
            max_steps_per_task=args.max_steps,
            initial_robot_obs=robot_obs,
            initial_scene_obs=scene_obs,
            policy_reset_fn=policy_reset_fn,
            post_reset_fn=make_ood_injector(wrapper, ood_cfg),
        )
        all_results.append(result)
        completed = result["completed_tasks"]
        completed_tasks.append(completed)
        for length in range(1, args.chain_length + 1):
            chain_successes[length].append(completed >= length)

        if args.progress_interval > 0 and (
            episode_idx % args.progress_interval == 0 or episode_idx == len(chains)
        ):
            current_l1 = sum(chain_successes[1]) / len(chain_successes[1])
            avg_completed = float(np.mean([r["completed_tasks"] for r in all_results]))
            print(
                f"    progress {method_name}: {episode_idx}/{len(chains)} episodes, "
                f"L1={current_l1:.1%}, avg_completed={avg_completed:.2f}",
                flush=True,
            )

    results_summary = {}
    for length in range(1, args.chain_length + 1):
        s = chain_successes[length]
        results_summary[f"chain_{length}"] = sum(s) / len(s) if s else 0.0
    results_summary["avg_completed_tasks"] = float(
        np.mean([r["completed_tasks"] for r in all_results]))
    results_summary["per_chain_completed_tasks"] = completed_tasks
    results_summary["completed_tasks_distribution"] = {
        str(i): int(sum(1 for c in completed_tasks if c == i))
        for i in range(args.chain_length + 1)
    }
    return results_summary


def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)

    from src.config.config_manager import ConfigManager
    cm = ConfigManager(config_path=args.config)
    config_dict = cm.config
    device, device_index, requested_device = resolve_eval_device(args, config_dict)
    use_egl = configure_egl(args, device_index)

    selected_scenarios = []
    for ood_type in args.ood_types:
        if args.severity_sweep:
            for severity in args.severities:
                selected_scenarios.append((ood_type, severity, OOD_SEVERITY_SCENARIOS[ood_type][severity]))
        else:
            selected_scenarios.append((ood_type, None, OOD_SCENARIOS[ood_type]))

    print("=" * 70)
    print("CE-AIS OOD Perturbation Experiments")
    print(f"OOD Types: {args.ood_types}")
    print(f"Severity sweep: {args.severity_sweep} ({args.severities if args.severity_sweep else 'fixed'})")
    print(f"Episodes per type: {args.n_episodes}")
    print(f"Methods: {args.methods}")
    print(f"Device: {device} (requested={requested_device}, visible={os.environ.get('CUDA_VISIBLE_DEVICES')})")
    print(f"EGL: {'disabled' if not use_egl else 'enabled'} (EGL_VISIBLE_DEVICES={os.environ.get('EGL_VISIBLE_DEVICES')})")
    print("=" * 70)

    from src.evaluation.calvin_integration import CALVINWrapper

    vla_config = build_eval_vla_config(args, device)
    all_results = {}

    for ood_type, severity, ood_cfg in selected_scenarios:
        label = f"{ood_type}/{severity}" if severity else ood_type
        print(f"\n{'='*60}")
        print(f"OOD Scenario: {label} — {ood_cfg['description']}")
        print(f"{'='*60}")

        if args.severity_sweep:
            all_results.setdefault(ood_type, {})[severity] = {}
            result_bucket = all_results[ood_type][severity]
        else:
            all_results[ood_type] = {}
            result_bucket = all_results[ood_type]

        for method_name in args.methods:
            print(f"\n  [{method_name}] Evaluating under {label} OOD...")
            method_rng = np.random.RandomState(args.seed)

            env_config = {
                "use_real_env": True,
                "scene": "calvin_scene_D",
                "cameras": "static_and_gripper",
                "use_egl": use_egl,
                "seed": args.seed,
                "max_chain_length": args.chain_length,
            }
            wrapper = CALVINWrapper(config=env_config)
            chains = sample_task_chains(wrapper.available_tasks, args.n_episodes,
                                        args.chain_length, method_rng)

            bl = None
            topology = None
            if method_name == "ce_ais":
                from scripts.run_paper_experiments import build_ce_ais_policy
                topology = build_ce_ais_policy(
                    config_dict, vla_config, device, args.encoder_ckpt, args.cewm_ckpt
                )
                topology.reset_diagnostics()

                def policy_fn(obs_dict, instruction):
                    obs_on_device = to_device_obs(obs_dict, device)
                    action, _ = topology.safe_step(obs_on_device, str(instruction))
                    return action.squeeze(1).cpu()

                policy_reset_fn = getattr(topology, "reset", None)
            else:
                from src.evaluation.baseline_framework import BASELINE_REGISTRY
                bl_cls = BASELINE_REGISTRY[method_name]
                bl = bl_cls({**vla_config, "vla_type": args.vla_type})
                bl.setup()

                def policy_fn(obs_dict, instruction, _bl=bl):
                    obs_on_device = to_device_obs(obs_dict, device)
                    return _bl.predict(obs_on_device, str(instruction)).squeeze(1).cpu()

                policy_reset_fn = getattr(bl, "reset_task", None)

            results = run_ood_evaluation(
                method_name, policy_fn, wrapper, chains,
                args.data_dir, ood_cfg, args, method_rng,
                policy_reset_fn=policy_reset_fn,
            )
            if topology is not None:
                results["ce_ais_diagnostics"] = topology.get_diagnostics()

            result_bucket[method_name] = results
            print(f"    L1={results['chain_1']:.1%}, "
                  f"avg_completed={results['avg_completed_tasks']:.2f}")

            if bl is not None and hasattr(bl, "teardown"):
                bl.teardown()
            wrapper.close()

    os.makedirs(args.output_dir, exist_ok=True)
    output = {
        "config": {
            "n_episodes": args.n_episodes,
            "chain_length": args.chain_length,
            "max_steps_per_task": args.max_steps,
            "seed": args.seed,
            "methods": args.methods,
            "ood_scenarios": args.ood_types,
            "severity_sweep": args.severity_sweep,
            "severities": args.severities if args.severity_sweep else [],
            **eval_metadata(args, device, requested_device, use_egl),
        },
        "results": all_results,
    }
    output_path = os.path.join(args.output_dir, "ood_experiment.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nOOD results saved: {output_path}")

    print("\n" + "=" * 70)
    print(f"{'Method':<20}", end="")
    for ood_type, severity, _ in selected_scenarios:
        label = f"{ood_type}/{severity}" if severity else ood_type
        print(f" {label:>16}", end="")
    print()
    print("-" * 70)
    for method in args.methods:
        print(f"{method:<20}", end="")
        for ood_type, severity, _ in selected_scenarios:
            if severity:
                r = all_results.get(ood_type, {}).get(severity, {}).get(method, {})
            else:
                r = all_results.get(ood_type, {}).get(method, {})
            print(f" {r.get('chain_1', 0):>15.1%}", end="")
        print()
    print("=" * 70)


if __name__ == "__main__":
    main()
