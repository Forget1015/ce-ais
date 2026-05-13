#!/usr/bin/env python3
"""CE-AIS U 型反弹恢复曲线实验脚本。

在执行第 inject_step 步注入 OOD 干扰，追踪后续步的任务成功率变化，
生成"CE-AIS 反弹 vs baseline 永久下跌"的恢复曲线。

输出: results/recovery_curve.json

Usage:
    PYOPENGL_PLATFORM=egl uv run python scripts/evaluate_recovery.py \
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
    parser = argparse.ArgumentParser(description="CE-AIS Recovery Curve Experiment")
    parser.add_argument("--data-dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "data", "task_ABC_D"))
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=100,
                        help="Total steps per episode (pre + post injection)")
    parser.add_argument("--inject-step", type=int, default=50,
                        help="Step at which to inject OOD perturbation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str,
                        default=os.path.join(PROJECT_ROOT, "configs", "base.yaml"))
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "results"))
    parser.add_argument("--vla-type", type=str, default="proxy",
                        choices=["openvla", "proxy"])
    parser.add_argument("--methods", type=str, nargs="*",
                        default=["ce_ais", "frozen_openvla", "pdf"])
    parser.add_argument("--ood-type", type=str, default="physics",
                        choices=["physics", "visual", "camera"])
    parser.add_argument("--window-size", type=int, default=5,
                        help="Rolling window size for smoothing")
    return parser.parse_args()


def load_frame_state(data_dir, frame_id):
    frame_path = os.path.join(data_dir, "validation", f"episode_{frame_id:07d}.npz")
    if not os.path.exists(frame_path):
        return None, None
    data = np.load(frame_path)
    return data["robot_obs"].astype(np.float64), data["scene_obs"].astype(np.float64)


OOD_INJECT_MAP = {
    "physics": {"perturbation_type": "physics", "mass_scale": 2.0, "friction_scale": 0.3},
    "visual": {"perturbation_type": "visual", "brightness": 0.5, "noise_std": 0.1},
    "camera": {"perturbation_type": "camera", "camera_offset": 0.05},
}


def run_recovery_episode(policy_fn, wrapper, task_name, inject_step, max_steps,
                          ood_type, data_dir, rng, val_frame_ids):
    """单个 episode 的恢复曲线追踪。

    Returns:
        step_energies: list of (step, task_success_indicator) tuples
    """
    frame_id = rng.choice(val_frame_ids)
    robot_obs, scene_obs = load_frame_state(data_dir, frame_id)
    obs = wrapper.reset(task=task_name, robot_obs=robot_obs, scene_obs=scene_obs)

    step_records = []
    injected = False

    for step in range(max_steps):
        if step == inject_step and not injected:
            inject_args = dict(OOD_INJECT_MAP[ood_type])
            p_type = inject_args.pop("perturbation_type")
            wrapper.inject_ood(p_type, **inject_args)
            injected = True

        obs_dict = {"rgb": obs.rgb, "depth": obs.depth, "pose": obs.pose}
        action = policy_fn(obs_dict, task_name)
        obs, reward, done, info = wrapper.step(action)

        step_records.append({
            "step": step,
            "reward": reward,
            "success": reward > 0,
            "post_injection": step >= inject_step,
        })

        if done:
            for remaining in range(step + 1, max_steps):
                step_records.append({
                    "step": remaining,
                    "reward": 0.0,
                    "success": False,
                    "post_injection": remaining >= inject_step,
                })
            break

    return step_records


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
    print("CE-AIS Recovery Curve (U-shaped Rebound) Experiment")
    print(f"OOD injection: {args.ood_type} at step {args.inject_step}")
    print(f"Episodes: {args.n_episodes}, Total steps: {args.max_steps}")
    print("=" * 70)

    from src.evaluation.calvin_integration import CALVINWrapper

    vla_config = {"type": args.vla_type, "device": device,
                  "action_dim": 7, "chunk_size": 1}

    all_method_curves = {}

    for method_name in args.methods:
        print(f"\n[{method_name}] Running recovery experiments...")
        method_rng = np.random.RandomState(args.seed)

        env_config = {
            "use_real_env": True, "scene": "calvin_scene_D",
            "cameras": "static_and_gripper", "use_egl": True,
            "seed": args.seed, "max_chain_length": 5,
        }
        wrapper = CALVINWrapper(config=env_config)

        val_dir = os.path.join(args.data_dir, "validation")
        val_files = sorted(f for f in os.listdir(val_dir) if f.endswith(".npz"))
        val_frame_ids = [int(f.split("_")[1].split(".")[0]) for f in val_files]

        tasks = wrapper.available_tasks
        if not tasks:
            tasks = ["pick_up_object"]

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

        # 收集每步的成功率
        all_step_success = [[] for _ in range(args.max_steps)]

        for ep in range(args.n_episodes):
            task_name = method_rng.choice(tasks)
            records = run_recovery_episode(
                policy_fn, wrapper, task_name, args.inject_step,
                args.max_steps, args.ood_type, args.data_dir,
                method_rng, val_frame_ids)

            for rec in records:
                s = rec["step"]
                if s < args.max_steps:
                    all_step_success[s].append(1.0 if rec["reward"] > 0 else 0.0)

        # 计算逐步成功率 + 滑动平均
        step_rates = []
        for s in range(args.max_steps):
            vals = all_step_success[s]
            rate = np.mean(vals) if vals else 0.0
            step_rates.append(rate)

        # 滑动窗口平滑
        w = args.window_size
        smoothed = []
        for i in range(len(step_rates)):
            window = step_rates[max(0, i - w):i + 1]
            smoothed.append(float(np.mean(window)))

        all_method_curves[method_name] = {
            "raw_rates": step_rates,
            "smoothed_rates": smoothed,
            "inject_step": args.inject_step,
        }

        pre_avg = np.mean(step_rates[:args.inject_step]) if args.inject_step > 0 else 0
        post_avg = np.mean(step_rates[args.inject_step:]) if args.inject_step < len(step_rates) else 0
        print(f"  Pre-injection avg: {pre_avg:.3f}")
        print(f"  Post-injection avg: {post_avg:.3f}")
        print(f"  Recovery delta: {post_avg - pre_avg:+.3f}")

        if method_name != "ce_ais" and hasattr(bl, "teardown"):
            bl.teardown()
        wrapper.close()

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    output = {
        "config": {
            "ood_type": args.ood_type,
            "inject_step": args.inject_step,
            "max_steps": args.max_steps,
            "n_episodes": args.n_episodes,
            "seed": args.seed,
            "window_size": args.window_size,
        },
        "curves": all_method_curves,
    }
    output_path = os.path.join(args.output_dir, "recovery_curve.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nRecovery curve data saved: {output_path}")


if __name__ == "__main__":
    main()
