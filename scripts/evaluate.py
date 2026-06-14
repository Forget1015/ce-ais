#!/usr/bin/env python3
"""CE-AIS CALVIN 评估脚本。

支持:
- ABC→D 零样本多任务链式评估
- OOD 干扰注入（物理/视觉/相机）
- 多策略对比（CE-AIS / frozen OpenVLA / proxy）

Usage:
    # 基础评估（随机任务链）
    PYOPENGL_PLATFORM=egl uv run python scripts/evaluate.py --n-chains 10

    # OOD 注入
    PYOPENGL_PLATFORM=egl uv run python scripts/evaluate.py --n-chains 10 --ood visual --ood-brightness 0.3

    # 使用 proxy VLA（无需 OpenVLA 权重）
    PYOPENGL_PLATFORM=egl uv run python scripts/evaluate.py --n-chains 5 --vla-type proxy
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
    parser = argparse.ArgumentParser(description="CE-AIS CALVIN Evaluation")
    parser.add_argument("--scene", type=str, default="calvin_scene_D")
    parser.add_argument("--data-dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "data", "calvin_debug_dataset"))
    parser.add_argument("--n-chains", type=int, default=10,
                        help="Number of task chains to evaluate")
    parser.add_argument("--chain-length", type=int, default=5,
                        help="Tasks per chain")
    parser.add_argument("--max-steps", type=int, default=360,
                        help="Max steps per task")
    parser.add_argument("--seed", type=int, default=42)

    # VLA
    parser.add_argument("--vla-type", type=str, default="proxy",
                        choices=["openvla", "proxy"],
                        help="VLA adapter type")
    parser.add_argument("--vla-model", type=str, default=None,
                        help="OpenVLA model path (default: openvla/openvla-7b)")
    parser.add_argument("--vla-dtype", type=str, default="bf16")

    # OOD
    parser.add_argument("--ood", type=str, nargs="*", default=[],
                        choices=["physics", "visual", "camera"])
    parser.add_argument("--ood-mass", type=float, default=2.0)
    parser.add_argument("--ood-friction", type=float, default=0.3)
    parser.add_argument("--ood-brightness", type=float, default=0.3)
    parser.add_argument("--ood-noise", type=float, default=0.1)
    parser.add_argument("--ood-camera", type=float, default=0.05)

    # Steering
    parser.add_argument("--no-steering", action="store_true",
                        help="Disable CE-AIS steering (baseline mode)")

    # Output
    parser.add_argument("--output-dir", type=str, default=os.path.join(PROJECT_ROOT, "results"))
    return parser.parse_args()


def load_validation_annotations(data_dir: str):
    """从 CALVIN 数据集加载验证集注释。"""
    ann_path = os.path.join(data_dir, "validation", "lang_annotations", "auto_lang_ann.npy")
    if os.path.exists(ann_path):
        ann = np.load(ann_path, allow_pickle=True).item()
        return ann
    return None


def load_frame_state(data_dir: str, frame_id: int):
    """加载指定帧的机器人和场景状态。"""
    frame_path = os.path.join(data_dir, "validation", f"episode_{frame_id:07d}.npz")
    if not os.path.exists(frame_path):
        return None, None
    data = np.load(frame_path)
    return data["robot_obs"].astype(np.float64), data["scene_obs"].astype(np.float64)


def sample_task_chains(available_tasks, n_chains, chain_length, rng):
    """随机生成任务链（每个链内任务不重复连续）。"""
    chains = []
    task_list = list(available_tasks)
    for _ in range(n_chains):
        chain = []
        for j in range(chain_length):
            candidates = [t for t in task_list if not chain or t != chain[-1]]
            chain.append(rng.choice(candidates))
        chains.append(chain)
    return chains


def main():
    args = parse_args()
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 60)
    print("CE-AIS CALVIN Evaluation")
    print("=" * 60)

    # ---- 1. 初始化 CALVIN 环境 ----
    print("\n[1/4] Initializing CALVIN environment...")
    from src.evaluation.calvin_integration import CALVINWrapper

    env_config = {
        "use_real_env": True,
        "scene": args.scene,
        "cameras": "static_and_gripper",
        "use_egl": True,
        "seed": args.seed,
        "max_chain_length": args.chain_length,
    }
    wrapper = CALVINWrapper(config=env_config)
    print(f"  Scene: {args.scene}")
    print(f"  Available tasks: {len(wrapper.available_tasks)}")

    # ---- 2. 初始化策略 ----
    print("\n[2/4] Loading policy...")
    from src.dual_stream.vla_adapter import build_vla_adapter

    vla_config = {
        "type": args.vla_type,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "action_dim": 7,
        "chunk_size": 1,
    }
    if args.vla_type == "openvla":
        vla_config["model_path"] = args.vla_model
        vla_config["dtype"] = args.vla_dtype
    vla = build_vla_adapter(vla_config)
    print(f"  VLA: {args.vla_type}")
    print(f"  Steering: {'disabled' if args.no_steering else 'enabled'}")

    def policy_fn(obs_dict, instruction):
        device = vla_config["device"]
        obs_on_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in obs_dict.items()}
        action = vla.predict(obs_on_device, str(instruction))
        return action.squeeze(1).cpu()  # [B, 7]

    # ---- 3. OOD 注入 ----
    if args.ood:
        print(f"\n  OOD perturbations: {args.ood}")
        for ood_type in args.ood:
            if ood_type == "physics":
                wrapper.inject_ood("physics", mass_scale=args.ood_mass,
                                   friction_scale=args.ood_friction)
            elif ood_type == "visual":
                wrapper.inject_ood("visual", brightness=args.ood_brightness,
                                   noise_std=args.ood_noise)
            elif ood_type == "camera":
                wrapper.inject_ood("camera", camera_offset=args.ood_camera)

    # ---- 4. 生成任务链并评估 ----
    print(f"\n[3/4] Running evaluation ({args.n_chains} chains x {args.chain_length} tasks)...")

    # 获取验证集帧 ID 用于重置环境
    val_dir = os.path.join(args.data_dir, "validation")
    val_files = sorted(f for f in os.listdir(val_dir) if f.endswith(".npz"))
    val_frame_ids = [int(f.split("_")[1].split(".")[0]) for f in val_files]

    chains = sample_task_chains(wrapper.available_tasks, args.n_chains,
                                args.chain_length, rng)

    all_results = []
    chain_successes = {i: [] for i in range(1, args.chain_length + 1)}

    for ci, chain in enumerate(chains):
        # 随机选一个初始帧
        frame_id = rng.choice(val_frame_ids)
        robot_obs, scene_obs = load_frame_state(args.data_dir, frame_id)

        # 重置到该帧状态
        wrapper.reset(task=chain[0], robot_obs=robot_obs, scene_obs=scene_obs)

        result = wrapper.run_chain_evaluation(
            policy_fn=policy_fn,
            task_chain=chain,
            max_steps_per_task=args.max_steps,
        )
        all_results.append(result)

        completed = result["completed_tasks"]
        for length in range(1, args.chain_length + 1):
            chain_successes[length].append(completed >= length)

        status = "PASS" if result["chain_success"] else f"FAIL@{completed}"
        tasks_str = " → ".join(chain[:3]) + ("..." if len(chain) > 3 else "")
        print(f"  Chain {ci+1:3d}/{args.n_chains}: {status:8s} ({completed}/{len(chain)}) [{tasks_str}]")

    # ---- 5. 汇总结果 ----
    print(f"\n[4/4] Results Summary")
    print("=" * 60)

    print(f"\n{'Chain Length':<15} {'Success Rate':>15} {'Count':>10}")
    print("-" * 40)
    results_summary = {}
    for length in range(1, args.chain_length + 1):
        successes = chain_successes[length]
        rate = sum(successes) / len(successes) if successes else 0.0
        results_summary[f"chain_{length}"] = rate
        print(f"  {length:<13d} {rate:>14.1%} {sum(successes):>6d}/{len(successes)}")

    avg_completed = np.mean([r["completed_tasks"] for r in all_results])
    results_summary["avg_completed_tasks"] = float(avg_completed)
    print(f"\n  Avg completed tasks: {avg_completed:.2f} / {args.chain_length}")

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    output = {
        "config": {
            "scene": args.scene,
            "vla_type": args.vla_type,
            "steering": not args.no_steering,
            "n_chains": args.n_chains,
            "chain_length": args.chain_length,
            "max_steps_per_task": args.max_steps,
            "ood": args.ood,
            "seed": args.seed,
        },
        "results": results_summary,
        "chain_details": all_results,
    }
    output_path = os.path.join(args.output_dir, "calvin_eval.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved: {output_path}")

    wrapper.close()
    print("=" * 60)


if __name__ == "__main__":
    main()
