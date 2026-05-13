#!/usr/bin/env python3
"""CE-AIS 论文主实验脚本。

运行 CE-AIS vs 4 个 baseline 在 ABC→D 协议下的完整对比实验。

输出: results/main_experiment.json

Usage:
    PYOPENGL_PLATFORM=egl uv run python scripts/run_paper_experiments.py \
        --data-dir data/task_ABC_D --n-chains 200
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
    parser = argparse.ArgumentParser(description="CE-AIS Paper Main Experiments")
    parser.add_argument("--data-dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "data", "task_ABC_D"))
    parser.add_argument("--n-chains", type=int, default=200)
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
                        default=["ce_ais", "frozen_openvla", "pdf", "tt_vla", "ada_world_policy"])
    return parser.parse_args()


def load_validation_annotations(data_dir: str):
    ann_path = os.path.join(data_dir, "validation", "lang_annotations", "auto_lang_ann.npy")
    if os.path.exists(ann_path):
        return np.load(ann_path, allow_pickle=True).item()
    return None


def load_frame_state(data_dir: str, frame_id: int):
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


def build_ce_ais_policy(config_dict, vla_config, device):
    """构建完整 CE-AIS 推理拓扑。"""
    from src.config.config_manager import ConfigManager
    from src.config.schema import (CEAISConfig, EncoderConfig, CEWMConfig,
                                    SteeringConfig, BilateralGatingConfig)
    from src.dual_stream.vla_adapter import build_vla_adapter
    from src.dual_stream.topology import DualStreamTopology
    from src.encoders.contrastive_encoder import ContrastiveEncoder
    from src.world_model.ce_wm import CausalEnergyWorldModel
    from src.steering.efe_steering import EFESteering
    from src.steering.bilateral_gating import BilateralGating

    enc_cfg = config_dict.get("encoder", {})
    cewm_cfg = config_dict.get("ce_wm", {})
    steer_cfg = config_dict.get("steering", {})
    gate_cfg = config_dict.get("bilateral_gating", {})

    encoder_config = EncoderConfig(**{k: v for k, v in enc_cfg.items()
                                      if k in EncoderConfig.__dataclass_fields__})
    cewm_config = CEWMConfig(**{k: v for k, v in cewm_cfg.items()
                                 if k in CEWMConfig.__dataclass_fields__})
    steering_config = SteeringConfig(**{k: v for k, v in steer_cfg.items()
                                        if k in SteeringConfig.__dataclass_fields__})
    gating_config = BilateralGatingConfig(**{k: v for k, v in gate_cfg.items()
                                             if k in BilateralGatingConfig.__dataclass_fields__})

    vla = build_vla_adapter(vla_config)
    encoder = ContrastiveEncoder(encoder_config).to(device).eval()
    ce_wm = CausalEnergyWorldModel(cewm_config).to(device).eval()
    steering = EFESteering(steering_config)
    gating = BilateralGating(gating_config)

    # 尝试加载检查点
    ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints")
    if os.path.isdir(ckpt_dir):
        from src.utils.checkpoint import CheckpointManager
        enc_mgr = CheckpointManager(checkpoint_dir=ckpt_dir, prefix="encoder")
        enc_ckpt = enc_mgr.find_latest()
        if enc_ckpt:
            enc_mgr.load(filepath=enc_ckpt, model=encoder, map_location=device)
            print(f"  Loaded encoder checkpoint: {enc_ckpt}")

        cewm_mgr = CheckpointManager(checkpoint_dir=ckpt_dir, prefix="cewm")
        cewm_ckpt = cewm_mgr.find_latest()
        if cewm_ckpt:
            cewm_mgr.load(filepath=cewm_ckpt, model=ce_wm, map_location=device)
            print(f"  Loaded CE-WM checkpoint: {cewm_ckpt}")

    topology = DualStreamTopology(
        vla_adapter=vla, encoder=encoder, ce_wm=ce_wm,
        steering=steering, gating=gating,
        mc_samples=gate_cfg.get("mc_samples", 5),
    )
    return topology


def evaluate_method(method_name, policy_fn, wrapper, chains, data_dir, args, rng):
    """对单个方法运行完整评估。"""
    val_dir = os.path.join(data_dir, "validation")
    val_files = sorted(f for f in os.listdir(val_dir) if f.endswith(".npz"))
    val_frame_ids = [int(f.split("_")[1].split(".")[0]) for f in val_files]

    chain_successes = {i: [] for i in range(1, args.chain_length + 1)}
    all_results = []
    latencies = []

    for ci, chain in enumerate(chains):
        frame_id = rng.choice(val_frame_ids)
        robot_obs, scene_obs = load_frame_state(data_dir, frame_id)
        wrapper.reset(task=chain[0], robot_obs=robot_obs, scene_obs=scene_obs)

        t_start = time.time()
        result = wrapper.run_chain_evaluation(
            policy_fn=policy_fn,
            task_chain=chain,
            max_steps_per_task=args.max_steps,
        )
        latencies.append((time.time() - t_start) * 1000 / max(sum(
            r["steps"] for r in result["task_results"]), 1))

        all_results.append(result)
        completed = result["completed_tasks"]
        for length in range(1, args.chain_length + 1):
            chain_successes[length].append(completed >= length)

        if (ci + 1) % 10 == 0:
            avg = sum(chain_successes[1]) / len(chain_successes[1]) * 100
            print(f"    [{method_name}] {ci+1}/{len(chains)} chains, "
                  f"L1 success: {avg:.1f}%")

    results_summary = {}
    for length in range(1, args.chain_length + 1):
        s = chain_successes[length]
        results_summary[f"chain_{length}"] = sum(s) / len(s) if s else 0.0

    results_summary["avg_completed_tasks"] = float(
        np.mean([r["completed_tasks"] for r in all_results]))
    results_summary["avg_latency_ms"] = float(np.mean(latencies)) if latencies else 0.0

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
    print("CE-AIS Paper Main Experiments")
    print(f"Methods: {args.methods}")
    print(f"Chains: {args.n_chains} x {args.chain_length}")
    print("=" * 70)

    # 初始化 CALVIN 环境
    print("\n[1] Initializing CALVIN environment...")
    from src.evaluation.calvin_integration import CALVINWrapper

    env_config = {
        "use_real_env": True,
        "scene": "calvin_scene_D",
        "cameras": "static_and_gripper",
        "use_egl": True,
        "seed": args.seed,
        "max_chain_length": args.chain_length,
    }
    wrapper = CALVINWrapper(config=env_config)
    print(f"  Available tasks: {len(wrapper.available_tasks)}")

    chains = sample_task_chains(wrapper.available_tasks, args.n_chains,
                                args.chain_length, rng)

    # VLA 共享配置
    vla_config = {
        "type": args.vla_type,
        "device": device,
        "action_dim": 7,
        "chunk_size": 1,
    }

    all_method_results = {}

    for method_name in args.methods:
        print(f"\n[Eval] Running: {method_name}")
        method_rng = np.random.RandomState(args.seed)

        if method_name == "ce_ais":
            topology = build_ce_ais_policy(config_dict, vla_config, device)

            def ce_ais_policy(obs_dict, instruction):
                obs_on_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                                 for k, v in obs_dict.items()}
                action, info = topology.safe_step(obs_on_device, str(instruction))
                return action.squeeze(1).cpu()

            results = evaluate_method(
                method_name, ce_ais_policy, wrapper, chains, args.data_dir, args, method_rng)

        elif method_name == "frozen_openvla":
            from src.evaluation.baseline_framework import FrozenOpenVLABaseline
            bl = FrozenOpenVLABaseline({**vla_config, "vla_type": args.vla_type})
            bl.setup()

            def frozen_policy(obs_dict, instruction):
                obs_on_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                                 for k, v in obs_dict.items()}
                return bl.predict(obs_on_device, str(instruction)).squeeze(1).cpu()

            results = evaluate_method(
                method_name, frozen_policy, wrapper, chains, args.data_dir, args, method_rng)
            bl.teardown()

        else:
            from src.evaluation.baseline_framework import BASELINE_REGISTRY
            bl_cls = BASELINE_REGISTRY[method_name]
            bl = bl_cls({**vla_config, "vla_type": args.vla_type})
            bl.setup()

            def baseline_policy(obs_dict, instruction, _bl=bl):
                obs_on_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                                 for k, v in obs_dict.items()}
                return _bl.predict(obs_on_device, str(instruction)).squeeze(1).cpu()

            results = evaluate_method(
                method_name, baseline_policy, wrapper, chains, args.data_dir, args, method_rng)
            bl.teardown()

        all_method_results[method_name] = results
        print(f"  [{method_name}] L1={results['chain_1']:.1%}, "
              f"L3={results.get('chain_3', 0):.1%}, "
              f"L5={results.get('chain_5', 0):.1%}, "
              f"latency={results['avg_latency_ms']:.1f}ms")

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    output = {
        "config": {
            "n_chains": args.n_chains,
            "chain_length": args.chain_length,
            "max_steps_per_task": args.max_steps,
            "seed": args.seed,
            "vla_type": args.vla_type,
            "methods": args.methods,
        },
        "results": all_method_results,
    }
    output_path = os.path.join(args.output_dir, "main_experiment.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved: {output_path}")

    # 打印对比表
    print("\n" + "=" * 70)
    print(f"{'Method':<20} {'L1':>8} {'L2':>8} {'L3':>8} {'L4':>8} {'L5':>8} {'Lat(ms)':>8}")
    print("-" * 70)
    for name, res in all_method_results.items():
        print(f"{name:<20} {res.get('chain_1',0):>7.1%} {res.get('chain_2',0):>7.1%} "
              f"{res.get('chain_3',0):>7.1%} {res.get('chain_4',0):>7.1%} "
              f"{res.get('chain_5',0):>7.1%} {res.get('avg_latency_ms',0):>7.1f}")
    print("=" * 70)

    wrapper.close()


if __name__ == "__main__":
    main()
