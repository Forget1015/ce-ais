#!/usr/bin/env python3
"""CE-AIS 消融实验启动脚本。

运行 3 组消融实验（no_gating, mse_energy, mamba1_backbone），
使用 CALVIN 环境进行真实评估。

Usage:
    PYOPENGL_PLATFORM=egl uv run python scripts/ablation.py \
        --config configs/base.yaml --data-dir data/task_ABC_D
    PYOPENGL_PLATFORM=egl uv run python scripts/ablation.py \
        --config configs/base.yaml --variants no_gating mse_energy
    python scripts/ablation.py --config configs/base.yaml --list
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config.config_manager import ConfigManager, parse_overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CE-AIS Ablation Experiment Script")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--variants", type=str, nargs="*", default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--data-dir", type=str,
                        default=os.path.join(project_root, "data", "task_ABC_D"))
    parser.add_argument("--n-chains", type=int, default=50)
    parser.add_argument("--chain-length", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vla-type", type=str, default="proxy",
                        choices=["openvla", "proxy"])
    parser.add_argument("--override", type=str, nargs="*", default=[])
    return parser.parse_args()


def build_ce_ais_for_config(config_dict, vla_config, device):
    """根据配置构建 CE-AIS 推理拓扑。"""
    from src.config.schema import (EncoderConfig, CEWMConfig,
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
    ckpt_dir = os.path.join(project_root, "checkpoints")
    if os.path.isdir(ckpt_dir):
        from src.utils.checkpoint import CheckpointManager
        enc_mgr = CheckpointManager(checkpoint_dir=ckpt_dir, prefix="encoder")
        enc_ckpt = enc_mgr.find_latest()
        if enc_ckpt:
            enc_mgr.load(filepath=enc_ckpt, model=encoder, map_location=device)

        cewm_mgr = CheckpointManager(checkpoint_dir=ckpt_dir, prefix="cewm")
        cewm_ckpt = cewm_mgr.find_latest()
        if cewm_ckpt:
            cewm_mgr.load(filepath=cewm_ckpt, model=ce_wm, map_location=device)

    topology = DualStreamTopology(
        vla_adapter=vla, encoder=encoder, ce_wm=ce_wm,
        steering=steering, gating=gating,
        mc_samples=gate_cfg.get("mc_samples", 5),
    )
    return topology


def real_eval_fn(config_dict, args, device, vla_config):
    """真实评估函数：在 CALVIN 环境中评估指定配置。"""
    from src.evaluation.calvin_integration import CALVINWrapper

    env_config = {
        "use_real_env": True, "scene": "calvin_scene_D",
        "cameras": "static_and_gripper", "use_egl": True,
        "seed": args.seed, "max_chain_length": args.chain_length,
    }
    wrapper = CALVINWrapper(config=env_config)

    topology = build_ce_ais_for_config(config_dict, vla_config, device)

    def policy_fn(obs_dict, instruction):
        obs_on_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in obs_dict.items()}
        action, info = topology.safe_step(obs_on_device, str(instruction))
        return action.squeeze(1).cpu()

    rng = np.random.RandomState(args.seed)
    tasks = wrapper.available_tasks
    if not tasks:
        tasks = ["pick_up_object"]

    chains = []
    for _ in range(args.n_chains):
        chain = []
        for j in range(args.chain_length):
            candidates = [t for t in tasks if not chain or t != chain[-1]]
            chain.append(rng.choice(candidates))
        chains.append(chain)

    val_dir = os.path.join(args.data_dir, "validation")
    val_files = sorted(f for f in os.listdir(val_dir) if f.endswith(".npz"))
    val_frame_ids = [int(f.split("_")[1].split(".")[0]) for f in val_files]

    chain_successes = {i: [] for i in range(1, args.chain_length + 1)}
    latencies = []

    for ci, chain in enumerate(chains):
        frame_id = rng.choice(val_frame_ids)
        frame_path = os.path.join(args.data_dir, "validation", f"episode_{frame_id:07d}.npz")
        robot_obs, scene_obs = None, None
        if os.path.exists(frame_path):
            data = np.load(frame_path)
            robot_obs = data["robot_obs"].astype(np.float64)
            scene_obs = data["scene_obs"].astype(np.float64)

        wrapper.reset(task=chain[0], robot_obs=robot_obs, scene_obs=scene_obs)

        t_start = time.time()
        result = wrapper.run_chain_evaluation(
            policy_fn=policy_fn, task_chain=chain,
            max_steps_per_task=args.max_steps,
        )
        total_steps = sum(r["steps"] for r in result["task_results"])
        if total_steps > 0:
            latencies.append((time.time() - t_start) * 1000 / total_steps)

        completed = result["completed_tasks"]
        for length in range(1, args.chain_length + 1):
            chain_successes[length].append(completed >= length)

    wrapper.close()

    return {
        "chain_success_rate": {
            l: sum(s) / len(s) if s else 0.0
            for l, s in chain_successes.items()
        },
        "single_task_rate": {},
        "avg_steps": 0.0,
        "latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "trajectory_jerk": 0.0,
    }


def main():
    args = parse_args()
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    from src.evaluation.ablation import AblationFramework, ABLATION_VARIANTS

    if args.list:
        print("[CE-AIS] Available ablation variants:")
        for name, info in ABLATION_VARIANTS.items():
            print(f"  - {name}: {info['description']}")
            print(f"    Config: {info['config_file']}")
        return

    overrides = parse_overrides(args.override) if args.override else None
    cm = ConfigManager(config_path=args.config, overrides=overrides)
    cm.save_snapshot(args.log_dir)
    base_config = cm.config

    device = base_config.get("project", {}).get("device", "cuda:0")
    if not torch.cuda.is_available():
        device = "cpu"

    vla_config = {"type": args.vla_type, "device": device,
                  "action_dim": 7, "chunk_size": 1}

    print(f"[CE-AIS] Base config: {args.config}")
    print(f"[CE-AIS] Device: {device}")

    framework = AblationFramework(base_config=base_config, output_dir=args.log_dir)
    variant_names = args.variants or list(ABLATION_VARIANTS.keys())
    print(f"[CE-AIS] Ablation variants: {variant_names}")

    # 先跑 baseline（完整 CE-AIS）
    print("\n[CE-AIS] === Running full CE-AIS (baseline) ===")
    baseline_result = real_eval_fn(base_config, args, device, vla_config)
    print(f"  Baseline L1: {baseline_result['chain_success_rate'].get(1, 0):.1%}")
    print(f"  Baseline latency: {baseline_result['latency_ms']:.1f}ms")

    # 运行消融实验
    results = framework.run_all_ablations(
        eval_fn=lambda config: real_eval_fn(config, args, device, vla_config),
        variant_names=variant_names,
    )

    table_path = framework.generate_comparison_table()
    print(f"\n[CE-AIS] Ablation comparison saved to: {table_path}")

    # 打印对比表
    print("\n" + "=" * 70)
    print(f"{'Variant':<25} {'L1':>8} {'L3':>8} {'L5':>8} {'Lat(ms)':>8}")
    print("-" * 70)
    bl_csr = baseline_result["chain_success_rate"]
    print(f"{'CE-AIS (full)':<25} {bl_csr.get(1,0):>7.1%} {bl_csr.get(3,0):>7.1%} "
          f"{bl_csr.get(5,0):>7.1%} {baseline_result['latency_ms']:>7.1f}")
    for name, result in results.items():
        csr = result.chain_success_rate
        print(f"{name:<25} {csr.get(1,0):>7.1%} {csr.get(3,0):>7.1%} "
              f"{csr.get(5,0):>7.1%} {result.latency_ms:>7.1f}")
    print("=" * 70)

    # 保存完整结果
    full_output = {
        "baseline": {
            "chain_success_rate": baseline_result["chain_success_rate"],
            "latency_ms": baseline_result["latency_ms"],
        },
        "ablations": {
            name: {
                "description": r.description,
                "chain_success_rate": r.chain_success_rate,
                "latency_ms": r.latency_ms,
            }
            for name, r in results.items()
        },
    }
    full_path = os.path.join(args.log_dir, "ablations.json")
    with open(full_path, "w") as f:
        json.dump(full_output, f, indent=2, default=str)
    print(f"[CE-AIS] Full ablation results saved: {full_path}")


if __name__ == "__main__":
    main()
