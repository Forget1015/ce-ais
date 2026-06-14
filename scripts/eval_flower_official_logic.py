"""直接复用官方 FLOWER 评估逻辑的桥接脚本。

完全用官方 HulcWrapper + model.step + evaluate_sequence 评估。
避免 CE-AIS 自己封装的 CALVINWrapper 引入的潜在差异。
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

FLOWER_CODE = PROJECT_ROOT / "external" / "flower_vla_calvin"
sys.path.insert(0, str(FLOWER_CODE))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dataset-path", type=str, default="data/task_ABC_D")
    parser.add_argument("--checkpoint-dir", type=str, default="data/flower_calvin_abc")
    parser.add_argument("--num-sequences", type=int, default=1000)
    parser.add_argument("--max-subtasks", type=int, default=5)
    parser.add_argument("--seq-start", type=int, default=0)
    parser.add_argument("--seq-end", type=int, default=None)
    parser.add_argument("--output", type=str, default="results/flower_official_logic/results.json")
    args = parser.parse_args()

    if args.seq_end is None:
        args.seq_end = args.num_sequences

    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("PL_TORCH_DISTRIBUTED_BACKEND", "gloo")

    from pytorch_lightning import seed_everything
    seed_everything(0, workers=True)

    dataset_path = str(PROJECT_ROOT / args.dataset_path)
    checkpoint_dir = Path(PROJECT_ROOT / args.checkpoint_dir)

    device = torch.device(f"cuda:{args.device}")

    # --- 用官方逻辑加载模型和环境 ---
    from flower.evaluation.utils import get_default_mode_and_env, get_env_state_for_initial_condition
    from flower.evaluation.multistep_sequences import get_sequences
    from flower.evaluation.flower_evaluate import count_success

    train_folder = str(checkpoint_dir)
    checkpoint = str(checkpoint_dir / "model.safetensors")

    eval_cfg_overwrite = {
        "use_extracted_rel_actions": False,
        "datamodule": {
            "datasets": {
                "lang_dataset": {
                    "lang_folder": "lang_annotations"
                }
            }
        },
        "model": {
            "num_sampling_steps": 4
        }
    }

    model, env, _, lang_embeddings = get_default_mode_and_env(
        train_folder,
        dataset_path,
        checkpoint,
        env=None,
        lang_embeddings=None,
        eval_cfg_overwrite=eval_cfg_overwrite,
        device_id=args.device,
    )
    model = model.to(device)
    model.num_sampling_steps = 4
    model.multistep = 10
    print(f"sampling_steps / multistep: {model.num_sampling_steps} {model.multistep}")
    model.eval()

    # --- 加载 task oracle 和 annotations ---
    import hydra
    from omegaconf import OmegaConf

    task_cfg_path = FLOWER_CODE / "conf" / "callbacks" / "rollout_lh" / "tasks" / "new_playtable_tasks.yaml"
    task_cfg = OmegaConf.load(task_cfg_path)
    task_oracle = hydra.utils.instantiate(task_cfg)

    ann_path = FLOWER_CODE / "conf" / "annotations" / "new_playtable_validation.yaml"
    val_annotations = OmegaConf.load(ann_path)

    # --- 评估循环（完全复制官方 rollout 逻辑）---
    ep_len = 360

    all_sequences = get_sequences(args.num_sequences)
    shard = all_sequences[args.seq_start:args.seq_end]
    print(f"Shard [{args.seq_start}:{args.seq_end}] -> {len(shard)} sequences | max_subtasks={args.max_subtasks}")

    results = []
    t0 = time.time()

    for k, (initial_state, eval_sequence) in enumerate(shard):
        eval_sequence = eval_sequence[:args.max_subtasks]

        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

        success_counter = 0
        for subtask in eval_sequence:
            obs = env.get_obs()
            goal = {"lang_text": val_annotations[subtask][0]}
            model.reset()
            start_info = env.get_info()

            success = False
            for step in range(ep_len):
                action = model.step(obs, goal)
                obs, _, _, current_info = env.step(action)
                current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
                if len(current_task_info) > 0:
                    success = True
                    break

            if success:
                success_counter += 1
            else:
                break

        results.append(success_counter)

        done = k + 1
        sr = count_success(results)
        avg = sum(sr) / len(sr) * 5
        elapsed = time.time() - t0
        eta = elapsed / done * (len(shard) - done)
        print(f"[{args.seq_start}:{args.seq_end}] {done}/{len(shard)} | "
              f"avg_len={avg:.3f} | L1={sr[0]*100:.1f}% | "
              f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m", flush=True)

    # --- 保存结果 ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sr = count_success(results)
    summary = {
        "seq_start": args.seq_start,
        "seq_end": args.seq_end,
        "max_subtasks": args.max_subtasks,
        "ep_len": ep_len,
        "num_sampling_steps": 4,
        "multistep": 10,
        "results": results,
        "L1": sr[0] if len(sr) > 0 else 0,
        "avg_len": float(np.mean(results)),
    }
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nFinal: L1={sr[0]*100:.1f}% | avg_len={np.mean(results):.4f}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
