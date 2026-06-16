#!/usr/bin/env python3
"""收集 Frozen FLOWER rollout 数据用于重训 CE-WM Energy Head。

跑 1000 chains × 5 tasks，保存每步的 (z_t, a_t, task_success)。
输出格式和 CE-WM 训练 mmap 兼容：
  - latent.npy: [N, 128] float32 (encoder 输出的 z_t)
  - action.npy: [N, 7] float32 (FLOWER 实际执行的 action)
  - success.npy: [N] uint8 (该 step 所在 subtask 是否最终成功)
  - episode_info.npy: [num_subtasks, 4] int64 (start_idx, end_idx, chain_id, task_id_in_chain)
  - metadata.json: 配置信息

用法:
  python scripts/collect_rollout_data.py \
    --device cuda:6 \
    --n-chains 1000 \
    --seed 42 \
    --output-dir data/rollout_flower_seed42
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description="Collect rollout data for CE-WM Energy Head finetuning")
    parser.add_argument("--data-dir", type=str, default="data/task_ABC_D")
    parser.add_argument("--device", type=str, default="cuda:6")
    parser.add_argument("--n-chains", type=int, default=1000)
    parser.add_argument("--chain-length", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=360)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="data/rollout_flower_seed42")
    parser.add_argument("--vla-type", type=str, default="flower", choices=["flower", "robovlms"])
    parser.add_argument("--flower-checkpoint-dir", type=str, default="data/flower_calvin_abc")
    parser.add_argument("--flower-code-path", type=str, default="external/flower_vla_calvin")
    parser.add_argument("--robovlms-checkpoint-dir", type=str, default="data/robovlms")
    parser.add_argument("--robovlms-code-path", type=str, default="external/RoboVLMs")
    parser.add_argument("--encoder-ckpt", type=str, default="checkpoints/encoder_epoch0044.pt")
    return parser.parse_args()


def main():
    args = parse_args()

    from pytorch_lightning import seed_everything
    seed_everything(0, workers=True)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # === 1. Load Encoder (frozen) ===
    print("[1/4] Loading Encoder...", flush=True)
    from src.config.config_manager import ConfigManager
    cm = ConfigManager(config_path="configs/base.yaml")
    config = cm.config

    from src.encoders.contrastive_encoder import ContrastiveEncoder
    from src.config.schema import EncoderConfig
    enc_cfg = config.get("encoder", {})
    encoder_config = EncoderConfig(
        backbone_type=enc_cfg.get("backbone_type", "resnet18"),
        pose_dim=enc_cfg.get("pose_dim", 7),
        visual_dim=enc_cfg.get("visual_dim", 512),
        latent_dim=enc_cfg.get("latent_dim", 128),
        image_size=enc_cfg.get("image_size", [200, 200]),
    )
    encoder = ContrastiveEncoder(encoder_config).to(device)
    encoder.eval()

    if args.encoder_ckpt and os.path.exists(args.encoder_ckpt):
        ckpt = torch.load(args.encoder_ckpt, map_location=device)
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        encoder.load_state_dict(state, strict=False)
        print(f"  Encoder loaded from {args.encoder_ckpt}")
    else:
        print(f"  WARNING: No encoder checkpoint, using random init!")

    for p in encoder.parameters():
        p.requires_grad_(False)

    # === 2. Load VLA (frozen) ===
    print(f"[2/4] Loading {args.vla_type} VLA...", flush=True)
    from src.dual_stream.vla_adapter import build_vla_adapter
    if args.vla_type == "flower":
        vla_config = {
            "type": "flower",
            "device": str(device),
            "action_dim": 7,
            "chunk_size": 1,
            "flower_checkpoint_dir": args.flower_checkpoint_dir,
            "flower_code_path": args.flower_code_path,
        }
    elif args.vla_type == "robovlms":
        vla_config = {
            "type": "robovlms",
            "device": str(device),
            "action_dim": 7,
            "chunk_size": 1,
            "robovlms_checkpoint_dir": args.robovlms_checkpoint_dir,
            "robovlms_code_path": args.robovlms_code_path,
        }
    adapter = build_vla_adapter(vla_config)

    # === 3. Create CALVIN Env ===
    print("[3/4] Creating CALVIN environment...", flush=True)
    from src.evaluation.calvin_integration import CALVINWrapper
    env_config = {
        "use_real_env": True,
        "data_dir": args.data_dir,
        "scene": "calvin_scene_D",
        "cameras": "static_and_gripper",
        "use_egl": True,
        "max_chain_length": args.chain_length,
    }
    wrapper = CALVINWrapper(config=env_config)

    # === 4. Load eval sequences ===
    print("[4/4] Loading evaluation sequences...", flush=True)
    sys.path.insert(0, args.flower_code_path)
    from flower.evaluation.multistep_sequences import get_sequences
    from flower.evaluation.utils import get_env_state_for_initial_condition
    sequences = get_sequences(args.n_chains)

    val_ann = np.load(
        os.path.join(args.data_dir, "validation", "lang_annotations", "auto_lang_ann.npy"),
        allow_pickle=True
    ).item()
    annotations = {}
    for lang, task in zip(val_ann["language"]["ann"], val_ann["language"]["task"]):
        annotations.setdefault(task, []).append(lang)

    # === Collect Rollout Data ===
    print(f"\nStarting rollout collection: {args.n_chains} chains × {args.chain_length} tasks")
    print(f"Output: {args.output_dir}\n", flush=True)

    all_latents = []
    all_actions = []
    all_success = []
    episode_info = []  # (start_idx, end_idx, chain_id, task_idx)
    global_step = 0

    t_start = time.time()

    for chain_idx in range(args.n_chains):
        initial_state, eval_sequence = sequences[chain_idx]
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)

        wrapper.reset(
            task=eval_sequence[0],
            robot_obs=robot_obs.astype(np.float64),
            scene_obs=scene_obs.astype(np.float64),
        )
        adapter.reset()

        for task_idx in range(min(args.chain_length, len(eval_sequence))):
            subtask = eval_sequence[task_idx]
            instruction = annotations[subtask][0]

            if task_idx > 0:
                adapter.reset()
            wrapper._current_task = subtask
            start_info = wrapper._env.get_info()
            wrapper._start_info = start_info

            ep_start = global_step
            task_success = False

            for step in range(args.max_steps):
                raw_obs = wrapper._env.get_obs()

                # Encode observation to latent z_t
                rgb_static = raw_obs["rgb_obs"]["rgb_static"]  # [H, W, 3] uint8
                depth_static = raw_obs["depth_obs"]["depth_static"]  # [H, W] float
                pose = raw_obs["robot_obs"][:7]  # [7] float

                with torch.no_grad():
                    rgb_tensor = torch.from_numpy(rgb_static).permute(2, 0, 1).unsqueeze(0).float().to(device)
                    rgb_tensor = torch.nn.functional.interpolate(
                        rgb_tensor, size=(200, 200), mode="bilinear", align_corners=False
                    )
                    if depth_static.ndim == 2:
                        depth_tensor = torch.from_numpy(depth_static).unsqueeze(0).unsqueeze(0).float().to(device)
                    else:
                        depth_tensor = torch.from_numpy(depth_static).unsqueeze(0).float().to(device)
                    depth_tensor = torch.nn.functional.interpolate(
                        depth_tensor, size=(200, 200), mode="bilinear", align_corners=False
                    )
                    pose_tensor = torch.from_numpy(pose.astype(np.float32)).unsqueeze(0).to(device)
                    z_t = encoder(rgb_tensor, depth_tensor, pose_tensor)  # [1, 128]

                # Get action from FLOWER
                observation = {"raw_calvin_obs": raw_obs}
                action = adapter.predict(observation, instruction)
                action_np = action.squeeze().cpu().numpy()
                action_exec = action_np.copy()
                action_exec[-1] = 1.0 if action_exec[-1] >= 0 else -1.0

                # Store data
                all_latents.append(z_t.cpu().numpy().squeeze())  # [128]
                all_actions.append(action_np)  # [7] (raw action before gripper binarization)
                global_step += 1

                # Step env
                wrapper._env.step(action_exec)

                # Check task success
                end_info = wrapper._env.get_info()
                achieved = wrapper._tasks.get_task_info_for_set(start_info, end_info, {subtask})
                if subtask in achieved:
                    task_success = True
                    break

            ep_end = global_step
            episode_info.append((ep_start, ep_end, chain_idx, task_idx))

            # Mark success/failure for all steps in this episode
            success_val = 1 if task_success else 0
            all_success.extend([success_val] * (ep_end - ep_start))

            if not task_success:
                break  # Chain broken

        # Progress
        if (chain_idx + 1) % 10 == 0:
            elapsed = time.time() - t_start
            n_success = sum(1 for info in episode_info if info[3] == 0 and
                           all_success[info[0]] == 1)
            total_l1 = sum(1 for info in episode_info if info[3] == 0)
            l1_rate = n_success / total_l1 if total_l1 > 0 else 0
            print(f"  Chain {chain_idx+1}/{args.n_chains} | "
                  f"Steps: {global_step} | L1: {l1_rate*100:.1f}% | "
                  f"Time: {elapsed:.0f}s", flush=True)

    # === Save Data ===
    print(f"\nSaving {global_step} steps to {args.output_dir}...", flush=True)

    latent_arr = np.array(all_latents, dtype=np.float32)  # [N, 128]
    action_arr = np.array(all_actions, dtype=np.float32)  # [N, 7]
    success_arr = np.array(all_success, dtype=np.uint8)   # [N]
    episode_arr = np.array(episode_info, dtype=np.int64)  # [num_episodes, 4]

    np.save(os.path.join(args.output_dir, "latent.npy"), latent_arr)
    np.save(os.path.join(args.output_dir, "action.npy"), action_arr)
    np.save(os.path.join(args.output_dir, "success.npy"), success_arr)
    np.save(os.path.join(args.output_dir, "episode_info.npy"), episode_arr)

    # Metadata
    n_pos = int(success_arr.sum())
    n_neg = len(success_arr) - n_pos
    metadata = {
        "total_steps": global_step,
        "total_episodes": len(episode_info),
        "total_chains": args.n_chains,
        "chain_length": args.chain_length,
        "max_steps_per_task": args.max_steps,
        "seed": args.seed,
        "positive_steps": n_pos,
        "negative_steps": n_neg,
        "latent_dim": 128,
        "action_dim": 7,
        "encoder_ckpt": args.encoder_ckpt,
        "flower_checkpoint_dir": args.flower_checkpoint_dir,
        "calvin_env_version": "797142c",
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone! Summary:")
    print(f"  Total steps: {global_step}")
    print(f"  Positive steps (success): {n_pos} ({n_pos/global_step*100:.1f}%)")
    print(f"  Negative steps (failure): {n_neg} ({n_neg/global_step*100:.1f}%)")
    print(f"  Episodes: {len(episode_info)}")
    print(f"  Files: latent.npy, action.npy, success.npy, episode_info.npy, metadata.json")


if __name__ == "__main__":
    main()
