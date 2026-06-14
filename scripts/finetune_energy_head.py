#!/usr/bin/env python3
"""用 rollout 成功/失败数据微调 CE-WM 的 Energy Head。

只训练 Energy Head MLP，冻结 Mamba-3 backbone。
让 energy 反映"该 action 在当前 state 下是否导致任务成功"。

用法:
  python scripts/finetune_energy_head.py \
    --rollout-dir data/rollout_flower_seed42 \
    --cewm-ckpt checkpoints_calibrated_cewm/cewm_epoch0033.pt \
    --encoder-ckpt checkpoints/encoder_epoch0044.pt \
    --output-dir checkpoints_finetuned_energy_head \
    --epochs 30 --lr 5e-4 --batch-size 512
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class RolloutDataset(Dataset):
    """从 rollout 数据构造 (z_seq, a_seq, label) 用于 NCE 训练。

    正样本：成功 episode 中的连续 T 帧窗口
    负样本：失败 episode 中的连续 T 帧窗口
    """

    def __init__(self, rollout_dir: str, window_size: int = 16, neg_ratio: int = 5):
        self.window_size = window_size
        self.neg_ratio = neg_ratio

        latent = np.load(os.path.join(rollout_dir, "latent.npy"), mmap_mode="r")
        action = np.load(os.path.join(rollout_dir, "action.npy"), mmap_mode="r")
        success = np.load(os.path.join(rollout_dir, "success.npy"))
        episode_info = np.load(os.path.join(rollout_dir, "episode_info.npy"))

        self.latent = latent
        self.action = action

        # 按 success 分组：收集足够长的 episode 窗口
        self.pos_windows = []  # (start_idx, end_idx) pairs for positive
        self.neg_windows = []  # (start_idx, end_idx) pairs for negative

        for ep_start, ep_end, chain_id, task_id in episode_info:
            ep_len = ep_end - ep_start
            if ep_len < window_size:
                continue
            is_success = success[ep_start] == 1
            n_windows = ep_len - window_size + 1
            windows = [(ep_start + i, ep_start + i + window_size) for i in range(0, n_windows, max(1, n_windows // 4))]
            if is_success:
                self.pos_windows.extend(windows)
            else:
                self.neg_windows.extend(windows)

        print(f"  Positive windows: {len(self.pos_windows)}")
        print(f"  Negative windows: {len(self.neg_windows)}")

    def __len__(self):
        return len(self.pos_windows)

    def __getitem__(self, idx):
        # Positive sample
        ps, pe = self.pos_windows[idx]
        z_pos = torch.from_numpy(self.latent[ps:pe].copy())  # [T, 128]
        a_pos = torch.from_numpy(self.action[ps:pe].copy())  # [T, 7]

        # Negative samples: random from neg_windows
        neg_actions = []
        for _ in range(self.neg_ratio):
            ni = np.random.randint(len(self.neg_windows))
            ns, ne = self.neg_windows[ni]
            a_neg = torch.from_numpy(self.action[ns:ne].copy())  # [T, 7]
            neg_actions.append(a_neg)
        a_neg_stack = torch.stack(neg_actions, dim=0)  # [K, T, 7]

        return z_pos, a_pos, a_neg_stack


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-dir", type=str, default="data/rollout_flower_seed42")
    parser.add_argument("--cewm-ckpt", type=str, default="checkpoints_calibrated_cewm/cewm_epoch0033.pt")
    parser.add_argument("--output-dir", type=str, default="checkpoints_finetuned_energy_head")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--neg-ratio", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--target-margin", type=float, default=5.0)
    parser.add_argument("--min-margin", type=float, default=1.0)
    parser.add_argument("--energy-reg-weight", type=float, default=1e-4)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # === Load CE-WM ===
    print("[1/3] Loading CE-WM...")
    from src.config.config_manager import ConfigManager
    from src.config.schema import CEWMConfig
    from src.world_model.ce_wm import CausalEnergyWorldModel

    cm = ConfigManager(config_path="configs/base.yaml")
    config = cm.config
    cewm_cfg = config.get("ce_wm", {})

    model = CausalEnergyWorldModel(CEWMConfig(
        d_model=cewm_cfg.get("d_model", 640),
        d_state=cewm_cfg.get("d_state", 64),
        n_layers=cewm_cfg.get("n_layers", 32),
        expand_factor=cewm_cfg.get("expand_factor", 3),
        mimo_groups=cewm_cfg.get("mimo_groups", 4),
        action_dim=cewm_cfg.get("action_dim", 7),
        latent_dim=cewm_cfg.get("latent_dim", 128),
        dropout=cewm_cfg.get("dropout", 0.1),
        headdim=cewm_cfg.get("headdim", 64),
    )).to(device)

    ckpt = torch.load(args.cewm_ckpt, map_location=device)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state, strict=False)
    print(f"  Loaded from {args.cewm_ckpt}")

    # Freeze backbone, only train energy head
    for name, param in model.named_parameters():
        if "energy_head" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    # === Load Dataset ===
    print("[2/3] Loading rollout dataset...")
    dataset = RolloutDataset(args.rollout_dir, window_size=args.window_size, neg_ratio=args.neg_ratio)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                        pin_memory=True, drop_last=True)
    print(f"  Batches per epoch: {len(loader)}")

    # === Train ===
    print("[3/3] Training energy head...")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    model.train()
    best_loss = float("inf")

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_margin = 0.0
        n_batches = 0
        t0 = time.time()

        for z_pos, a_pos, a_neg_stack in loader:
            z_pos = z_pos.to(device)       # [B, T, 128]
            a_pos = a_pos.to(device)       # [B, T, 7]
            a_neg_stack = a_neg_stack.to(device)  # [B, K, T, 7]
            B, K, T, A = a_neg_stack.shape

            # Forward positive
            energy_pos = model(z_pos, a_pos)  # [B]

            # Forward negatives
            z_expanded = z_pos.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, T, 128)
            a_neg_flat = a_neg_stack.reshape(B * K, T, A)
            energy_neg = model(z_expanded, a_neg_flat).reshape(B, K)  # [B, K]

            # NCE Loss
            logits = torch.cat([-energy_pos.unsqueeze(1), -energy_neg], dim=1)  # [B, K+1]
            labels = torch.zeros(B, dtype=torch.long, device=device)
            loss_nce = F.cross_entropy(logits, labels)

            # Margin regularization
            margin = energy_neg.mean(dim=1) - energy_pos  # [B]
            margin_mean = margin.mean()
            loss_margin_upper = F.relu(margin_mean - args.target_margin).pow(2) * 0.01
            loss_margin_lower = F.relu(args.min_margin - margin_mean).pow(2) * 1.0

            # Energy scale regularization
            loss_reg = args.energy_reg_weight * (energy_pos.pow(2).mean() + energy_neg.pow(2).mean())

            loss = loss_nce + loss_margin_upper + loss_margin_lower + loss_reg

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_margin += margin_mean.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        avg_margin = epoch_margin / n_batches
        elapsed = time.time() - t0

        print(f"  Epoch {epoch+1:3d}/{args.epochs} | loss={avg_loss:.4f} | margin={avg_margin:.2f} | "
              f"lr={scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s")

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(args.output_dir, "energy_head_best.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "loss": avg_loss,
                "margin": avg_margin,
            }, save_path)

    # Save final
    save_path = os.path.join(args.output_dir, "energy_head_final.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": args.epochs,
        "loss": avg_loss,
        "margin": avg_margin,
    }, save_path)

    print(f"\nDone! Best loss: {best_loss:.4f}")
    print(f"  Best: {args.output_dir}/energy_head_best.pt")
    print(f"  Final: {args.output_dir}/energy_head_final.pt")


if __name__ == "__main__":
    main()
