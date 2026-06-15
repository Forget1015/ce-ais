#!/usr/bin/env python3
"""方案二：用 State-Conditioned Contrastive Pairs 微调 Energy Head。

核心区别：每个训练样本是 (z_anchor, a_pos, a_neg)，其中 a_pos 和 a_neg
来自相同 state 附近分别导致成功/失败的 action。Energy head 学到的是
"在给定 state 下 action 的好坏"。

用法:
  python scripts/finetune_energy_head_v2.py \
    --pairs-file data/rollout_flower_seed42/contrastive_pairs.npz \
    --cewm-ckpt checkpoints_calibrated_cewm/cewm_epoch0033.pt \
    --output-dir checkpoints_finetuned_energy_head_v2 \
    --epochs 20 --lr 3e-4 --batch-size 1024
"""

import argparse
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


class ContrastivePairsDataset(Dataset):
    """单步 contrastive pairs: (z, a_pos, a_neg)"""

    def __init__(self, pairs_file: str):
        data = np.load(pairs_file)
        self.anchor_z = data["anchor_z"]   # [P, 128]
        self.pos_action = data["pos_action"]  # [P, 7]
        self.neg_action = data["neg_action"]  # [P, 7]
        print(f"  Loaded {len(self.anchor_z)} contrastive pairs")

    def __len__(self):
        return len(self.anchor_z)

    def __getitem__(self, idx):
        z = torch.from_numpy(self.anchor_z[idx])
        a_pos = torch.from_numpy(self.pos_action[idx])
        a_neg = torch.from_numpy(self.neg_action[idx])
        return z, a_pos, a_neg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-file", type=str, default="data/rollout_flower_seed42/contrastive_pairs.npz")
    parser.add_argument("--cewm-ckpt", type=str, default="checkpoints_calibrated_cewm/cewm_epoch0033.pt")
    parser.add_argument("--output-dir", type=str, default="checkpoints_finetuned_energy_head_v2")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--margin", type=float, default=1.0, help="Margin for triplet-style loss")
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

    ckpt = torch.load(args.cewm_ckpt, map_location=device, weights_only=False)
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
    print("[2/3] Loading contrastive pairs...")
    dataset = ContrastivePairsDataset(args.pairs_file)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)
    print(f"  Batches per epoch: {len(loader)}")

    # === Train ===
    print("[3/3] Training energy head (state-conditioned contrastive)...")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    model.train()
    best_loss = float("inf")
    best_acc = 0.0

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_margin = 0.0
        n_batches = 0
        t0 = time.time()

        for z, a_pos, a_neg in loader:
            z = z.to(device)          # [B, 128]
            a_pos = a_pos.to(device)  # [B, 7]
            a_neg = a_neg.to(device)  # [B, 7]

            # Single-step energy: need to unsqueeze to [B, 1, dim] for model
            z_seq = z.unsqueeze(1)        # [B, 1, 128]
            a_pos_seq = a_pos.unsqueeze(1)  # [B, 1, 7]
            a_neg_seq = a_neg.unsqueeze(1)  # [B, 1, 7]

            # Forward
            e_pos = model(z_seq, a_pos_seq)  # [B]
            e_neg = model(z_seq, a_neg_seq)  # [B]

            # Margin ranking loss: E(pos) should be lower than E(neg) by margin
            loss_rank = F.margin_ranking_loss(
                e_neg, e_pos,
                target=torch.ones(len(z), device=device),
                margin=args.margin
            )

            # Binary cross-entropy style: treat as classification
            logits = torch.stack([-e_pos, -e_neg], dim=1)  # [B, 2]
            labels = torch.zeros(len(z), dtype=torch.long, device=device)  # pos is class 0
            loss_ce = F.cross_entropy(logits, labels)

            # Energy regularization
            loss_reg = args.energy_reg_weight * (e_pos.pow(2).mean() + e_neg.pow(2).mean())

            loss = loss_rank + loss_ce + loss_reg

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Metrics
            acc = (e_pos < e_neg).float().mean().item()
            margin_val = (e_neg - e_pos).mean().item()

            epoch_loss += loss.item()
            epoch_acc += acc
            epoch_margin += margin_val
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        avg_acc = epoch_acc / n_batches
        avg_margin = epoch_margin / n_batches
        elapsed = time.time() - t0

        print(f"  Epoch {epoch+1:3d}/{args.epochs} | loss={avg_loss:.4f} | "
              f"acc={avg_acc:.3f} | margin={avg_margin:.3f} | "
              f"lr={scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s")

        # Save best (by accuracy)
        if avg_acc > best_acc:
            best_acc = avg_acc
            best_loss = avg_loss
            save_path = os.path.join(args.output_dir, "energy_head_v2_best.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "loss": avg_loss,
                "acc": avg_acc,
                "margin": avg_margin,
            }, save_path)

    # Save final
    save_path = os.path.join(args.output_dir, "energy_head_v2_final.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": args.epochs,
        "loss": avg_loss,
        "acc": avg_acc,
        "margin": avg_margin,
    }, save_path)

    print(f"\nDone! Best acc: {best_acc:.3f}, loss: {best_loss:.4f}")
    print(f"  Best: {args.output_dir}/energy_head_v2_best.pt")
    print(f"  Final: {args.output_dir}/energy_head_v2_final.pt")


if __name__ == "__main__":
    main()
