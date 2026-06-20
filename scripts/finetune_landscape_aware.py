#!/usr/bin/env python3
"""带 Landscape 约束的微调：区分成功/失败 + 保持 energy landscape 形状。

核心改进：在 ranking loss 基础上加 landscape regularization，
防止微调为了区分 pos/neg 而扭曲 landscape（把随机方向 energy 拉低）。

L = L_ranking(pos vs neg) + L_ce(pos vs neg) + λ_land × L_landscape(pos vs random)

用法:
  CUDA_VISIBLE_DEVICES=3 python scripts/finetune_landscape_aware.py \
    --pairs-file data/rollout_flower_seed42/contrastive_pairs.npz \
    --cewm-ckpt checkpoints_calibrated_cewm/checkpoints_landscape_fix/cewm_epoch0025.pt \
    --output-dir checkpoints_calibrated_cewm/checkpoints_landscape_finetune \
    --epochs 20 --lr 1e-4 --batch-size 1024 --unfreeze-layers 8
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
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class ContrastivePairsDataset(Dataset):
    def __init__(self, pairs_file: str):
        data = np.load(pairs_file)
        self.anchor_z = torch.from_numpy(data["anchor_z"])
        self.pos_action = torch.from_numpy(data["pos_action"])
        self.neg_action = torch.from_numpy(data["neg_action"])

    def __len__(self):
        return len(self.anchor_z)

    def __getitem__(self, idx):
        z = self.anchor_z[idx]
        a_pos = self.pos_action[idx]
        a_neg = self.neg_action[idx]
        # 在线生成 landscape 约束用的随机负样本
        a_rand_medium = a_pos + torch.randn_like(a_pos) * 0.1
        a_rand_large = a_pos + torch.randn_like(a_pos) * 0.3
        a_uniform = torch.rand_like(a_pos) * 2 - 1
        return z, a_pos, a_neg, a_rand_medium, a_rand_large, a_uniform


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs-file", default="data/rollout_flower_seed42/contrastive_pairs.npz")
    p.add_argument("--cewm-ckpt", default="checkpoints_calibrated_cewm/checkpoints_landscape_fix/cewm_epoch0025.pt")
    p.add_argument("--output-dir", default="checkpoints_calibrated_cewm/checkpoints_landscape_finetune")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--margin", type=float, default=1.0)
    p.add_argument("--landscape-weight", type=float, default=1.0,
                   help="landscape 约束 loss 的权重")
    p.add_argument("--unfreeze-layers", type=int, default=8,
                   help="解冻 Mamba 后 N 层 (0=只训 energy head)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f"Config: epochs={args.epochs}, lr={args.lr}, bs={args.batch_size}, "
          f"landscape_weight={args.landscape_weight}, unfreeze={args.unfreeze_layers}")

    # === Load CE-WM ===
    print("[1/3] Loading CE-WM...")
    from src.config.config_manager import ConfigManager
    from src.config.schema import CEWMConfig
    from src.world_model.ce_wm import CausalEnergyWorldModel

    cm = ConfigManager(config_path="configs/base.yaml")
    cfg = cm.config.get("ce_wm", {})

    model = CausalEnergyWorldModel(CEWMConfig(
        d_model=cfg.get("d_model", 640), d_state=cfg.get("d_state", 64),
        n_layers=cfg.get("n_layers", 32), expand_factor=cfg.get("expand_factor", 3),
        mimo_groups=cfg.get("mimo_groups", 4), action_dim=cfg.get("action_dim", 7),
        latent_dim=cfg.get("latent_dim", 128), dropout=cfg.get("dropout", 0.1),
        headdim=cfg.get("headdim", 64),
    )).to(device)

    ckpt = torch.load(args.cewm_ckpt, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    print(f"  Loaded from {args.cewm_ckpt}")

    # Freeze/unfreeze strategy
    for param in model.parameters():
        param.requires_grad_(False)
    # Always train energy head
    for name, param in model.named_parameters():
        if "energy_head" in name:
            param.requires_grad_(True)
    # Unfreeze last N Mamba layers
    if args.unfreeze_layers > 0:
        n_layers = cfg.get("n_layers", 32)
        start_layer = n_layers - args.unfreeze_layers
        for name, param in model.named_parameters():
            for i in range(start_layer, n_layers):
                if f"layers.{i}." in name:
                    param.requires_grad_(True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

    # === Dataset ===
    print("[2/3] Loading dataset...")
    dataset = ContrastivePairsDataset(args.pairs_file)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True,
                        persistent_workers=True)
    print(f"  Samples: {len(dataset):,}, Batches/epoch: {len(loader)}")

    # === Train ===
    print("[3/3] Training...")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    model.train()
    best_score = 0.0

    for epoch in range(args.epochs):
        ep_loss, ep_rank, ep_land = 0., 0., 0.
        ep_acc, ep_land_acc = 0., 0.
        n = 0
        t0 = time.time()

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for z, a_pos, a_neg, a_rm, a_rl, a_uni in pbar:
            z = z.to(device, non_blocking=True)
            a_pos = a_pos.to(device, non_blocking=True)
            a_neg = a_neg.to(device, non_blocking=True)
            a_rm = a_rm.to(device, non_blocking=True)
            a_rl = a_rl.to(device, non_blocking=True)
            a_uni = a_uni.to(device, non_blocking=True)

            z_seq = z.unsqueeze(1)

            # Forward: pos, neg, and landscape negatives
            e_pos = model(z_seq, a_pos.unsqueeze(1))
            e_neg = model(z_seq, a_neg.unsqueeze(1))
            e_rm = model(z_seq, a_rm.unsqueeze(1))
            e_rl = model(z_seq, a_rl.unsqueeze(1))
            e_uni = model(z_seq, a_uni.unsqueeze(1))

            ones = torch.ones(len(z), device=device)

            # === Loss 1: Ranking (pos vs neg) ===
            loss_rank = F.margin_ranking_loss(
                e_neg, e_pos, target=ones, margin=args.margin)
            loss_ce = F.cross_entropy(
                torch.stack([-e_pos, -e_neg], dim=1),
                torch.zeros(len(z), dtype=torch.long, device=device))

            # === Loss 2: Landscape 约束 (pos < random) ===
            loss_land_rm = F.margin_ranking_loss(
                e_rm, e_pos, target=ones, margin=0.3)
            loss_land_rl = F.margin_ranking_loss(
                e_rl, e_pos, target=ones, margin=0.5)
            loss_land_uni = F.margin_ranking_loss(
                e_uni, e_pos, target=ones, margin=1.0)
            loss_landscape = (loss_land_rm + loss_land_rl + loss_land_uni) / 3

            # Total loss
            loss = loss_rank + loss_ce + args.landscape_weight * loss_landscape

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                acc = (e_pos < e_neg).float().mean().item()
                land_acc = (e_pos < e_uni).float().mean().item()

            ep_loss += loss.item()
            ep_rank += (loss_rank + loss_ce).item()
            ep_land += loss_landscape.item()
            ep_acc += acc
            ep_land_acc += land_acc
            n += 1

            if n % 50 == 0:
                pbar.set_postfix(acc=f"{acc:.3f}", land=f"{land_acc:.3f}")

        scheduler.step()
        al = ep_loss/n
        ar = ep_rank/n
        aland = ep_land/n
        am = ep_acc/n
        alm = ep_land_acc/n
        elapsed = time.time() - t0

        print(f"  Epoch {epoch+1:3d}/{args.epochs} | loss={al:.4f} (rank={ar:.4f} land={aland:.4f}) | "
              f"pos<neg={am:.3f} | pos<uniform={alm:.3f} | "
              f"lr={scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s")

        # Save best by combined score (high acc + high landscape)
        score = am * 0.7 + alm * 0.3
        if score > best_score:
            best_score = score
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch+1,
                        "acc": am, "landscape_acc": alm, "loss": al},
                       os.path.join(args.output_dir, "best.pt"))

        # Save every epoch
        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch+1,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "acc": am, "landscape_acc": alm, "loss": al},
                   os.path.join(args.output_dir, f"epoch{epoch+1:04d}.pt"))

    print(f"\nDone! Best score: {best_score:.3f}")


if __name__ == "__main__":
    main()
