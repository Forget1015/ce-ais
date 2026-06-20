#!/usr/bin/env python3
"""CE-WM 多尺度预训练：从头训练，解决 4280x action scale mismatch。

仿照 pretrain_pipeline.py 的训练模式:
- tqdm 进度条
- AMP 混合精度
- 能量统计 (pos/neg mean, margin, micro_acc)
- checkpoint 定期保存
- GPU 利用率优化 (大 batch + prefetch)

用法:
  CUDA_VISIBLE_DEVICES=5 python scripts/pretrain_cewm_multiscale.py \
    --data-file data/rollout_flower_seed42/contrastive_pairs.npz \
    --output-dir checkpoints_calibrated_cewm/checkpoints_multiscale \
    --epochs 40 --lr 1e-4 --batch-size 1024 --device cuda:0
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
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class MultiScaleDataset(Dataset):
    """多尺度负样本数据集，在线生成。"""

    def __init__(self, pairs_file: str):
        data = np.load(pairs_file)
        self.anchor_z = torch.from_numpy(data["anchor_z"])
        self.pos_action = torch.from_numpy(data["pos_action"])
        self.neg_action_orig = torch.from_numpy(data["neg_action"])

    def __len__(self):
        return len(self.anchor_z)

    def __getitem__(self, idx):
        z = self.anchor_z[idx]
        a_pos = self.pos_action[idx]
        a_orig = self.neg_action_orig[idx]
        a_micro = a_pos + torch.randn_like(a_pos) * 0.003
        a_med = a_pos + torch.randn_like(a_pos) * 0.01
        a_large = a_pos + torch.randn_like(a_pos) * 0.05
        a_xl = a_pos + torch.randn_like(a_pos) * 0.1
        return z, a_pos, a_micro, a_med, a_large, a_xl, a_orig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-file", default="data/rollout_flower_seed42/contrastive_pairs.npz")
    p.add_argument("--output-dir", default="checkpoints_calibrated_cewm/checkpoints_multiscale")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--nce-weight", type=float, default=0.5)
    p.add_argument("--save-interval", type=int, default=5)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    log_path = os.path.join(args.output_dir, "train.log")
    log_f = open(log_path, "w")

    def log(msg):
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

    log(f"Config: epochs={args.epochs}, bs={args.batch_size}, lr={args.lr}, "
        f"nce_weight={args.nce_weight}, amp={args.amp}")

    # === Build Model ===
    log("[1/3] Building CE-WM from scratch...")
    from src.config.config_manager import ConfigManager
    from src.config.schema import CEWMConfig
    from src.world_model.ce_wm import CausalEnergyWorldModel

    cm = ConfigManager(config_path="configs/base.yaml")
    cfg = cm.config.get("ce_wm", {})

    model = CausalEnergyWorldModel(CEWMConfig(
        d_model=cfg.get("d_model", 640),
        d_state=cfg.get("d_state", 64),
        n_layers=cfg.get("n_layers", 32),
        expand_factor=cfg.get("expand_factor", 3),
        mimo_groups=cfg.get("mimo_groups", 4),
        action_dim=cfg.get("action_dim", 7),
        latent_dim=cfg.get("latent_dim", 128),
        dropout=cfg.get("dropout", 0.1),
        headdim=cfg.get("headdim", 64),
    )).to(device)

    total = sum(p.numel() for p in model.parameters())
    log(f"  Params: {total:,} (all trainable)")

    # === Dataset ===
    log("[2/3] Loading dataset...")
    dataset = MultiScaleDataset(args.data_file)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        drop_last=True, persistent_workers=True,
        prefetch_factor=4,
    )
    log(f"  Samples: {len(dataset):,}, Batches/epoch: {len(loader)}")

    # === Optimizer ===
    log("[3/3] Starting training...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs)
    scaler = GradScaler(enabled=args.amp)

    start_epoch = 0
    best_micro_acc = 0.0

    # Resume from checkpoint
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        best_micro_acc = ckpt.get("micro_acc", 0.0)
        log(f"  Resumed from {args.resume}, epoch {start_epoch}")

    model.train()

    for epoch in range(start_epoch, args.epochs):
        # Warmup LR
        if epoch < args.warmup_epochs:
            lr_now = args.lr * (epoch + 1) / args.warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now
        else:
            lr_now = scheduler.get_last_lr()[0]

        ep_loss, ep_rank, ep_nce = 0., 0., 0.
        ep_micro_acc, ep_orig_acc = 0., 0.
        ep_pos_e, ep_micro_e, ep_orig_e = 0., 0., 0.
        n = 0
        t0 = time.time()

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for z, a_pos, a_micro, a_med, a_large, a_xl, a_orig in pbar:
            z = z.to(device, non_blocking=True)
            a_pos = a_pos.to(device, non_blocking=True)
            a_micro = a_micro.to(device, non_blocking=True)
            a_med = a_med.to(device, non_blocking=True)
            a_large = a_large.to(device, non_blocking=True)
            a_xl = a_xl.to(device, non_blocking=True)
            a_orig = a_orig.to(device, non_blocking=True)

            z_seq = z.unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=args.amp):
                e_pos = model(z_seq, a_pos.unsqueeze(1))
                e_micro = model(z_seq, a_micro.unsqueeze(1))
                e_med = model(z_seq, a_med.unsqueeze(1))
                e_large = model(z_seq, a_large.unsqueeze(1))
                e_xl = model(z_seq, a_xl.unsqueeze(1))
                e_orig = model(z_seq, a_orig.unsqueeze(1))

                ones = torch.ones(len(z), device=device)
                l_r1 = F.margin_ranking_loss(e_micro, e_pos, target=ones, margin=0.1)
                l_r2 = F.margin_ranking_loss(e_med, e_micro, target=ones, margin=0.2)
                l_r3 = F.margin_ranking_loss(e_large, e_med, target=ones, margin=0.3)
                l_r4 = F.margin_ranking_loss(e_xl, e_large, target=ones, margin=0.3)
                loss_rank = l_r1 + l_r2 + l_r3 + l_r4

                logits = torch.stack([-e_pos, -e_orig], dim=1)
                loss_nce = F.cross_entropy(
                    logits, torch.zeros(len(z), dtype=torch.long, device=device))

                loss = loss_rank + args.nce_weight * loss_nce

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                micro_acc = (e_pos < e_micro).float().mean().item()
                orig_acc = (e_pos < e_orig).float().mean().item()

            ep_loss += loss.item()
            ep_rank += loss_rank.item()
            ep_nce += loss_nce.item()
            ep_micro_acc += micro_acc
            ep_orig_acc += orig_acc
            ep_pos_e += e_pos.mean().item()
            ep_micro_e += e_micro.mean().item()
            ep_orig_e += e_orig.mean().item()
            n += 1

            if n % 50 == 0:
                pbar.set_postfix(
                    loss=f"{loss.item():.3f}",
                    micro=f"{micro_acc:.3f}",
                    orig=f"{orig_acc:.3f}",
                )

        if epoch >= args.warmup_epochs:
            scheduler.step()

        al, ar, an = ep_loss/n, ep_rank/n, ep_nce/n
        am, ao = ep_micro_acc/n, ep_orig_acc/n
        pe, me, oe = ep_pos_e/n, ep_micro_e/n, ep_orig_e/n
        elapsed = time.time() - t0
        margin = me - pe

        log(f"  Epoch {epoch+1:3d}/{args.epochs} | loss={al:.4f} (rank={ar:.4f} nce={an:.4f}) | "
            f"micro_acc={am:.3f} orig_acc={ao:.3f} | "
            f"E_pos={pe:.3f} E_micro={me:.3f} margin={margin:.3f} | "
            f"lr={lr_now:.2e} | {elapsed:.1f}s")

        if am > best_micro_acc:
            best_micro_acc = am
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch+1,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "micro_acc": am, "orig_acc": ao, "loss": al},
                       os.path.join(args.output_dir, "cewm_multiscale_best.pt"))

        if (epoch + 1) % args.save_interval == 0:
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch+1,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "micro_acc": am, "orig_acc": ao, "loss": al},
                       os.path.join(args.output_dir, f"cewm_epoch{epoch+1:04d}.pt"))

    torch.save({"model_state_dict": model.state_dict(), "epoch": args.epochs,
                "micro_acc": am, "orig_acc": ao, "loss": al},
               os.path.join(args.output_dir, "cewm_multiscale_final.pt"))
    log(f"\nDone! Best micro_acc: {best_micro_acc:.3f}")
    log_f.close()


if __name__ == "__main__":
    main()
