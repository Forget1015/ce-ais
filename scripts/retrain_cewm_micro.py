#!/usr/bin/env python3
"""CE-WM Micro-Scale Retrain: 解冻后8层+改造energy head+micro negatives。

解决 4280x train-test action scale mismatch:
- 解冻 Mamba 后8层 + energy head（~30M params）
- energy head 增加 action skip connection（直接看 raw action）
- 多尺度 micro negatives: sigma ∈ {3e-3, 5e-3, 1e-2, 2e-2, 5e-2}
- 混合 loss: micro ranking + 原始 NCE（不丢失大尺度能力）

用法:
  python scripts/retrain_cewm_micro.py \
    --data-file data/rollout_flower_seed42/contrastive_pairs.npz \
    --cewm-ckpt checkpoints_calibrated_cewm/checkpoints_finetuned_energy_head_v2/energy_head_v2_best.pt \
    --output-dir checkpoints_calibrated_cewm/checkpoints_micro_retrain \
    --epochs 30 --lr 5e-5 --batch-size 512 --device cuda:4
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

MICRO_SCALES = [3e-3, 5e-3, 1e-2, 2e-2, 5e-2]


class MicroScaleDataset(Dataset):
    """在线生成多尺度 micro negatives。"""

    def __init__(self, pairs_file: str):
        data = np.load(pairs_file)
        self.anchor_z = data["anchor_z"]
        self.pos_action = data["pos_action"]
        self.neg_action_orig = data["neg_action"]
        print(f"  Loaded {len(self.anchor_z)} samples")

    def __len__(self):
        return len(self.anchor_z)

    def __getitem__(self, idx):
        z = torch.from_numpy(self.anchor_z[idx])
        a_pos = torch.from_numpy(self.pos_action[idx])
        a_neg_orig = torch.from_numpy(self.neg_action_orig[idx])
        sigma_s = MICRO_SCALES[torch.randint(0, 2, (1,)).item()]
        sigma_l = MICRO_SCALES[torch.randint(2, 5, (1,)).item()]
        a_neg_small = a_pos + torch.randn_like(a_pos) * sigma_s
        a_neg_large = a_pos + torch.randn_like(a_pos) * sigma_l
        return z, a_pos, a_neg_small, a_neg_large, a_neg_orig


class EnergyHeadWithAction(nn.Module):
    """Energy head with action skip connection."""

    def __init__(self, d_model: int, action_dim: int = 7):
        super().__init__()
        d_in = d_model + action_dim
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_in // 2),
            nn.GELU(),
            nn.Linear(d_in // 2, 1),
        )

    def forward(self, h, a):
        return self.mlp(torch.cat([h, a], dim=-1)).squeeze(-1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-file", default="data/rollout_flower_seed42/contrastive_pairs.npz")
    p.add_argument("--cewm-ckpt", default="checkpoints_calibrated_cewm/checkpoints_finetuned_energy_head_v2/energy_head_v2_best.pt")
    p.add_argument("--output-dir", default="checkpoints_calibrated_cewm/checkpoints_micro_retrain")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--unfreeze-layers", type=int, default=8)
    p.add_argument("--micro-weight", type=float, default=1.0)
    p.add_argument("--orig-weight", type=float, default=0.3)
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

    log("[1/4] Loading CE-WM...")
    from src.config.config_manager import ConfigManager
    from src.config.schema import CEWMConfig
    from src.world_model.ce_wm import CausalEnergyWorldModel

    cm = ConfigManager(config_path="configs/base.yaml")
    cfg = cm.config.get("ce_wm", {})
    d_model = cfg.get("d_model", 640)
    action_dim = cfg.get("action_dim", 7)

    model = CausalEnergyWorldModel(CEWMConfig(
        d_model=d_model, d_state=cfg.get("d_state", 64),
        n_layers=cfg.get("n_layers", 32),
        expand_factor=cfg.get("expand_factor", 3),
        mimo_groups=cfg.get("mimo_groups", 4),
        action_dim=action_dim, latent_dim=cfg.get("latent_dim", 128),
        dropout=cfg.get("dropout", 0.1), headdim=cfg.get("headdim", 64),
    )).to(device)

    ckpt = torch.load(args.cewm_ckpt, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state, strict=False)
    log(f"  Loaded from {args.cewm_ckpt}")

    log("[2/4] Replacing energy head with action skip connection...")
    old_w = model.energy_head.mlp[0].weight.data.clone()
    old_b = model.energy_head.mlp[0].bias.data.clone()
    new_head = EnergyHeadWithAction(d_model, action_dim).to(device)
    with torch.no_grad():
        w = new_head.mlp[0].weight
        w[:old_w.shape[0], :old_w.shape[1]] = old_w
        new_head.mlp[0].bias[:old_b.shape[0]] = old_b
    model.energy_head = new_head
    log(f"  New head input: {d_model}+{action_dim}={d_model+action_dim}")

    n_layers = len(model.mamba_layers)
    freeze_until = n_layers - args.unfreeze_layers
    for p in model.parameters():
        p.requires_grad_(False)
    for i in range(freeze_until, n_layers):
        for p in model.mamba_layers[i].parameters():
            p.requires_grad_(True)
    for p in model.energy_head.parameters():
        p.requires_grad_(True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"  Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

    log("[3/4] Loading dataset...")
    dataset = MicroScaleDataset(args.data_file)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)
    log(f"  Batches per epoch: {len(loader)}")

    log("[4/4] Training...")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    model.train()

    for epoch in range(args.epochs):
        ep_loss, ep_micro, ep_orig, n = 0., 0., 0., 0
        t0 = time.time()

        for z, a_pos, a_neg_s, a_neg_l, a_neg_o in loader:
            z, a_pos = z.to(device), a_pos.to(device)
            a_neg_s, a_neg_l = a_neg_s.to(device), a_neg_l.to(device)
            a_neg_o = a_neg_o.to(device)

            z_seq = z.unsqueeze(1)

            def fwd(a):
                a_seq = a.unsqueeze(1)
                x = model.input_proj(torch.cat([z_seq, a_seq], dim=-1))
                for layer in model.mamba_layers:
                    x = layer(x)
                return model.energy_head(x[:, -1, :], a)

            e_pos = fwd(a_pos)
            e_ns = fwd(a_neg_s)
            e_nl = fwd(a_neg_l)
            e_no = fwd(a_neg_o)

            ones = torch.ones(len(z), device=device)
            l_r1 = F.margin_ranking_loss(e_ns, e_pos, target=ones, margin=0.1)
            l_r2 = F.margin_ranking_loss(e_nl, e_ns, target=ones, margin=0.2)
            l_micro = l_r1 + l_r2

            logits = torch.stack([-e_pos, -e_no], dim=1)
            l_orig = F.cross_entropy(logits, torch.zeros(len(z), dtype=torch.long, device=device))

            loss = args.micro_weight * l_micro + args.orig_weight * l_orig
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ma = ((e_pos < e_ns).float().mean().item() + (e_ns < e_nl).float().mean().item()) / 2
            oa = (e_pos < e_no).float().mean().item()
            ep_loss += loss.item(); ep_micro += ma; ep_orig += oa; n += 1

        scheduler.step()
        al, am, ao = ep_loss/n, ep_micro/n, ep_orig/n
        log(f"  Epoch {epoch+1:3d}/{args.epochs} | loss={al:.4f} | micro={am:.3f} | orig={ao:.3f} | lr={scheduler.get_last_lr()[0]:.2e} | {time.time()-t0:.1f}s")

        if am > best_acc:
            best_acc = am
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch+1,
                        "micro_acc": am, "orig_acc": ao},
                       os.path.join(args.output_dir, "cewm_micro_best.pt"))

    torch.save({"model_state_dict": model.state_dict(), "epoch": args.epochs,
                "micro_acc": am, "orig_acc": ao},
               os.path.join(args.output_dir, "cewm_micro_final.pt"))
    log(f"\nDone! Best micro_acc: {best_acc:.3f}")
    log_f.close()


if __name__ == "__main__":
    main()
