"""MVP Day 3: 预训练小 CE-WM (NCE 能量景观)。

两阶段训练:
  Stage 1: ContrastiveEncoder 预训练 (InfoNCE, ~20 epochs)
  Stage 2: CausalEnergyWorldModel 预训练 (NCE, ~50 epochs)

MVP 配置:
  Encoder: ResNet18 backbone, d_z=128
  CE-WM:   d_model=128, n_layers=4, d_state=16

成功标准: 能量 margin (E_neg - E_pos) > 1.0

保存:
  checkpoints/mvp/encoder_mvp.pt
  checkpoints/mvp/cewm_mvp.pt
"""

import os
import sys

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)

from src.config.schema import CEWMConfig, EncoderConfig
from src.data.calvin_dataset import CALVINDataset
from src.data.perturbation import PerturbationRegistry
from src.encoders.contrastive_encoder import ContrastiveEncoder
from src.training.losses import InfoNCELoss, NCELoss
from src.world_model.ce_wm import CausalEnergyWorldModel

# ============================================================
# MVP 超参数
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "calvin_debug_dataset")
CKPT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "mvp")

# Encoder training
ENC_EPOCHS = 20
ENC_BATCH = 32
ENC_LR = 3e-4

# CE-WM training
CEWM_EPOCHS = 50
CEWM_BATCH = 16
CEWM_LR = 1e-4
CEWM_WINDOW = 8
NEG_RATIO = 3
USE_AMP = True

# CALVINDataset config dict
DATA_CONFIG = {
    "encoder": {"image_size": [200, 200], "pose_dim": 7, "latent_dim": 128},
    "ce_wm": {"action_dim": 7},
    "data": {"window_size": CEWM_WINDOW, "encoder_pos_offset_max": 4},
}


def train_encoder():
    """Stage 1: Contrastive Encoder 预训练 (InfoNCE)."""
    print("\n" + "=" * 60)
    print("Stage 1: Contrastive Encoder Pre-training")
    print("=" * 60)

    enc_config = EncoderConfig(
        backbone_type="resnet18",
        pose_dim=7,
        visual_dim=512,
        latent_dim=128,
        temperature=0.07,
        image_size=[200, 200],
    )
    encoder = ContrastiveEncoder(enc_config).to(DEVICE)
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"Encoder params: {n_params:,} ({n_params / 1e6:.2f}M)")

    dataset = CALVINDataset(
        DATA_DIR, "training", config=DATA_CONFIG, mode="encoder"
    )
    loader = DataLoader(
        dataset, batch_size=ENC_BATCH, shuffle=True, num_workers=0, drop_last=True
    )
    print(f"Training samples: {len(dataset)}")

    optimizer = torch.optim.AdamW(
        encoder.parameters(), lr=ENC_LR, weight_decay=1e-5
    )
    loss_fn = InfoNCELoss(temperature=0.07)
    scaler = GradScaler(enabled=USE_AMP)

    for epoch in range(1, ENC_EPOCHS + 1):
        encoder.train()
        total_loss, n = 0.0, 0
        for batch in tqdm(loader, desc=f"Enc epoch {epoch}", leave=False):
            rgb = batch["rgb"].to(DEVICE)
            depth = batch["depth"].to(DEVICE)
            pose = batch["pose"].to(DEVICE)
            rgb_p = batch["rgb_pos"].to(DEVICE)
            depth_p = batch["depth_pos"].to(DEVICE)
            pose_p = batch["pose_pos"].to(DEVICE)

            optimizer.zero_grad()
            with autocast(enabled=USE_AMP):
                z_a = encoder(rgb, depth, pose)
                z_p = encoder(rgb_p, depth_p, pose_p)
                loss = loss_fn(z_a, z_p)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            n += 1

        avg = total_loss / max(n, 1)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{ENC_EPOCHS}  InfoNCE loss={avg:.4f}")

    os.makedirs(CKPT_DIR, exist_ok=True)
    enc_path = os.path.join(CKPT_DIR, "encoder_mvp.pt")
    torch.save(encoder.state_dict(), enc_path)
    print(f"Saved: {enc_path}")
    return encoder


def train_cewm(encoder):
    """Stage 2: CE-WM 预训练 (NCE 能量景观)."""
    print("\n" + "=" * 60)
    print("Stage 2: CE-WM Pre-training (NCE)")
    print("=" * 60)

    cewm_config = CEWMConfig(
        d_model=128,
        d_state=16,
        n_layers=4,
        expand_factor=2,
        mimo_groups=2,
        action_dim=7,
        latent_dim=128,
        dropout=0.1,
    )
    cewm = CausalEnergyWorldModel(cewm_config).to(DEVICE)
    n_params = sum(p.numel() for p in cewm.parameters())
    print(f"CE-WM params: {n_params:,} ({n_params / 1e6:.2f}M)")

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    dataset = CALVINDataset(
        DATA_DIR, "training", config=DATA_CONFIG, mode="ce_wm"
    )
    loader = DataLoader(
        dataset, batch_size=CEWM_BATCH, shuffle=True, num_workers=0, drop_last=True
    )
    print(f"Training samples: {len(dataset)}")

    optimizer = torch.optim.AdamW(
        cewm.parameters(), lr=CEWM_LR, weight_decay=1e-5
    )
    loss_fn = NCELoss(temperature=1.0)
    scaler = GradScaler(enabled=USE_AMP)
    strategies = ["velocity_reversal", "gripper_anomaly", "random_displacement"]

    final_margin = 0.0
    for epoch in range(1, CEWM_EPOCHS + 1):
        cewm.train()
        total_loss, n = 0.0, 0
        pos_sum, neg_sum = 0.0, 0.0
        pos_cnt, neg_cnt = 0, 0

        for batch in tqdm(loader, desc=f"CEWM epoch {epoch}", leave=False):
            rgb_seq = batch["rgb_seq"].to(DEVICE)
            depth_seq = batch["depth_seq"].to(DEVICE)
            pose_seq = batch["pose_seq"].to(DEVICE)
            a_pos = batch["a_pos"].to(DEVICE)

            B, T = rgb_seq.shape[0], rgb_seq.shape[1]

            with torch.no_grad():
                z_flat = encoder(
                    rgb_seq.view(B * T, 3, 200, 200),
                    depth_seq.view(B * T, 1, 200, 200),
                    pose_seq.view(B * T, 7),
                )
                z_seq = z_flat.view(B, T, -1)

            a_neg_list = []
            for k in range(NEG_RATIO):
                strat = strategies[k % len(strategies)]
                a_neg_list.append(PerturbationRegistry.apply(strat, a_pos))
            a_neg = torch.stack(a_neg_list, dim=1)  # [B, K, T, 7]

            optimizer.zero_grad()
            with autocast(enabled=USE_AMP):
                energy_pos = cewm(z_seq, a_pos)

                K = a_neg.shape[1]
                a_neg_flat = a_neg.view(B * K, T, 7)
                z_exp = z_seq.repeat_interleave(K, dim=0)
                energy_neg = cewm(z_exp, a_neg_flat).view(B, K)

                loss = loss_fn(energy_pos, energy_neg)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n += 1

            with torch.no_grad():
                pos_sum += energy_pos.float().sum().item()
                pos_cnt += energy_pos.numel()
                neg_sum += energy_neg.float().sum().item()
                neg_cnt += energy_neg.numel()

        avg_loss = total_loss / max(n, 1)
        pos_mean = pos_sum / max(pos_cnt, 1)
        neg_mean = neg_sum / max(neg_cnt, 1)
        final_margin = neg_mean - pos_mean

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{CEWM_EPOCHS}  "
                f"NCE loss={avg_loss:.4f}  "
                f"E_pos={pos_mean:.3f}  E_neg={neg_mean:.3f}  "
                f"margin={final_margin:.3f}"
            )

    cewm_path = os.path.join(CKPT_DIR, "cewm_mvp.pt")
    torch.save(
        {
            "model_state_dict": cewm.state_dict(),
            "config": {
                "d_model": 128,
                "d_state": 16,
                "n_layers": 4,
                "expand_factor": 2,
                "mimo_groups": 2,
                "action_dim": 7,
                "latent_dim": 128,
                "dropout": 0.1,
            },
            "final_margin": final_margin,
        },
        cewm_path,
    )
    print(f"\nSaved: {cewm_path}")
    print(f"Final energy margin: {final_margin:.3f} (target > 1.0)")

    if final_margin > 1.0:
        print("  Margin > 1.0: NCE training successful")
    else:
        print("  Margin < 1.0: may need more epochs or tuning")

    return cewm


def main():
    print("=" * 60)
    print("CE-AIS MVP: Train Encoder + CE-WM")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    torch.manual_seed(SEED)

    encoder = train_encoder()
    train_cewm(encoder)

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
