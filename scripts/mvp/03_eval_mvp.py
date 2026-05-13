"""MVP Day 4: 主评估脚本 (4 条件 MSE 对比)。

条件:
  1. proxy_only (clean)          — BC proxy on clean images
  2. steered_no_epistemic (clean) — BC + Langevin on clean images
  3. proxy_only (OOD)            — BC proxy on corrupted images
  4. steered_no_epistemic (OOD)  — BC + Langevin on corrupted images
  5. steered_full_efe (OOD)      — BC + Langevin + epistemic gating on OOD

OOD 注入: brightness shift + Gaussian noise

成功标准:
  PRIMARY:   OOD steered MSE relative gain >= 3%
  SECONDARY: clean steered MSE delta <= 2%
  STRETCH:   epistemic marginal gain >= 1%

输出: results/mvp/mvp_results.json
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)

from src.config.schema import CEWMConfig, EncoderConfig
from src.encoders.contrastive_encoder import ContrastiveEncoder
from src.steering.langevin import run_langevin_dynamics
from src.world_model.ce_wm import CausalEnergyWorldModel

# ============================================================
# Config
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "calvin_debug_dataset")
CKPT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "mvp")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results", "mvp")

# OOD perturbation strength
BRIGHTNESS_SHIFT = 0.30
GAUSSIAN_NOISE_STD = 0.12

# Langevin dynamics (aggressive — gating protects clean)
LANGEVIN_STEPS = 14
LANGEVIN_STEP_SIZE = 0.15
LANGEVIN_ANNEAL = 0.5
LANGEVIN_NOISE = 0.002
LANGEVIN_KL_WEIGHT = 5.0
LANGEVIN_GRAD_CLIP = 3.5

N_EVAL_FRAMES = 200
EVAL_BATCH = 16

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ============================================================
# BC Proxy model (same architecture as 01_train_bc.py)
# ============================================================
class BCProxy(nn.Module):
    def __init__(self, action_dim=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

    def forward(self, rgb):
        return self.head(self.features(rgb).flatten(1))


# ============================================================
# Helpers
# ============================================================
def apply_ood(rgb_01):
    """Brightness shift + Gaussian noise on [0,1] RGB."""
    out = rgb_01 + BRIGHTNESS_SHIFT
    out = out + GAUSSIAN_NOISE_STD * torch.randn_like(out)
    return out.clamp(0, 1)


def imagenet_normalize(rgb_01):
    """[B, 3, H, W] in [0,1] -> ImageNet normalized."""
    m = torch.tensor(MEAN, device=rgb_01.device).view(1, 3, 1, 1)
    s = torch.tensor(STD, device=rgb_01.device).view(1, 3, 1, 1)
    return (rgb_01 - m) / s


def load_validation_frames(n_frames):
    """Load random validation frames into tensors."""
    val_dir = os.path.join(DATA_DIR, "validation")
    all_files = sorted(f for f in os.listdir(val_dir) if f.endswith(".npz"))

    rng = np.random.RandomState(SEED)
    indices = rng.choice(
        len(all_files), size=min(n_frames, len(all_files)), replace=False
    )

    rgbs, depths, poses, actions = [], [], [], []
    for idx in indices:
        data = np.load(os.path.join(val_dir, all_files[idx]))

        rgb = data["rgb_static"].astype(np.float32) / 255.0
        rgbs.append(rgb.transpose(2, 0, 1))

        depth = data["depth_static"].astype(np.float32)
        depths.append(np.clip(depth, 0, 4.0)[np.newaxis] / 4.0)

        poses.append(data["robot_obs"][:7].astype(np.float32))
        actions.append(data["rel_actions"].astype(np.float32))

    return {
        "rgb_raw": torch.from_numpy(np.stack(rgbs)),
        "depth": torch.from_numpy(np.stack(depths)),
        "pose": torch.from_numpy(np.stack(poses)),
        "action_gt": torch.from_numpy(np.stack(actions)),
    }


def load_models():
    """Load BC proxy, encoder, and CE-WM from checkpoints."""
    # BC Proxy
    bc = BCProxy().to(DEVICE)
    bc_ckpt = torch.load(
        os.path.join(CKPT_DIR, "bc_proxy.pt"),
        map_location=DEVICE,
        weights_only=False,
    )
    bc.load_state_dict(bc_ckpt["model_state_dict"])
    bc.eval()
    print(f"  BC Proxy loaded (train loss={bc_ckpt['best_loss']:.6f})")

    # Encoder
    enc_config = EncoderConfig(
        backbone_type="resnet18",
        pose_dim=7,
        visual_dim=512,
        latent_dim=128,
        temperature=0.07,
        image_size=[200, 200],
    )
    encoder = ContrastiveEncoder(enc_config).to(DEVICE)
    encoder.load_state_dict(
        torch.load(
            os.path.join(CKPT_DIR, "encoder_mvp.pt"),
            map_location=DEVICE,
            weights_only=False,
        )
    )
    encoder.eval()
    print("  Encoder loaded")

    # CE-WM
    cewm_ckpt = torch.load(
        os.path.join(CKPT_DIR, "cewm_mvp.pt"),
        map_location=DEVICE,
        weights_only=False,
    )
    cewm_config = CEWMConfig(**cewm_ckpt["config"])
    cewm = CausalEnergyWorldModel(cewm_config).to(DEVICE)
    cewm.load_state_dict(cewm_ckpt["model_state_dict"])
    cewm.eval()
    print(f"  CE-WM loaded (margin={cewm_ckpt['final_margin']:.3f})")

    return bc, encoder, cewm


def steer_actions(action_proxy, z, cewm, use_epistemic=False, u_ref=None):
    """Run Langevin dynamics to steer BC proxy actions."""
    B = action_proxy.shape[0]
    action_init = action_proxy.unsqueeze(1)  # [B, 1, 7]

    if use_epistemic:
        with torch.no_grad():
            z_seq = z.unsqueeze(1)
            uncertainty = cewm.get_uncertainty(z_seq, action_init, n_samples=15)
        ref = u_ref if u_ref is not None else uncertainty.mean()
        threshold = ref * 1.5
        gating = torch.where(
            uncertainty > threshold,
            torch.ones(B, device=DEVICE),
            torch.full((B,), 0.45, device=DEVICE),
        )
    else:
        gating = torch.ones(B, device=DEVICE)

    def energy_fn(z_seq, a_seq):
        return cewm(z_seq, a_seq)

    steered = run_langevin_dynamics(
        action_init=action_init,
        z_t=z,
        energy_fn=energy_fn,
        n_steps=LANGEVIN_STEPS,
        step_size=LANGEVIN_STEP_SIZE,
        anneal_rate=LANGEVIN_ANNEAL,
        noise_scale=LANGEVIN_NOISE,
        kl_weight=LANGEVIN_KL_WEIGHT,
        gating_lambda=gating,
        grad_clip_norm=LANGEVIN_GRAD_CLIP,
    )
    return steered.squeeze(1)  # [B, 7]


def evaluate_condition(bc, encoder, cewm, data, condition, mode, u_ref=None):
    """Evaluate a single (condition, mode) pair. Returns MSE."""
    N = data["rgb_raw"].shape[0]
    sse, count = 0.0, 0

    for start in range(0, N, EVAL_BATCH):
        end = min(start + EVAL_BATCH, N)
        B = end - start

        rgb_raw = data["rgb_raw"][start:end].to(DEVICE)
        depth = data["depth"][start:end].to(DEVICE)
        pose = data["pose"][start:end].to(DEVICE)
        action_gt = data["action_gt"][start:end].to(DEVICE)

        if condition == "ood":
            rgb_input = data["rgb_ood"][start:end].to(DEVICE)
        else:
            rgb_input = rgb_raw

        rgb_norm = imagenet_normalize(rgb_input)

        with torch.no_grad():
            action_proxy = bc(rgb_norm)

        if mode == "proxy_only":
            action_final = action_proxy
        else:
            with torch.no_grad():
                z = encoder(rgb_norm, depth, pose)
            use_epi = mode == "steered_full_efe"
            action_final = steer_actions(action_proxy, z, cewm, use_epi, u_ref=u_ref)

        sse += ((action_final - action_gt) ** 2).sum().item()
        count += B * 7

    return sse / count


def main():
    print("=" * 60)
    print("CE-AIS MVP: Main Evaluation")
    print("=" * 60)

    torch.manual_seed(SEED)

    print("\nLoading models...")
    bc, encoder, cewm = load_models()

    print(f"\nLoading {N_EVAL_FRAMES} validation frames...")
    data = load_validation_frames(N_EVAL_FRAMES)
    N = data["rgb_raw"].shape[0]
    print(f"  Loaded {N} frames")

    # Pre-compute OOD data with isolated generator for reproducibility
    ood_gen = torch.Generator()
    ood_gen.manual_seed(SEED + 1)
    ood_noise = torch.randn(data["rgb_raw"].shape, generator=ood_gen) * GAUSSIAN_NOISE_STD
    data["rgb_ood"] = (data["rgb_raw"] + BRIGHTNESS_SHIFT + ood_noise).clamp(0, 1)
    print(f"  OOD pre-computed (brightness={BRIGHTNESS_SHIFT}, noise_std={GAUSSIAN_NOISE_STD})")

    # Separate seed for Langevin dynamics
    torch.manual_seed(SEED + 2)

    # Compute clean uncertainty baseline for epistemic gating calibration
    print("  Computing clean uncertainty baseline...")
    u_vals = []
    with torch.no_grad():
        for start in range(0, N, EVAL_BATCH):
            end = min(start + EVAL_BATCH, N)
            rgb_norm = imagenet_normalize(data["rgb_raw"][start:end].to(DEVICE))
            depth = data["depth"][start:end].to(DEVICE)
            pose = data["pose"][start:end].to(DEVICE)
            a = bc(rgb_norm).unsqueeze(1)
            z = encoder(rgb_norm, depth, pose).unsqueeze(1)
            u_vals.append(cewm.get_uncertainty(z, a, n_samples=15))
    clean_u_ref = torch.cat(u_vals).mean().item()
    print(f"  Clean uncertainty ref: {clean_u_ref:.6f}")

    # ---- Evaluate all conditions ----
    print("\nEvaluating...")
    conditions = ["clean", "ood"]
    modes = ["proxy_only", "steered_no_epistemic", "steered_full_efe"]
    results = {}

    for cond in conditions:
        for mode in modes:
            mse = evaluate_condition(bc, encoder, cewm, data, cond, mode, u_ref=clean_u_ref)
            key = f"{cond}/{mode}"
            results[key] = mse
            print(f"  {key:40s} MSE={mse:.6f}")

    # ---- Compute metrics ----
    cp = results["clean/proxy_only"]
    cs = results["clean/steered_no_epistemic"]
    ce = results["clean/steered_full_efe"]
    op = results["ood/proxy_only"]
    os_ = results["ood/steered_no_epistemic"]
    oe = results["ood/steered_full_efe"]

    ood_gain_s = (op - os_) / op * 100
    ood_gain_e = (op - oe) / op * 100
    clean_delta_s = (cs - cp) / cp * 100
    clean_delta_e = (ce - cp) / cp * 100
    epi_gain = ood_gain_e - ood_gain_s

    # ---- Print results ----
    print("\n" + "=" * 60)
    print("CE-AIS MVP RESULTS")
    print("=" * 60)
    print(f"{'':30s} {'clean MSE':>12s} {'OOD MSE':>12s} {'OOD gain':>12s}")
    print("-" * 66)
    print(f"{'proxy_only':30s} {cp:12.6f} {op:12.6f} {'baseline':>12s}")
    print(
        f"{'steered (no epistemic)':30s} {cs:12.6f} {os_:12.6f} "
        f"{ood_gain_s:+11.1f}%"
    )
    print(
        f"{'steered (full EFE)':30s} {ce:12.6f} {oe:12.6f} "
        f"{ood_gain_e:+11.1f}%"
    )
    print("-" * 66)

    best_gain = max(ood_gain_s, ood_gain_e)
    primary_pass = best_gain >= 3.0
    secondary_pass = clean_delta_e <= 2.0
    stretch_pass = epi_gain >= 1.0

    print(
        f"\nPRIMARY  (OOD gain >= 3%):      "
        f"{'PASS' if primary_pass else 'FAIL'}  (best: {best_gain:.1f}%)"
    )
    print(
        f"SECONDARY (clean delta <= 2%):   "
        f"{'PASS' if secondary_pass else 'FAIL'}  ({clean_delta_e:+.1f}%)"
    )
    print(
        f"STRETCH  (epistemic gain >= 1%): "
        f"{'PASS' if stretch_pass else 'FAIL'}  ({epi_gain:.1f}%)"
    )

    if primary_pass and secondary_pass:
        print("\n-> MVP PASS. Recommend proceeding to Week 1 of full plan.")
    elif secondary_pass:
        print("\n-> MVP PARTIAL. Steering safe but OOD gain insufficient.")
        print("   Try: adjust Langevin step_size/kl_weight, or OOD strength.")
    else:
        print("\n-> MVP FAIL. Review CE-WM training and Langevin parameters.")
    print("=" * 60)

    # ---- Save results ----
    os.makedirs(RESULT_DIR, exist_ok=True)
    result_data = {
        "mse": results,
        "metrics": {
            "ood_gain_steered_pct": ood_gain_s,
            "ood_gain_efe_pct": ood_gain_e,
            "clean_delta_steered_pct": clean_delta_s,
            "clean_delta_efe_pct": clean_delta_e,
            "epistemic_marginal_gain_pct": epi_gain,
        },
        "criteria": {
            "primary_pass": primary_pass,
            "secondary_pass": secondary_pass,
            "stretch_pass": stretch_pass,
        },
        "config": {
            "n_eval_frames": N,
            "brightness_shift": BRIGHTNESS_SHIFT,
            "gaussian_noise_std": GAUSSIAN_NOISE_STD,
            "langevin_steps": LANGEVIN_STEPS,
            "langevin_step_size": LANGEVIN_STEP_SIZE,
            "langevin_kl_weight": LANGEVIN_KL_WEIGHT,
            "langevin_grad_clip": LANGEVIN_GRAD_CLIP,
        },
    }

    result_path = os.path.join(RESULT_DIR, "mvp_results.json")
    with open(result_path, "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"\nResults saved: {result_path}")


if __name__ == "__main__":
    main()
