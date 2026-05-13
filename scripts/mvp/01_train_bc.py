"""MVP Day 2: 训练 BC Proxy 策略。

训练一个 ~1M 参数的 BC (Behavior Cloning) CNN,
模拟"会被 OOD 视觉干扰击垮的 VLA"。

输入: RGB image [3, 200, 200] (ImageNet 归一化)
输出: rel_action [7] (dx,dy,dz,droll,dpitch,dyaw,gripper)

保存: checkpoints/mvp/bc_proxy.pt
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)

# ============================================================
# 超参数
# ============================================================
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "calvin_debug_dataset")
CKPT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "mvp")

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ============================================================
# BC Proxy Model (~1M params)
# ============================================================
class BCProxy(nn.Module):
    """4-layer CNN + 2-layer MLP, ~1M params."""

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
# 简易数据集：全部加载进内存（2777 帧 ≈ 1.3GB）
# ============================================================
class SimpleCalvinDataset(Dataset):
    def __init__(self, data_dir, split="training"):
        split_dir = os.path.join(data_dir, split)
        npz_files = sorted(
            os.path.join(split_dir, f)
            for f in os.listdir(split_dir)
            if f.endswith(".npz")
        )
        print(f"Loading {len(npz_files)} frames from {split}...")

        rgbs, actions = [], []
        for path in tqdm(npz_files, desc="Loading data"):
            data = np.load(path)
            rgb = data["rgb_static"].astype(np.float32) / 255.0
            rgb = (rgb - MEAN) / STD
            rgb = rgb.transpose(2, 0, 1)  # HWC → CHW
            rgbs.append(rgb)
            actions.append(data["rel_actions"].astype(np.float32))

        self.rgbs = np.stack(rgbs)
        self.actions = np.stack(actions)
        print(f"Loaded: rgb {self.rgbs.shape}, actions {self.actions.shape}")

    def __len__(self):
        return len(self.rgbs)

    def __getitem__(self, idx):
        return {
            "rgb": torch.from_numpy(self.rgbs[idx]),
            "action": torch.from_numpy(self.actions[idx]),
        }


def main():
    print("=" * 60)
    print("CE-AIS MVP: Train BC Proxy")
    print("=" * 60)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    dataset = SimpleCalvinDataset(DATA_DIR, "training")
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True
    )

    model = BCProxy().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nBC Proxy params: {n_params:,} ({n_params / 1e6:.2f}M)")
    print(f"Device: {DEVICE}")
    print(f"Epochs: {EPOCHS}, Batch: {BATCH_SIZE}, LR: {LR}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_loss = float("inf")
    print("\nTraining...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            rgb = batch["rgb"].to(DEVICE)
            action_gt = batch["action"].to(DEVICE)

            pred = model(rgb)
            loss = nn.functional.mse_loss(pred, action_gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{EPOCHS}  "
                f"loss={avg_loss:.6f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        if avg_loss < best_loss:
            best_loss = avg_loss

    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, "bc_proxy.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "action_dim": 7,
            "n_params": n_params,
            "best_loss": best_loss,
        },
        ckpt_path,
    )
    print(f"\nSaved: {ckpt_path}")
    print(f"Best training loss: {best_loss:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
