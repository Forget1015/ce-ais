#!/usr/bin/env python3
"""Micro-Scale Negatives Finetune: 用多尺度微小 perturbation 重训 energy head。

解决训练-测试 4280x action scale mismatch 问题。
原训练负样本 L∞ gap ≈ 1.13，CE-AIS 测试 correction ≈ 0.0003。
本脚本用 sigma ∈ {5e-4, 1e-3, 2e-3, 5e-3, 1e-2} 的微小高斯噪声
构造负样本，让 energy head 学习在 CE-AIS 工作尺度上的局部能量排序。

用法:
  python scripts/finetune_energy_head_micro.py \
    --data-file data/rollout_flower_seed42/contrastive_pairs.npz \
    --cewm-ckpt checkpoints_calibrated_cewm/checkpoints_finetuned_energy_head_v2/energy_head_v2_best.pt \
    --output-dir checkpoints_calibrated_cewm/checkpoints_micro_scale \
    --epochs 20 --lr 1e-4 --batch-size 1024 --device cuda:4
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

MICRO_SCALES = [5e-4, 1e-3, 2e-3, 5e-3, 1e-2]


class MicroScaleDataset(Dataset):
    """从 expert actions 在线生成多尺度微小负样本。

    每个样本输出:
    - z: [128] state embedding
    - a_pos: [7] expert action
    - a_neg_small: [7] 小尺度扰动 (sigma 从前3个scale随机选)
    - a_neg_large: [7] 大尺度扰动 (sigma 从后2个scale随机选)
    """

    def __init__(self, pairs_file: str):
        data = np.load(pairs_file)
        self.anchor_z = data["anchor_z"]      # [N, 128]
        self.pos_action = data["pos_action"]  # [N, 7]
        print(f"  Loaded {len(self.anchor_z)} samples for micro-scale training")
