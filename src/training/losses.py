"""NCE / InfoNCE 损失函数。

- InfoNCELoss: 用于 Contrastive Encoder 预训练（时序对比学习）
- NCELoss:     用于 CE-WM 能量景观训练（正负样本能量区分）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    InfoNCE 对比损失。

    用于 Contrastive Encoder 预训练：使时序相邻的合法状态在潜空间中
    距离接近，时序无关的状态距离远离。

    L = -log[ exp(sim(z_i, z_pos) / τ) / Σ_j exp(sim(z_i, z_j) / τ) ]

    其中 sim(·,·) 为余弦相似度，τ 为温度超参数。
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        self.temperature = temperature

    def forward(
        self, z_anchor: torch.Tensor, z_positive: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 InfoNCE 损失。

        使用批内其他样本作为负样本（in-batch negatives）。

        Args:
            z_anchor:   [B, d_z] 锚点嵌入（已 L2 归一化）
            z_positive: [B, d_z] 正样本嵌入（已 L2 归一化）

        Returns:
            loss: 标量损失值
        """
        # 归一化（防御性，即使输入已归一化）
        z_anchor = F.normalize(z_anchor, dim=-1)
        z_positive = F.normalize(z_positive, dim=-1)

        B = z_anchor.shape[0]

        # 余弦相似度矩阵: [B, B]
        # logits[i, j] = sim(z_anchor_i, z_positive_j) / τ
        logits = torch.mm(z_anchor, z_positive.t()) / self.temperature

        # 对角线为正样本对，标签为 [0, 1, 2, ..., B-1]
        labels = torch.arange(B, device=logits.device)

        return F.cross_entropy(logits, labels)


class NCELoss(nn.Module):
    """
    噪声对比估计损失，用于 CE-WM 能量景观训练。

    正样本（专家轨迹）应获得低能量，负样本（对抗摄动）应获得高能量。
    通过将负能量作为 logits 进行分类：正样本的负能量最大（即能量最低）。

    logits = cat[-E_pos, -E_neg_1, ..., -E_neg_K] / τ
    label  = 0  (第一个位置是正样本)
    loss   = CrossEntropy(logits, label)
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        self.temperature = temperature

    def forward(
        self, energy_pos: torch.Tensor, energy_neg: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 NCE 损失。

        Args:
            energy_pos: [B]    正样本（专家轨迹）能量值
            energy_neg: [B, K] 负样本（对抗摄动）能量值，K 为负样本数

        Returns:
            loss: 标量损失值
        """
        # 拼接: [B, 1+K]，正样本在第 0 列
        logits = torch.cat(
            [-energy_pos.unsqueeze(1), -energy_neg], dim=1
        ) / self.temperature

        # 标签全为 0（正样本在第一个位置）
        labels = torch.zeros(
            logits.size(0), dtype=torch.long, device=logits.device
        )

        return F.cross_entropy(logits, labels)
