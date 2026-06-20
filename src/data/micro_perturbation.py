"""多尺度微小 perturbation 生成器。

用于构造和 CE-AIS 测试时 steering correction 同尺度的负样本,
解决训练-测试之间 4000x 的 action gap mismatch 问题。

设计原则:
- 负样本 sigma 覆盖 CE-AIS 工作尺度: 5e-4 ~ 1e-2
- 支持多尺度 ranking: 大 sigma 的负样本能量应高于小 sigma 的
- 可直接用于 NCE loss 或 margin ranking loss
"""

import torch
import numpy as np
from typing import List, Tuple, Optional


MICRO_SCALES = [5e-4, 1e-3, 2e-3, 5e-3, 1e-2]


def generate_micro_negatives(
    action: torch.Tensor,
    scales: Optional[List[float]] = None,
    n_per_scale: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """为单个 positive action 生成多尺度微小负样本。

    Args:
        action: [d_a] 或 [T, d_a] expert action
        scales: perturbation sigma 列表, 默认 MICRO_SCALES
        n_per_scale: 每个 scale 生成几个负样本

    Returns:
        neg_actions: [K, d_a] 或 [K, T, d_a] 负样本
        neg_scales: [K] 每个负样本对应的 sigma
    """
    if scales is None:
        scales = MICRO_SCALES

    neg_list = []
    scale_list = []

    for sigma in scales:
        for _ in range(n_per_scale):
            noise = torch.randn_like(action) * sigma
            neg_list.append(action + noise)
            scale_list.append(sigma)

    neg_actions = torch.stack(neg_list, dim=0)
    neg_scales = torch.tensor(scale_list, dtype=torch.float32)
    return neg_actions, neg_scales
