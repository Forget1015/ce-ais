"""正负样本数据构造器。

从专家轨迹提取正样本（状态-动作元组），通过摄动策略生成负样本，
组织为 PyTorch DataLoader 批次，支持可配置的正负样本比例 K。
"""

import hashlib
import os
import pickle
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from src.data.perturbation import PerturbationRegistry


class TrajectoryPairDataset(Dataset):
    """正负样本对数据集。

    每个样本包含:
    - z_pos: [T, d_z] 正样本潜变量序列
    - a_pos: [T, d_a] 正样本动作序列
    - a_neg: [K, T, d_a] K 个负样本动作序列
    """

    def __init__(
        self,
        positive_actions: List[torch.Tensor],
        positive_states: List[torch.Tensor],
        neg_ratio: int = 5,
        perturbation_strategies: Optional[List[str]] = None,
    ):
        self.positive_actions = positive_actions
        self.positive_states = positive_states
        self.neg_ratio = neg_ratio
        self.perturbation_strategies = perturbation_strategies or [
            "velocity_reversal",
            "gripper_anomaly",
            "random_displacement",
        ]

    def __len__(self) -> int:
        return len(self.positive_actions)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        a_pos = self.positive_actions[idx]
        z_pos = self.positive_states[idx]

        # 生成 K 个负样本
        neg_actions = []
        strategies = self.perturbation_strategies
        for k in range(self.neg_ratio):
            strategy_name = strategies[k % len(strategies)]
            a_neg = PerturbationRegistry.apply(strategy_name, a_pos.unsqueeze(0))
            neg_actions.append(a_neg.squeeze(0))

        return {
            "z_pos": z_pos,
            "a_pos": a_pos,
            "a_neg": torch.stack(neg_actions, dim=0),  # [K, d_a] or [K, T, d_a]
        }


class DataConstructor:
    """正负样本数据构造器。

    从 CALVIN 专家演示数据构造能量景观训练所需的正负样本对。

    Args:
        neg_ratio: 负样本比例 K（每个正样本对应 K 个负样本）。
        perturbation_strategies: 使用的摄动策略名称列表。
        cache_dir: 缓存目录路径，为 None 时不缓存。
        batch_size: DataLoader 批大小。
        num_workers: DataLoader 工作进程数。
    """

    def __init__(
        self,
        neg_ratio: int = 5,
        perturbation_strategies: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
        batch_size: int = 64,
        num_workers: int = 0,
    ):
        self.neg_ratio = neg_ratio
        self.perturbation_strategies = perturbation_strategies or [
            "velocity_reversal",
            "gripper_anomaly",
            "random_displacement",
        ]
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _cache_key(
        self, actions: List[torch.Tensor], states: List[torch.Tensor]
    ) -> str:
        """生成缓存键。"""
        n = len(actions)
        key_str = (
            f"{n}_{self.neg_ratio}_"
            f"{'_'.join(self.perturbation_strategies)}"
        )
        return hashlib.md5(key_str.encode()).hexdigest()

    def _cache_path(
        self, actions: List[torch.Tensor], states: List[torch.Tensor]
    ) -> str:
        """缓存文件路径。"""
        assert self.cache_dir is not None
        os.makedirs(self.cache_dir, exist_ok=True)
        key = self._cache_key(actions, states)
        return os.path.join(self.cache_dir, f"pairs_{key}.pkl")

    def construct_dataset(
        self,
        positive_actions: List[torch.Tensor],
        positive_states: List[torch.Tensor],
    ) -> TrajectoryPairDataset:
        """构造正负样本对数据集。

        Args:
            positive_actions: 正样本动作列表。
            positive_states: 正样本状态（潜变量）列表。

        Returns:
            TrajectoryPairDataset 实例。
        """
        return TrajectoryPairDataset(
            positive_actions=positive_actions,
            positive_states=positive_states,
            neg_ratio=self.neg_ratio,
            perturbation_strategies=self.perturbation_strategies,
        )

    def construct_dataloader(
        self,
        positive_actions: List[torch.Tensor],
        positive_states: List[torch.Tensor],
    ) -> DataLoader:
        """构造 DataLoader。

        Args:
            positive_actions: 正样本动作列表。
            positive_states: 正样本状态（潜变量）列表。

        Returns:
            PyTorch DataLoader。
        """
        dataset = self.construct_dataset(positive_actions, positive_states)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def construct_and_cache(
        self,
        positive_actions: List[torch.Tensor],
        positive_states: List[torch.Tensor],
    ) -> TrajectoryPairDataset:
        """构造数据集并缓存结果。

        如果缓存存在则直接加载，否则构造后保存缓存。

        Args:
            positive_actions: 正样本动作列表。
            positive_states: 正样本状态（潜变量）列表。

        Returns:
            TrajectoryPairDataset 实例。
        """
        if self.cache_dir:
            cache_path = self._cache_path(positive_actions, positive_states)
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    return pickle.load(f)

        dataset = self.construct_dataset(positive_actions, positive_states)

        if self.cache_dir:
            cache_path = self._cache_path(positive_actions, positive_states)
            with open(cache_path, "wb") as f:
                pickle.dump(dataset, f)

        return dataset
