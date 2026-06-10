"""LIBERO 专家演示数据集加载与解析。

从 LIBERO benchmark 的 HDF5 demo 文件中加载数据，供 CE-AIS 的 encoder 和 CE-WM 训练。

支持两种采样模式:
- "encoder": 返回 anchor + 时序近邻 positive 帧对，供 InfoNCE 训练
- "ce_wm":   返回连续 T 帧窗口，供 CE-WM 训练

支持多 suite 联合训练:
- 单 suite: suite_names="libero_spatial"
- 多 suite: suite_names="libero_spatial,libero_object,libero_goal,libero_10"
- LIBERO-90: suite_names="libero_90"

字段映射（LIBERO → CE-AIS）:
    agentview_rgb (128,128,3) uint8   → rgb   [3, H, W] float32, ImageNet 归一化
    eye_in_hand_rgb (128,128,3) uint8 → rgb_gripper [3, H, W] float32 (可选)
    joint_states (7,) float64         → pose[:7]  关节角度
    gripper_states (2,) float64       → pose[7:9] 夹爪状态
    actions (7,) float64              → action [7] 动作
"""

import glob
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class LIBERODataset(Dataset):
    """LIBERO 专家演示数据集。

    Args:
        data_dir:      LIBERO 数据根目录（包含 libero_spatial/ 等子目录）。
        suite_names:   逗号分隔的 suite 名称，或单个 suite。
                       例: "libero_spatial" 或 "libero_spatial,libero_object,libero_goal,libero_10"
        config:        完整配置字典（从 ConfigManager.config 读取）。
        mode:          "encoder" | "ce_wm"。
        max_demos:     每个 task 最大加载 demo 数（用于调试）。
        image_size:    输出图像尺寸 (H, W)。
    """

    def __init__(
        self,
        data_dir: str,
        suite_names: str = "libero_spatial",
        config: Optional[Dict[str, Any]] = None,
        mode: str = "encoder",
        max_demos: Optional[int] = None,
        image_size: Tuple[int, int] = (128, 128),
    ):
        if mode not in ("encoder", "ce_wm"):
            raise ValueError(f"mode must be 'encoder' or 'ce_wm', got {mode}")

        self.data_dir = data_dir
        self.mode = mode
        self.config = config or {}
        self.image_size = image_size

        encoder_cfg = self.config.get("encoder", {})
        ce_wm_cfg = self.config.get("ce_wm", {})
        data_cfg = self.config.get("data", {})

        self.pose_dim: int = int(encoder_cfg.get("pose_dim", 9))
        self.action_dim: int = int(ce_wm_cfg.get("action_dim", 7))
        self.window_size: int = int(data_cfg.get("window_size", 16))
        self.encoder_pos_offset_max: int = int(
            data_cfg.get("encoder_pos_offset_max", 4)
        )

        suites = [s.strip() for s in suite_names.split(",")]
        self.trajectories: List[Dict[str, np.ndarray]] = []
        self._load_all_suites(suites, max_demos)

        self._index: List[Tuple[int, int]] = self._build_sample_index()

        if len(self._index) == 0:
            raise RuntimeError(
                f"No valid samples found. data_dir={data_dir}, "
                f"suites={suites}, mode={mode}, "
                f"trajectories={len(self.trajectories)}"
            )

    def _load_all_suites(
        self, suites: List[str], max_demos: Optional[int]
    ) -> None:
        """加载所有指定 suite 的 demo 轨迹到内存。"""
        for suite in suites:
            suite_dir = os.path.join(self.data_dir, suite)
            if not os.path.isdir(suite_dir):
                raise FileNotFoundError(f"Suite directory not found: {suite_dir}")

            hdf5_files = sorted(glob.glob(os.path.join(suite_dir, "*.hdf5")))
            if not hdf5_files:
                raise FileNotFoundError(f"No HDF5 files found in {suite_dir}")

            for fpath in hdf5_files:
                self._load_hdf5(fpath, max_demos)

    def _load_hdf5(self, fpath: str, max_demos: Optional[int]) -> None:
        """从单个 HDF5 文件加载 demo 轨迹。"""
        with h5py.File(fpath, "r") as f:
            demo_keys = sorted(
                f["data"].keys(),
                key=lambda x: int(x.replace("demo_", "")),
            )
            if max_demos is not None:
                demo_keys = demo_keys[:max_demos]

            for dk in demo_keys:
                demo = f[f"data/{dk}"]
                actions = demo["actions"][:].astype(np.float32)
                obs = demo["obs"]

                agentview_rgb = obs["agentview_rgb"][:]
                joint_states = obs["joint_states"][:].astype(np.float32)
                gripper_states = obs["gripper_states"][:].astype(np.float32)

                # pose = joint_states (7) + gripper_states (2) = 9 dim
                pose = np.concatenate([joint_states, gripper_states], axis=-1)

                traj = {
                    "rgb": agentview_rgb,  # (T, 128, 128, 3) uint8
                    "pose": pose,  # (T, 9) float32
                    "actions": actions,  # (T, 7) float32
                }

                # eye_in_hand 可选加载（如果 encoder 需要双视角）
                if "eye_in_hand_rgb" in obs:
                    traj["rgb_gripper"] = obs["eye_in_hand_rgb"][:]

                self.trajectories.append(traj)

    def _build_sample_index(self) -> List[Tuple[int, int]]:
        """构建采样索引：(traj_idx, frame_offset)。"""
        index: List[Tuple[int, int]] = []
        for traj_idx, traj in enumerate(self.trajectories):
            n = traj["actions"].shape[0]
            if self.mode == "encoder":
                for t in range(n - 1):
                    index.append((traj_idx, t))
            else:  # ce_wm
                for t in range(max(0, n - self.window_size + 1)):
                    index.append((traj_idx, t))
        return index

    def _process_rgb(self, rgb_frame: np.ndarray) -> torch.Tensor:
        """(H, W, 3) uint8 → [3, H', W'] float32, ImageNet 归一化。"""
        rgb = torch.from_numpy(rgb_frame).permute(2, 0, 1).float() / 255.0
        if (rgb.shape[1], rgb.shape[2]) != self.image_size:
            rgb = F.interpolate(
                rgb.unsqueeze(0),
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
        return rgb

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        traj_idx, t = self._index[idx]
        traj = self.trajectories[traj_idx]

        if self.mode == "encoder":
            n = traj["actions"].shape[0]
            max_offset = min(self.encoder_pos_offset_max, n - 1 - t)
            offset = random.randint(1, max(max_offset, 1))
            t_pos = t + offset

            rgb_anc = self._process_rgb(traj["rgb"][t])
            rgb_pos = self._process_rgb(traj["rgb"][t_pos])
            pose_anc = torch.from_numpy(traj["pose"][t][: self.pose_dim])
            pose_pos = torch.from_numpy(traj["pose"][t_pos][: self.pose_dim])

            return {
                "rgb": rgb_anc,
                "pose": pose_anc,
                "rgb_pos": rgb_pos,
                "pose_pos": pose_pos,
            }

        # ce_wm 模式：连续 T 帧窗口
        window_end = t + self.window_size
        rgb_seq = torch.stack(
            [self._process_rgb(traj["rgb"][i]) for i in range(t, window_end)],
            dim=0,
        )
        pose_seq = torch.from_numpy(
            traj["pose"][t:window_end, : self.pose_dim].copy()
        )
        a_pos = torch.from_numpy(
            traj["actions"][t:window_end, : self.action_dim].copy()
        )

        return {
            "rgb_seq": rgb_seq,  # (T, 3, H, W)
            "pose_seq": pose_seq,  # (T, pose_dim)
            "a_pos": a_pos,  # (T, action_dim)
        }

    def get_stats(self) -> Dict[str, Any]:
        """返回数据集统计信息。"""
        total_frames = sum(t["actions"].shape[0] for t in self.trajectories)
        lengths = [t["actions"].shape[0] for t in self.trajectories]
        return {
            "n_trajectories": len(self.trajectories),
            "total_frames": total_frames,
            "n_samples": len(self._index),
            "avg_traj_length": total_frames / max(len(self.trajectories), 1),
            "min_traj_length": min(lengths) if lengths else 0,
            "max_traj_length": max(lengths) if lengths else 0,
            "mode": self.mode,
            "window_size": self.window_size if self.mode == "ce_wm" else None,
        }
