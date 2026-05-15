"""CALVIN 专家演示数据集加载与解析。

支持两种数据后端:
1. 原生 .npz 文件（每帧一个文件，兼容原始 CALVIN 格式）
2. 内存映射 .npy 文件（由 scripts/convert_calvin_mmap.py 生成，读取速度快 10-50 倍）

当 data.mmap_dir 配置项指向有效的 mmap 目录时自动切换后端。

字段映射（CALVIN → CE-AIS Observation/Action）:
    rgb_static (200,200,3) uint8   → rgb   [3, H, W] float32, ImageNet 归一化
    depth_static (200,200) float32 → depth [1, H, W] float32, 截断到 [0, depth_max] 后归一化
    robot_obs[:7] (7,) float64     → pose  [7]   xyz(3) + rpy(3) + gripper_width(1)
    rel_actions (7,) float64       → action [7]  dx,dy,dz,droll,dpitch,dyaw,gripper

支持两种采样模式:
- "encoder": 返回 anchor + 时序近邻 positive 帧对，供 InfoNCE 训练
- "ce_wm":   返回连续 T 帧窗口，供 CE-WM 训练（trainer 会用冻结 encoder 在线编码 z）
"""

import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


# ImageNet 标准化常量（与 ResNet/ViT 预训练分布对齐）
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# CALVIN 深度图的经验上界（米）。超出截断后归一化到 [0, 1]。
_DEPTH_MAX = 4.0


class CALVINDataset(Dataset):
    """CALVIN 专家演示数据集。

    Args:
        data_dir:    CALVIN 数据根目录，包含 training/ 和 validation/ 子目录。
        split:       "training" 或 "validation"。
        config:      完整配置字典（来自 ConfigManager.config）。从中读取
                     encoder.image_size、encoder.pose_dim、ce_wm.action_dim、
                     data.* 等字段。
        mode:        "encoder" | "ce_wm"。决定 __getitem__ 返回结构。
        max_episodes: 最大加载 episode 数（用于调试时限制规模）。
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "training",
        config: Optional[Dict[str, Any]] = None,
        mode: str = "encoder",
        max_episodes: Optional[int] = None,
    ):
        if mode not in ("encoder", "ce_wm"):
            raise ValueError(f"mode must be 'encoder' or 'ce_wm', got {mode}")

        self.data_dir = data_dir
        self.split = split
        self.mode = mode
        self.config = config or {}

        # 从 config 读取参数（带默认值）
        encoder_cfg = self.config.get("encoder", {})
        ce_wm_cfg = self.config.get("ce_wm", {})
        data_cfg = self.config.get("data", {})

        self.image_size: Tuple[int, int] = tuple(
            encoder_cfg.get("image_size", [200, 200])
        )
        self.pose_dim: int = int(encoder_cfg.get("pose_dim", 7))
        self.action_dim: int = int(ce_wm_cfg.get("action_dim", 7))

        self.window_size: int = int(data_cfg.get("window_size", 16))
        self.encoder_pos_offset_max: int = int(
            data_cfg.get("encoder_pos_offset_max", 4)
        )
        max_eps_from_cfg = data_cfg.get("max_episodes")
        if max_episodes is None and max_eps_from_cfg is not None:
            max_episodes = int(max_eps_from_cfg)
        self.max_episodes = max_episodes

        # 检查是否有 mmap 后端
        mmap_dir = data_cfg.get("mmap_dir")
        self._use_mmap = False
        if mmap_dir:
            mmap_split_dir = os.path.join(mmap_dir, split)
            rgb_path = os.path.join(mmap_split_dir, "rgb.npy")
            if os.path.exists(rgb_path):
                self._use_mmap = True
                self._mmap_dir = mmap_split_dir

        # 加载 episode 索引
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"CALVIN split directory not found: {split_dir}"
            )
        self.split_dir = split_dir
        self.episodes: List[List[int]] = self._build_episode_index()

        # 构建采样索引（每个元素 = (ep_idx, frame_offset_within_ep)）
        self._index: List[Tuple[int, int]] = self._build_sample_index()

        if len(self._index) == 0:
            raise RuntimeError(
                f"No valid samples found in {split_dir}. "
                f"mode={mode}, window_size={self.window_size}, "
                f"episodes={len(self.episodes)}"
            )

        # 初始化 mmap 后端
        if self._use_mmap:
            self._init_mmap()

    # ---- mmap 后端 ----

    def _init_mmap(self) -> None:
        """初始化内存映射数组和帧 ID → 数组索引映射。"""
        d = self._mmap_dir
        frame_ids = np.load(os.path.join(d, "frame_ids.npy"))
        self._fid_to_idx = {int(fid): i for i, fid in enumerate(frame_ids)}
        self._rgb_mmap = np.load(os.path.join(d, "rgb.npy"), mmap_mode="r")
        self._depth_mmap = np.load(os.path.join(d, "depth.npy"), mmap_mode="r")
        self._pose_mmap = np.load(os.path.join(d, "pose.npy"), mmap_mode="r")
        self._action_mmap = np.load(os.path.join(d, "action.npy"), mmap_mode="r")

    def _load_frame_mmap(self, frame_id: int) -> Dict[str, torch.Tensor]:
        """从 mmap 加载单帧（零拷贝读取 + 就地预处理）。"""
        idx = self._fid_to_idx[frame_id]

        rgb = torch.from_numpy(self._rgb_mmap[idx].copy()).permute(2, 0, 1).float() / 255.0
        if (rgb.shape[1], rgb.shape[2]) != self.image_size:
            rgb = F.interpolate(
                rgb.unsqueeze(0), size=self.image_size,
                mode="bilinear", align_corners=False,
            ).squeeze(0)
        rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD

        depth = torch.from_numpy(self._depth_mmap[idx].copy()).unsqueeze(0).float()
        depth = depth.clamp(0.0, _DEPTH_MAX) / _DEPTH_MAX
        if (depth.shape[1], depth.shape[2]) != self.image_size:
            depth = F.interpolate(
                depth.unsqueeze(0), size=self.image_size,
                mode="bilinear", align_corners=False,
            ).squeeze(0)

        pose = torch.from_numpy(self._pose_mmap[idx].copy())
        action = torch.from_numpy(self._action_mmap[idx].copy())

        return {"rgb": rgb, "depth": depth, "pose": pose, "action": action}

    # ---- 索引构建 ----

    def _build_episode_index(self) -> List[List[int]]:
        """解析 ep_start_end_ids.npy，构建每个 episode 的帧 ID 列表。

        Returns:
            episodes: [[frame_id, ...], ...]，按 episode 分组的全局帧 ID 列表。
        """
        # 收集所有可用的 .npz 帧 ID
        all_frame_ids = set()
        for fname in os.listdir(self.split_dir):
            if fname.startswith("episode_") and fname.endswith(".npz"):
                try:
                    fid = int(
                        fname.removeprefix("episode_").removesuffix(".npz")
                    )
                    all_frame_ids.add(fid)
                except ValueError:
                    continue

        # 加载 episode 边界
        ep_ids_path = os.path.join(self.split_dir, "ep_start_end_ids.npy")
        if not os.path.exists(ep_ids_path):
            # 退化路径：把所有帧当作一个 episode
            sorted_frames = sorted(all_frame_ids)
            return [sorted_frames] if sorted_frames else []

        ep_start_end = np.load(ep_ids_path)  # [N_ep, 2]

        episodes: List[List[int]] = []
        min_required = (
            self.window_size if self.mode == "ce_wm" else 2
        )  # encoder 模式至少要 2 帧才能取到 anchor+positive
        for start, end in ep_start_end:
            ep_frames = sorted(
                fid for fid in all_frame_ids if start <= fid <= end
            )
            if len(ep_frames) >= min_required:
                episodes.append(ep_frames)

        if self.max_episodes is not None:
            episodes = episodes[: self.max_episodes]
        return episodes

    def _build_sample_index(self) -> List[Tuple[int, int]]:
        """为当前模式构建 __getitem__ 索引。"""
        index: List[Tuple[int, int]] = []
        for ep_idx, ep_frames in enumerate(self.episodes):
            n = len(ep_frames)
            if self.mode == "encoder":
                # 每帧都可作为 anchor（除了最后一帧，因为没有 positive 候选）
                for t in range(n - 1):
                    index.append((ep_idx, t))
            else:  # ce_wm
                # 每个能容纳完整窗口的起点
                for t in range(n - self.window_size + 1):
                    index.append((ep_idx, t))
        return index

    # ---- 帧加载 ----

    def _frame_path(self, frame_id: int) -> str:
        return os.path.join(
            self.split_dir, f"episode_{frame_id:07d}.npz"
        )

    def _load_frame(self, frame_id: int) -> Dict[str, torch.Tensor]:
        """加载单帧并完成预处理。自动选择 mmap 或 npz 后端。"""
        if self._use_mmap:
            return self._load_frame_mmap(frame_id)
        return self._load_frame_npz(frame_id)

    def _load_frame_npz(self, frame_id: int) -> Dict[str, torch.Tensor]:
        """加载单帧并完成预处理。"""
        data = np.load(self._frame_path(frame_id))

        # RGB: (200, 200, 3) uint8 → [3, H, W] float32, ImageNet 归一化
        rgb = torch.from_numpy(data["rgb_static"]).permute(2, 0, 1).float() / 255.0
        if (rgb.shape[1], rgb.shape[2]) != self.image_size:
            rgb = F.interpolate(
                rgb.unsqueeze(0),
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD

        # Depth: (200, 200) float32 → [1, H, W]，截断后归一化到 [0,1]
        depth = torch.from_numpy(data["depth_static"]).unsqueeze(0).float()
        depth = depth.clamp(0.0, _DEPTH_MAX) / _DEPTH_MAX
        if (depth.shape[1], depth.shape[2]) != self.image_size:
            depth = F.interpolate(
                depth.unsqueeze(0),
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        # Pose: robot_obs[:pose_dim] = xyz(3) + rpy(3) + gripper_width(1)
        pose = torch.from_numpy(
            data["robot_obs"][: self.pose_dim].astype(np.float32)
        )

        # Action: rel_actions [action_dim] = (dx,dy,dz,droll,dpitch,dyaw,gripper)
        action = torch.from_numpy(
            data["rel_actions"][: self.action_dim].astype(np.float32)
        )

        return {"rgb": rgb, "depth": depth, "pose": pose, "action": action}

    # ---- Dataset 接口 ----

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_idx, t = self._index[idx]
        ep_frames = self.episodes[ep_idx]

        if self.mode == "encoder":
            # anchor = 第 t 帧；positive = 同 episode 后续 1..k 帧之一
            max_offset = min(
                self.encoder_pos_offset_max, len(ep_frames) - 1 - t
            )
            offset = random.randint(1, max(max_offset, 1))
            t_pos = t + offset
            anc = self._load_frame(ep_frames[t])
            pos = self._load_frame(ep_frames[t_pos])
            return {
                "rgb": anc["rgb"],
                "depth": anc["depth"],
                "pose": anc["pose"],
                "rgb_pos": pos["rgb"],
                "depth_pos": pos["depth"],
                "pose_pos": pos["pose"],
            }

        # ce_wm 模式：返回连续 T 帧窗口
        window = ep_frames[t : t + self.window_size]
        frames = [self._load_frame(fid) for fid in window]
        return {
            "rgb_seq": torch.stack([f["rgb"] for f in frames], dim=0),
            "depth_seq": torch.stack([f["depth"] for f in frames], dim=0),
            "pose_seq": torch.stack([f["pose"] for f in frames], dim=0),
            "a_pos": torch.stack([f["action"] for f in frames], dim=0),
        }
