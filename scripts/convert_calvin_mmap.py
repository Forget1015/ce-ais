#!/usr/bin/env python3
"""将 CALVIN .npz 数据集转换为内存映射 .npy 格式（多进程并行）。

把数十万个小 .npz 压缩文件合并为 4 个大的 numpy memmap 文件，
消除 zip 解压和文件打开开销，训练吞吐提升一个数量级。

使用多进程并行解压，56 核 CPU 上约 15-20 分钟完成 179 万帧。

输出结构:
    <output_dir>/
        rgb.npy       # [N, 200, 200, 3] uint8
        depth.npy     # [N, 200, 200] float16
        pose.npy      # [N, 7] float32
        action.npy    # [N, 7] float32
        frame_ids.npy # [N] int64 — 第 i 行对应 episode_{frame_ids[i]:07d}.npz
        ep_start_end_ids.npy  # 直接拷贝

Usage:
    python scripts/convert_calvin_mmap.py \
        --src data/task_ABC_D/training \
        --dst /tmp/calvin_mmap/training
"""

import argparse
import multiprocessing as mp
import os
import shutil
import sys

import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Convert CALVIN npz → mmap npy")
    parser.add_argument("--src", required=True, help="Source CALVIN split dir")
    parser.add_argument("--dst", required=True, help="Output mmap dir")
    parser.add_argument("--image-size", type=int, nargs=2, default=[200, 200])
    parser.add_argument("--pose-dim", type=int, default=7)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel workers (default: auto, half of CPU cores)")
    parser.add_argument("--chunk-size", type=int, default=200,
                        help="Frames per chunk (smaller = smoother progress bar)")
    return parser.parse_args()


def collect_frame_ids(src_dir: str):
    ids = []
    for fname in os.listdir(src_dir):
        if fname.startswith("episode_") and fname.endswith(".npz"):
            try:
                fid = int(fname.removeprefix("episode_").removesuffix(".npz"))
                ids.append(fid)
            except ValueError:
                continue
    ids.sort()
    return ids


# ---- Worker 初始化与处理 ----

_worker_state = {}


def _worker_init(dst_dir, src_dir, H, W, pose_dim, action_dim, N):
    """每个 worker 进程打开自己的 mmap 句柄（r+ 模式）。"""
    _worker_state["src_dir"] = src_dir
    _worker_state["pose_dim"] = pose_dim
    _worker_state["action_dim"] = action_dim
    _worker_state["rgb"] = np.lib.format.open_memmap(
        os.path.join(dst_dir, "rgb.npy"), mode="r+",
        dtype=np.uint8, shape=(N, H, W, 3),
    )
    _worker_state["depth"] = np.lib.format.open_memmap(
        os.path.join(dst_dir, "depth.npy"), mode="r+",
        dtype=np.float16, shape=(N, H, W),
    )
    _worker_state["pose"] = np.lib.format.open_memmap(
        os.path.join(dst_dir, "pose.npy"), mode="r+",
        dtype=np.float32, shape=(N, pose_dim),
    )
    _worker_state["action"] = np.lib.format.open_memmap(
        os.path.join(dst_dir, "action.npy"), mode="r+",
        dtype=np.float32, shape=(N, action_dim),
    )


def _worker_process(task):
    """处理一批 (index, frame_id) 对。"""
    src_dir = _worker_state["src_dir"]
    pose_dim = _worker_state["pose_dim"]
    action_dim = _worker_state["action_dim"]
    rgb_mm = _worker_state["rgb"]
    depth_mm = _worker_state["depth"]
    pose_mm = _worker_state["pose"]
    action_mm = _worker_state["action"]

    count = 0
    for idx, fid in task:
        path = os.path.join(src_dir, f"episode_{fid:07d}.npz")
        data = np.load(path)
        rgb_mm[idx] = data["rgb_static"]
        depth_mm[idx] = data["depth_static"].astype(np.float16)
        pose_mm[idx] = data["robot_obs"][:pose_dim].astype(np.float32)
        action_mm[idx] = data["rel_actions"][:action_dim].astype(np.float32)
        count += 1

    rgb_mm.flush()
    depth_mm.flush()
    pose_mm.flush()
    action_mm.flush()
    return count


def main():
    args = parse_args()
    src_dir = args.src
    dst_dir = args.dst
    H, W = args.image_size
    pose_dim = args.pose_dim
    action_dim = args.action_dim
    n_workers = args.workers or min(mp.cpu_count() // 2, 24)

    if not os.path.isdir(src_dir):
        print(f"Error: source directory not found: {src_dir}")
        sys.exit(1)

    print(f"[convert] Scanning {src_dir} ...")
    frame_ids = collect_frame_ids(src_dir)
    N = len(frame_ids)
    print(f"[convert] Found {N:,} frames")
    print(f"[convert] Using {n_workers} workers")

    rgb_bytes = N * H * W * 3
    depth_bytes = N * H * W * 2
    total_gb = (rgb_bytes + depth_bytes + N * (pose_dim + action_dim) * 4) / 1e9
    print(f"[convert] Estimated output: {total_gb:.1f} GB")

    stat = shutil.disk_usage(os.path.dirname(dst_dir) or "/tmp")
    free_gb = stat.free / 1e9
    print(f"[convert] Destination free space: {free_gb:.1f} GB")
    if free_gb < total_gb * 1.05:
        print(f"Error: not enough space ({free_gb:.1f} GB < {total_gb * 1.05:.1f} GB needed)")
        sys.exit(1)

    os.makedirs(dst_dir, exist_ok=True)

    # 主进程创建 memmap 文件（w+ 模式，分配磁盘空间）
    print("[convert] Creating memmap files ...")
    for name, dtype, shape in [
        ("rgb.npy", np.uint8, (N, H, W, 3)),
        ("depth.npy", np.float16, (N, H, W)),
        ("pose.npy", np.float32, (N, pose_dim)),
        ("action.npy", np.float32, (N, action_dim)),
    ]:
        mm = np.lib.format.open_memmap(
            os.path.join(dst_dir, name), mode="w+", dtype=dtype, shape=shape,
        )
        mm.flush()
        del mm

    # 将帧分成 chunks，每个 chunk 交给一个 worker
    chunk_size = args.chunk_size
    tasks = []
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk = [(i, frame_ids[i]) for i in range(start, end)]
        tasks.append(chunk)

    print(f"[convert] Converting {N:,} frames in {len(tasks)} chunks ...")

    done = 0
    with mp.Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(dst_dir, src_dir, H, W, pose_dim, action_dim, N),
    ) as pool:
        with tqdm(total=N, desc="Converting", unit="frame") as pbar:
            for count in pool.imap_unordered(_worker_process, tasks):
                done += count
                pbar.update(count)

    # 保存帧 ID 索引
    np.save(os.path.join(dst_dir, "frame_ids.npy"),
            np.array(frame_ids, dtype=np.int64))

    # 拷贝 ep_start_end_ids.npy
    ep_path = os.path.join(src_dir, "ep_start_end_ids.npy")
    if os.path.exists(ep_path):
        shutil.copy2(ep_path, os.path.join(dst_dir, "ep_start_end_ids.npy"))

    print(f"\n[convert] Done! Output: {dst_dir}")
    print("[convert] Files:")
    for f in sorted(os.listdir(dst_dir)):
        size = os.path.getsize(os.path.join(dst_dir, f))
        print(f"  {f:25s} {size / 1e9:8.2f} GB")


if __name__ == "__main__":
    main()
