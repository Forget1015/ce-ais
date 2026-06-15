#!/usr/bin/env python3
"""方案二：基于 State 近邻构造 Contrastive Pairs。

在相同 state 附近对比"导致成功的 action"和"导致失败的 action"，
让 energy head 学到"在给定 state 下 action 的好坏"。

用法:
  python scripts/build_state_contrastive_pairs.py \
    --rollout-dir data/rollout_flower_seed42 \
    --output data/rollout_flower_seed42/contrastive_pairs.npz \
    --top-k 64 --sim-threshold 0.85
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-dir", type=str, default="data/rollout_flower_seed42")
    parser.add_argument("--output", type=str, default="data/rollout_flower_seed42/contrastive_pairs.npz")
    parser.add_argument("--top-k", type=int, default=64, help="Number of nearest neighbors to search")
    parser.add_argument("--sim-threshold", type=float, default=0.85, help="Cosine similarity threshold")
    parser.add_argument("--max-pairs-per-anchor", type=int, default=5, help="Max neg samples per anchor")
    parser.add_argument("--neg-tail-ratio", type=float, default=0.5,
                        help="Only use last X fraction of failed episodes as neg source")
    return parser.parse_args()


def main():
    args = parse_args()

    print("[1/4] Loading rollout data...")
    latent = np.load(os.path.join(args.rollout_dir, "latent.npy"))  # [N, 128]
    action = np.load(os.path.join(args.rollout_dir, "action.npy"))  # [N, 7]
    success = np.load(os.path.join(args.rollout_dir, "success.npy"))  # [N]
    episode_info = np.load(os.path.join(args.rollout_dir, "episode_info.npy"))  # [M, 4]

    N = len(latent)
    print(f"  Total steps: {N}, pos: {success.sum()}, neg: {N - success.sum()}")

    # 标记每个 step 属于哪个 episode
    ep_id = np.zeros(N, dtype=np.int32)
    for idx, (start, end, chain, task) in enumerate(episode_info):
        ep_id[start:end] = idx

    # 标记失败 episode 中"后半段"的步骤（更可能是真正错误的 action）
    is_neg_tail = np.zeros(N, dtype=bool)
    for start, end, chain, task in episode_info:
        if success[start] == 0:  # failed episode
            ep_len = end - start
            tail_start = start + int(ep_len * (1 - args.neg_tail_ratio))
            is_neg_tail[tail_start:end] = True

    neg_tail_count = is_neg_tail.sum()
    print(f"  Neg tail steps (last {args.neg_tail_ratio*100:.0f}% of failed eps): {neg_tail_count}")

    # 正样本 index 和负样本 index
    pos_mask = success == 1
    neg_mask = is_neg_tail
    pos_indices = np.where(pos_mask)[0]
    neg_indices = np.where(neg_mask)[0]
    print(f"  Pos indices: {len(pos_indices)}, Neg indices: {len(neg_indices)}")

    # === 2. Build FAISS index on neg latents ===
    print("[2/4] Building FAISS index on negative latents...")
    import faiss

    # Normalize latents for cosine similarity
    latent_norm = latent / (np.linalg.norm(latent, axis=1, keepdims=True) + 1e-8)
    latent_norm = latent_norm.astype(np.float32)

    # Build index on neg steps (we want to find neg neighbors for pos anchors)
    neg_latents = np.ascontiguousarray(latent_norm[neg_indices])
    d = neg_latents.shape[1]

    # Use flat index for accuracy (neg set is only ~40K)
    index_neg = faiss.IndexFlatIP(d)  # inner product = cosine sim (normalized)
    index_neg.add(neg_latents)
    print(f"  FAISS index built: {index_neg.ntotal} vectors")

    # === 3. For each pos anchor, find similar neg neighbors ===
    print("[3/4] Finding contrastive pairs...")
    t0 = time.time()

    # Search in batches
    batch_size = 10000
    all_anchors_z = []
    all_pos_a = []
    all_neg_a = []

    n_searched = 0
    n_pairs = 0

    for batch_start in range(0, len(pos_indices), batch_size):
        batch_end = min(batch_start + batch_size, len(pos_indices))
        batch_pos_idx = pos_indices[batch_start:batch_end]

        queries = np.ascontiguousarray(latent_norm[batch_pos_idx])
        sims, nn_ids = index_neg.search(queries, args.top_k)  # [batch, top_k]

        for i in range(len(batch_pos_idx)):
            anchor_idx = batch_pos_idx[i]
            anchor_ep = ep_id[anchor_idx]

            # Filter by similarity threshold and different episode
            valid = []
            for j in range(args.top_k):
                sim = sims[i, j]
                if sim < args.sim_threshold:
                    break
                neg_real_idx = neg_indices[nn_ids[i, j]]
                if ep_id[neg_real_idx] == anchor_ep:
                    continue
                valid.append(neg_real_idx)
                if len(valid) >= args.max_pairs_per_anchor:
                    break

            # Create pairs
            for neg_idx in valid:
                all_anchors_z.append(latent[anchor_idx])
                all_pos_a.append(action[anchor_idx])
                all_neg_a.append(action[neg_idx])
                n_pairs += 1

        n_searched += len(batch_pos_idx)
        if n_searched % 50000 == 0 or batch_end == len(pos_indices):
            elapsed = time.time() - t0
            print(f"  Searched {n_searched}/{len(pos_indices)} anchors, pairs: {n_pairs}, {elapsed:.1f}s")

    # === 4. Save ===
    print(f"[4/4] Saving {n_pairs} contrastive pairs...")

    anchors_z = np.array(all_anchors_z, dtype=np.float32)  # [P, 128]
    pos_a = np.array(all_pos_a, dtype=np.float32)          # [P, 7]
    neg_a = np.array(all_neg_a, dtype=np.float32)          # [P, 7]

    np.savez_compressed(
        args.output,
        anchor_z=anchors_z,
        pos_action=pos_a,
        neg_action=neg_a,
    )

    file_size = os.path.getsize(args.output) / 1024 / 1024
    print(f"\nDone!")
    print(f"  Total pairs: {n_pairs}")
    print(f"  Output: {args.output} ({file_size:.1f} MB)")
    print(f"  anchor_z: {anchors_z.shape}")
    print(f"  pos_action: {pos_a.shape}")
    print(f"  neg_action: {neg_a.shape}")


if __name__ == "__main__":
    main()
