"""MVP Day 1: 环境与数据自检。

检查项:
1. Python >= 3.10
2. torch + CUDA 可用性
3. GPU 显存 >= 8GB
4. CALVIN debug 数据集存在且格式正确
5. 关键 src 模块可导入
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "calvin_debug_dataset")


def check_python():
    v = sys.version_info
    ok = v >= (3, 10)
    status = "OK" if ok else f"FAIL (need >=3.10)"
    print(f"  Python {v.major}.{v.minor}.{v.micro}: {status}")
    return ok


def check_torch():
    try:
        import torch

        print(f"  PyTorch {torch.__version__}: OK")
        if torch.cuda.is_available():
            dev = torch.cuda.get_device_name(0)
            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  CUDA device: {dev} ({mem_gb:.1f} GB)")
            if mem_gb < 8:
                print("  WARNING: GPU memory < 8GB, may OOM on large batches")
            return True
        else:
            print("  WARNING: CUDA not available, will use CPU (very slow)")
            return True
    except ImportError:
        print("  FAIL: torch not installed")
        return False


def check_data():
    if not os.path.isdir(DATA_DIR):
        print(f"  FAIL: dataset not found: {DATA_DIR}")
        return False

    ok = True
    for split in ("training", "validation"):
        split_dir = os.path.join(DATA_DIR, split)
        if not os.path.isdir(split_dir):
            print(f"  FAIL: {split} directory not found")
            ok = False
            continue
        npz_files = [f for f in os.listdir(split_dir) if f.endswith(".npz")]
        print(f"  {split}: {len(npz_files)} frames")
        if len(npz_files) == 0:
            print(f"  FAIL: no .npz files in {split_dir}")
            ok = False

    # Check single frame format
    import numpy as np

    train_dir = os.path.join(DATA_DIR, "training")
    sample = sorted(
        f for f in os.listdir(train_dir) if f.endswith(".npz")
    )[0]
    data = np.load(os.path.join(train_dir, sample))
    required_keys = ["rgb_static", "depth_static", "robot_obs", "rel_actions"]
    for k in required_keys:
        if k not in data:
            print(f"  FAIL: key '{k}' missing from {sample}")
            ok = False

    if ok:
        print(
            f"  Data format: OK "
            f"(rgb_static={data['rgb_static'].shape}, "
            f"rel_actions={data['rel_actions'].shape})"
        )
    return ok


def check_imports():
    ok = True
    modules = [
        "src.encoders.contrastive_encoder",
        "src.world_model.ce_wm",
        "src.steering.langevin",
        "src.data.calvin_dataset",
        "src.data.perturbation",
        "src.training.losses",
    ]
    for mod in modules:
        try:
            __import__(mod)
            print(f"  {mod}: OK")
        except Exception as e:
            print(f"  {mod}: FAIL ({e})")
            ok = False
    return ok


def main():
    print("=" * 60)
    print("CE-AIS MVP: Environment Check")
    print("=" * 60)

    results = {}

    print("\n[1/4] Python version")
    results["python"] = check_python()

    print("\n[2/4] PyTorch + CUDA")
    results["torch"] = check_torch()

    print("\n[3/4] CALVIN debug dataset")
    results["data"] = check_data()

    print("\n[4/4] Source module imports")
    results["imports"] = check_imports()

    print("\n" + "=" * 60)
    all_ok = all(results.values())
    if all_ok:
        print("ALL CHECKS PASSED. Ready for MVP.")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"FAILED: {', '.join(failed)}")
        print("Fix the above issues before proceeding.")
    print("=" * 60)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
