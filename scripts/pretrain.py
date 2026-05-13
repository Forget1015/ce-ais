#!/usr/bin/env python3
"""CE-AIS 预训练启动脚本。

解析命令行参数，加载配置，启动预训练管线。

Usage:
    python scripts/pretrain.py --config configs/base.yaml
    python scripts/pretrain.py --config configs/base.yaml --override training.batch_size=32
    python scripts/pretrain.py --config configs/base.yaml --stage encoder
    python scripts/pretrain.py --config configs/base.yaml --stage cewm --resume

    # 多卡并行
    torchrun --nproc_per_node=2 scripts/pretrain.py --config configs/base.yaml
"""

import argparse
import os
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config.config_manager import ConfigManager, parse_overrides
from src.config.schema import CEAISConfig, EncoderConfig, CEWMConfig, TrainingConfig
from accelerate import Accelerator


def build_config_from_dict(config_dict: dict) -> CEAISConfig:
    """从配置字典构建 CEAISConfig 数据类。"""
    encoder_cfg = EncoderConfig(
        **{
            k: v
            for k, v in config_dict.get("encoder", {}).items()
            if k in EncoderConfig.__dataclass_fields__
        }
    )
    cewm_cfg = CEWMConfig(
        **{
            k: v
            for k, v in config_dict.get("ce_wm", {}).items()
            if k in CEWMConfig.__dataclass_fields__
        }
    )
    training_cfg = TrainingConfig(
        **{
            k: v
            for k, v in config_dict.get("training", {}).items()
            if k in TrainingConfig.__dataclass_fields__
        }
    )

    config = CEAISConfig()
    config.encoder = encoder_cfg
    config.ce_wm = cewm_cfg
    config.training = training_cfg
    return config


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="CE-AIS Pretraining Script"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["encoder", "cewm", "both"],
        default="both",
        help="Pretraining stage: encoder, cewm, or both (default: both)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Log output directory",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Checkpoint save directory",
    )

    parser.add_argument(
        "--encoder-ckpt",
        type=str,
        default=None,
        help="Path to encoder checkpoint to load before CE-WM training. "
             "Required when --stage cewm is run alone (encoder weights must "
             "be initialized). If not specified, will auto-discover the latest "
             "encoder_epoch*.pt in --checkpoint-dir.",
    )

    # 收集所有 --override 参数
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        default=[],
        help="Override config values: key.subkey=value",
    )

    args = parser.parse_args()
    return args


def main() -> None:
    """主入口函数。"""
    args = parse_args()

    # 加载配置
    overrides = parse_overrides(args.override) if args.override else None
    cm = ConfigManager(config_path=args.config, overrides=overrides)

    # 保存配置快照
    cm.save_snapshot(args.log_dir)

    # 构建配置对象
    config = build_config_from_dict(cm.config)

    # 创建 Accelerator（多卡 DDP + 混合精度）
    accelerator = Accelerator(
        mixed_precision="fp16" if config.training.amp else "no",
    )

    if accelerator.is_main_process:
        print(f"[CE-AIS] Config loaded from: {args.config}")
        print(f"[CE-AIS] Stage: {args.stage}")
        print(f"[CE-AIS] Resume: {args.resume}")
        print(f"[CE-AIS] Device: {accelerator.device}")
        print(f"[CE-AIS] Num processes: {accelerator.num_processes}")

    # 延迟导入以避免无 GPU 时的 CUDA 错误
    import torch
    from src.encoders.contrastive_encoder import ContrastiveEncoder
    from src.world_model.ce_wm import CausalEnergyWorldModel
    from src.training.pretrain_pipeline import PretrainPipeline

    torch.backends.cudnn.benchmark = True

    # 实例化模型
    encoder = ContrastiveEncoder(config.encoder)
    ce_wm = CausalEnergyWorldModel(config.ce_wm)

    # 创建预训练管线
    pipeline = PretrainPipeline(
        config=config,
        encoder=encoder,
        ce_wm=ce_wm,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        accelerator=accelerator,
    )

    # 当只跑 CE-WM 阶段时，必须先把 encoder 权重加载进来（否则 z 全是噪声）
    if args.stage == "cewm":
        from src.utils.checkpoint import CheckpointManager

        ckpt_path = args.encoder_ckpt
        if ckpt_path is None:
            # 自动发现 checkpoint_dir 下最新的 encoder_epoch*.pt
            mgr = CheckpointManager(
                checkpoint_dir=args.checkpoint_dir,
                prefix="encoder",
            )
            ckpt_path = mgr.find_latest()

        if ckpt_path is None:
            raise FileNotFoundError(
                f"--stage cewm requires a trained encoder, but no "
                f"encoder_epoch*.pt found in {args.checkpoint_dir}/ "
                f"and --encoder-ckpt was not provided. "
                f"Either run --stage both first, or pass --encoder-ckpt PATH."
            )

        if accelerator.is_main_process:
            print(f"[CE-AIS] Loading encoder weights from: {ckpt_path}")
        loader = CheckpointManager(
            checkpoint_dir=os.path.dirname(ckpt_path) or ".",
            prefix="encoder",
        )
        state = loader.load(
            filepath=ckpt_path,
            model=encoder,
            map_location=accelerator.device,
        )
        if accelerator.is_main_process:
            print(
                f"[CE-AIS] Encoder loaded (trained {state.get('epoch', 0)} epochs, "
                f"step {state.get('global_step', 0)})"
            )

    if accelerator.is_main_process:
        print("[CE-AIS] Pipeline initialized. Starting training...")

    # 创建数据加载器（优先使用真实 CALVIN 数据，否则用合成数据）
    from torch.utils.data import DataLoader

    data_cfg = cm.config.get("data", {})
    data_dir = data_cfg.get("calvin_dir", "data/calvin")
    num_workers = int(data_cfg.get("num_workers", 0))
    encoder_bs = config.training.encoder_batch_size or config.training.batch_size
    cewm_bs = config.training.cewm_batch_size or config.training.batch_size

    # 检查是否有真实 CALVIN 数据
    calvin_path = Path(data_dir)
    has_calvin_data = calvin_path.exists() and any(calvin_path.iterdir())

    encoder_dataset = None
    cewm_dataset = None

    if has_calvin_data:
        if accelerator.is_main_process:
            print(f"[CE-AIS] Found CALVIN data at {data_dir}")
        from src.data.calvin_dataset import CALVINDataset
        try:
            if args.stage in ("encoder", "both"):
                encoder_dataset = CALVINDataset(
                    data_dir=str(calvin_path),
                    split="training",
                    config=cm.config,
                    mode="encoder",
                )
                if accelerator.is_main_process:
                    backend = "mmap" if encoder_dataset._use_mmap else "npz"
                    backend_path = (
                        encoder_dataset._mmap_dir if encoder_dataset._use_mmap
                        else encoder_dataset.split_dir
                    )
                    print(
                        f"[CE-AIS] Encoder dataset: {len(encoder_dataset)} samples "
                        f"({len(encoder_dataset.episodes)} episodes), "
                        f"backend={backend}, path={backend_path}"
                    )
            if args.stage in ("cewm", "both"):
                cewm_dataset = CALVINDataset(
                    data_dir=str(calvin_path),
                    split="training",
                    config=cm.config,
                    mode="ce_wm",
                )
                if accelerator.is_main_process:
                    backend = "mmap" if cewm_dataset._use_mmap else "npz"
                    backend_path = (
                        cewm_dataset._mmap_dir if cewm_dataset._use_mmap
                        else cewm_dataset.split_dir
                    )
                    print(
                        f"[CE-AIS] CE-WM dataset: {len(cewm_dataset)} windows "
                        f"(window_size={cewm_dataset.window_size}), "
                        f"backend={backend}, path={backend_path}"
                    )
        except Exception as e:
            if accelerator.is_main_process:
                print(f"[CE-AIS] Failed to load CALVIN data: {e}")
                print("[CE-AIS] Falling back to synthetic data")
            encoder_dataset = None
            cewm_dataset = None
    else:
        if accelerator.is_main_process:
            print("[CE-AIS] No CALVIN data found. Using synthetic data for validation.")

    # 合成数据集回退（用于 smoke test，不需要 GPU 数据）
    if encoder_dataset is None or cewm_dataset is None:
        n_samples = max(encoder_bs * 10, 64)
        H, W = config.encoder.image_size
        d_p = config.encoder.pose_dim
        d_a = config.ce_wm.action_dim
        d_z = config.encoder.latent_dim
        K = config.training.neg_sample_ratio

        class SyntheticEncoderDataset(torch.utils.data.Dataset):
            def __init__(self, size):
                self.size = size
            def __len__(self):
                return self.size
            def __getitem__(self, idx):
                return {
                    "rgb": torch.randn(3, H, W),
                    "depth": torch.randn(1, H, W),
                    "pose": torch.randn(d_p),
                }

        class SyntheticCEWMDataset(torch.utils.data.Dataset):
            def __init__(self, size):
                self.size = size
            def __len__(self):
                return self.size
            def __getitem__(self, idx):
                return {
                    "z_pos": torch.randn(d_z),
                    "a_pos": torch.randn(d_a),
                    "a_neg": torch.randn(K, d_a),
                }

        if encoder_dataset is None:
            encoder_dataset = SyntheticEncoderDataset(n_samples)
        if cewm_dataset is None:
            cewm_dataset = SyntheticCEWMDataset(n_samples)

    use_persistent = num_workers > 0
    prefetch = 2 if num_workers > 0 else None

    encoder_loader = DataLoader(
        encoder_dataset,
        batch_size=encoder_bs,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=use_persistent,
        prefetch_factor=prefetch,
    )
    cewm_loader = DataLoader(
        cewm_dataset,
        batch_size=cewm_bs,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=use_persistent,
        prefetch_factor=prefetch,
    )

    # 执行预训练
    if args.stage in ("encoder", "both"):
        if accelerator.is_main_process:
            print(f"\n[CE-AIS] === Stage 1: Encoder Pretraining ({config.training.encoder_epochs} epochs, batch_size={encoder_bs}) ===")
        enc_result = pipeline.pretrain_encoder(encoder_loader, resume=args.resume)
        if accelerator.is_main_process:
            print(f"[CE-AIS] Encoder final loss: {enc_result['final_loss']:.6f}")
            print(f"[CE-AIS] Loss history: {[f'{l:.4f}' for l in enc_result['losses_history']]}")

    if args.stage in ("cewm", "both"):
        if accelerator.is_main_process:
            print(f"\n[CE-AIS] === Stage 2: CE-WM Pretraining ({config.training.ce_wm_epochs} epochs, batch_size={cewm_bs}) ===")
        cewm_result = pipeline.pretrain_cewm(cewm_loader, resume=args.resume)
        if accelerator.is_main_process:
            print(f"[CE-AIS] CE-WM final loss: {cewm_result['final_loss']:.6f}")
            print(f"[CE-AIS] Loss history: {[f'{l:.4f}' for l in cewm_result['losses_history']]}")
            if "energy_report" in cewm_result:
                report = cewm_result["energy_report"]
                print(f"[CE-AIS] Energy Report:")
                print(f"  Positive mean energy: {report.get('pos_energy_mean', 'N/A')}")
                print(f"  Negative mean energy: {report.get('neg_energy_mean', 'N/A')}")
                print(f"  Energy margin: {report.get('energy_margin', 'N/A')}")
            if cewm_result.get("energy_history"):
                print(f"[CE-AIS] Energy margin per epoch:")
                for entry in cewm_result["energy_history"]:
                    print(
                        f"  epoch {entry['epoch']:3d}: "
                        f"pos={entry['pos_mean']:+.4f}  "
                        f"neg={entry['neg_mean']:+.4f}  "
                        f"margin={entry['margin']:+.4f}"
                    )

    if accelerator.is_main_process:
        print("\n[CE-AIS] Pretraining complete!")
        print(f"[CE-AIS] Checkpoints saved to: {args.checkpoint_dir}/")
        print(f"[CE-AIS] Logs saved to: {args.log_dir}/")


if __name__ == "__main__":
    main()
