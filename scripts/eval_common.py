"""Shared CLI/config helpers for CALVIN evaluation scripts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Optional

import torch

from src.dual_stream.vla_adapter import SUPPORTED_VLA_TYPES


def add_common_eval_args(parser: argparse.ArgumentParser, project_root: str) -> None:
    parser.add_argument("--vla-type", type=str, default="proxy", choices=SUPPORTED_VLA_TYPES)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Evaluation device, e.g. cuda:0 or cpu. Defaults to project.device",
    )
    parser.add_argument("--no-egl", action="store_true", help="Disable PyBullet EGL rendering")
    parser.add_argument(
        "--encoder-ckpt",
        type=str,
        default=None,
        help="Encoder checkpoint path. Defaults to latest checkpoints/encoder_epoch*.pt",
    )
    parser.add_argument(
        "--cewm-ckpt",
        type=str,
        default=None,
        help="CE-WM checkpoint path. Defaults to latest checkpoints/cewm_epoch*.pt",
    )
    parser.add_argument(
        "--vla-model",
        type=str,
        default=None,
        help="OpenVLA model path or HuggingFace model id",
    )
    parser.add_argument("--vla-dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--load-in-8bit", action="store_true", help="Load OpenVLA in 8-bit")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load OpenVLA in 4-bit")
    parser.add_argument("--unnorm-key", type=str, default=None, help="OpenVLA action unnormalization key")
    parser.add_argument("--hf-token", type=str, default=None, help="HuggingFace token for gated models")
    parser.add_argument(
        "--calvin-policy-ckpt",
        type=str,
        default=None,
        help="CALVIN-native policy checkpoint for --vla-type calvin",
    )
    parser.add_argument(
        "--calvin-train-folder",
        type=str,
        default=None,
        help="CALVIN training folder containing .hydra/config.yaml",
    )
    parser.add_argument(
        "--calvin-dataset-path",
        type=str,
        default=None,
        help="Dataset path for official CALVIN policy loading. Defaults to --data-dir",
    )
    parser.add_argument(
        "--flower-checkpoint-dir",
        type=str,
        default=str(Path(project_root) / "data" / "flower_calvin_abc"),
        help="FLOWER VLA checkpoint directory containing config.yaml and model.safetensors",
    )
    parser.add_argument("--flower-code-path", type=str, default=None, help="External FLOWER VLA code checkout path")


def resolve_eval_device(args: argparse.Namespace, config_dict: dict[str, Any]) -> tuple[str, Optional[int], str]:
    requested_device = str(args.device or config_dict.get("project", {}).get("device", "cuda:0"))
    if requested_device.startswith("cuda") and os.environ.get("CUDA_VISIBLE_DEVICES"):
        hidden = os.environ.pop("CUDA_VISIBLE_DEVICES")
        print(
            f"[INFO] Ignoring CUDA_VISIBLE_DEVICES={hidden}; "
            f"--device {requested_device} selects the physical GPU index."
        )

    device_index = None
    if not torch.cuda.is_available():
        return "cpu", None, requested_device

    if requested_device.startswith("cuda"):
        try:
            device_index = int(requested_device.split(":", 1)[1]) if ":" in requested_device else 0
        except ValueError as exc:
            raise ValueError(f"Invalid CUDA device: {requested_device}") from exc
        if device_index >= torch.cuda.device_count():
            visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            raise ValueError(
                f"Requested {requested_device}, but PyTorch sees only {torch.cuda.device_count()} CUDA device(s). "
                f"CUDA_VISIBLE_DEVICES={visible}. Unset CUDA_VISIBLE_DEVICES if you want --device cuda:N to mean physical GPU N."
            )
        torch.cuda.set_device(requested_device)
        return requested_device, device_index, requested_device

    return requested_device, None, requested_device


def configure_egl(args: argparse.Namespace, device_index: Optional[int]) -> bool:
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    use_egl = not getattr(args, "no_egl", False)
    if use_egl and device_index is not None:
        os.environ["EGL_VISIBLE_DEVICES"] = str(device_index)
    return use_egl


def build_eval_vla_config(args: argparse.Namespace, device: str) -> dict[str, Any]:
    return {
        "type": args.vla_type,
        "device": device,
        "action_dim": 7,
        "chunk_size": 1,
        "model_path": getattr(args, "vla_model", None),
        "dtype": getattr(args, "vla_dtype", "bf16"),
        "load_in_8bit": getattr(args, "load_in_8bit", False),
        "load_in_4bit": getattr(args, "load_in_4bit", False),
        "unnorm_key": getattr(args, "unnorm_key", None),
        "hf_token": getattr(args, "hf_token", None),
        "calvin_policy_ckpt": getattr(args, "calvin_policy_ckpt", None),
        "calvin_train_folder": getattr(args, "calvin_train_folder", None),
        "calvin_dataset_path": getattr(args, "calvin_dataset_path", None) or getattr(args, "data_dir", None),
        "flower_checkpoint_dir": getattr(args, "flower_checkpoint_dir", None),
        "flower_code_path": getattr(args, "flower_code_path", None),
    }


def to_device_obs(obs_dict: dict[str, Any], device: str) -> dict[str, Any]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in obs_dict.items()}


def eval_metadata(args: argparse.Namespace, device: str, requested_device: str, use_egl: bool) -> dict[str, Any]:
    return {
        "vla_type": args.vla_type,
        "device": device,
        "requested_device": requested_device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "egl_visible_devices": os.environ.get("EGL_VISIBLE_DEVICES"),
        "use_egl": use_egl,
        "encoder_ckpt": getattr(args, "encoder_ckpt", None),
        "cewm_ckpt": getattr(args, "cewm_ckpt", None),
        "calvin_policy_ckpt": getattr(args, "calvin_policy_ckpt", None),
        "calvin_train_folder": getattr(args, "calvin_train_folder", None),
        "calvin_dataset_path": getattr(args, "calvin_dataset_path", None) or getattr(args, "data_dir", None),
        "flower_checkpoint_dir": getattr(args, "flower_checkpoint_dir", None),
        "flower_code_path": getattr(args, "flower_code_path", None),
    }
