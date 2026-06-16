"""VLA 插件适配器。

定义统一的 VLA 基座模型接口（VLAAdapter ABC），
支持以插件方式挂载不同的 VLA 模型。

实现:
- VLAAdapter: 抽象基类，定义 predict() 接口
- OpenVLAAdapter: OpenVLA 7B 真实接入（HuggingFace transformers）
- ProxyVLAAdapter: 开发/测试用 MLP 代理（无依赖、无 GPU 也可跑）

OpenVLA 7B 在 RTX 4090 24GB 上的资源占用：
  - bfloat16: ~14GB（推荐）
  - 8bit (load_in_8bit=True): ~7GB（OOM 回退）
  - 4bit (load_in_4bit=True): ~4GB（极限节省，精度下降）
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


SUPPORTED_VLA_TYPES = ("proxy", "openvla", "flower", "calvin", "robovlms")


class VLAAdapter(ABC):
    """统一的 VLA 基座模型接口。

    所有 VLA 适配器必须实现 predict() 方法，
    输出候选动作分块 a ∈ R^{T×d_a}。
    """

    @abstractmethod
    def predict(
        self, observation: dict, instruction: str
    ) -> torch.Tensor:
        """生成候选动作序列。

        Args:
            observation: 多模态观测字典
                {"rgb": Tensor[B,3,H,W] in [0,1], "depth": Tensor[B,1,H,W], "pose": Tensor[B,d_p]}
            instruction: 语言指令字符串。

        Returns:
            action: Tensor[B, T, d_a] 候选动作分块。
        """
        ...

    @abstractmethod
    def parameters(self):
        """返回模型参数迭代器（用于冻结检查）。"""
        ...


class ProxyVLAAdapter(VLAAdapter):
    """开发/测试用代理 VLA。

    无外部依赖，使用 MLP 把 RGB 全局均值 + 简单 instruction hash 映射到动作。
    与真实 OpenVLA 输出 shape 一致，可在无 transformers / 无 GPU / 无网络场景下跑通管线。

    注意：此 adapter 仅用于开发和单元测试，**不要**用它跑论文实验。

    Args:
        action_dim: 动作维度。
        chunk_size: 动作分块长度 T。
        device: 计算设备。
        seed: 用于 instruction hash 的随机种子，使不同 instruction 产生不同输出。
    """

    def __init__(
        self,
        action_dim: int = 7,
        chunk_size: int = 1,
        device: str = "cpu",
        seed: int = 42,
    ):
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.device = device
        # 双输入投影：RGB 特征 + instruction hash
        self.model = nn.Sequential(
            nn.Linear(3 + 16, 128),
            nn.GELU(),
            nn.Linear(128, action_dim * chunk_size),
        )
        # 用 seed 初始化以保证可复现
        gen = torch.Generator().manual_seed(seed)
        for p in self.model.parameters():
            with torch.no_grad():
                p.copy_(torch.randn(p.shape, generator=gen) * 0.1)
        self.model.to(device)
        for p in self.model.parameters():
            p.requires_grad = False

    @staticmethod
    def _hash_instruction(instruction: str, dim: int = 16) -> torch.Tensor:
        """把 instruction 字符串确定性地映射到 16 维向量，使不同指令有不同表征。"""
        # 简单 hash：每个字符的 ord 累加到固定维度
        h = np.zeros(dim, dtype=np.float32)
        for i, ch in enumerate(instruction.encode("utf-8")):
            h[i % dim] += float(ch)
        h = h / (np.linalg.norm(h) + 1e-8)
        return torch.from_numpy(h)

    def predict(self, observation: dict, instruction: str) -> torch.Tensor:
        rgb = observation["rgb"]
        B = rgb.shape[0]
        with torch.no_grad():
            rgb_feat = rgb.mean(dim=(-2, -1))  # [B, 3]
            instr_feat = self._hash_instruction(instruction).to(rgb.device)
            instr_feat = instr_feat.unsqueeze(0).expand(B, -1)  # [B, 16]
            feat = torch.cat([rgb_feat, instr_feat], dim=-1)  # [B, 19]
            raw = self.model(feat)  # [B, T*d_a]
            action = raw.view(B, self.chunk_size, self.action_dim)
        # 收敛到 [-1, 1] 区间，与 OpenVLA 归一化输出一致
        action = torch.tanh(action)
        return action

    def parameters(self):
        return self.model.parameters()


class OpenVLAAdapter(VLAAdapter):
    """OpenVLA 7B 真实适配器（HuggingFace transformers 后端）。

    加载 ``openvla/openvla-7b``，冻结所有参数，使用语言指令 + RGB 图像生成动作。

    OpenVLA 标准 prompt 模板：
        ``"In: What action should the robot take to {instruction.lower()}?\\nOut:"``

    OpenVLA 输出 7 维动作（dx, dy, dz, droll, dpitch, dyaw, gripper），
    这里通过 ``unnorm_key`` 选择数据集特定的反归一化（默认 "bridge_orig"）。

    Args:
        model_path: HuggingFace 模型 ID，默认 "openvla/openvla-7b"。
        action_dim: 动作维度（OpenVLA 固定 7）。
        chunk_size: 动作分块长度 T；OpenVLA 原生输出单步，T>1 时复制扩展。
        device: 计算设备。
        dtype: 模型 dtype，"bf16" / "fp16" / "fp32"。
        load_in_8bit: 启用 bitsandbytes 8bit 量化（4090 OOM 回退）。
        load_in_4bit: 启用 bitsandbytes 4bit 量化。
        unnorm_key: OpenVLA 反归一化数据集 key（"bridge_orig" / "fractal20220817_data" 等）。
        hf_token: HuggingFace token（OpenVLA 需要 gated 访问）。
    """

    DEFAULT_MODEL_ID = "openvla/openvla-7b"
    DEFAULT_UNNORM_KEY = "bridge_orig"

    def __init__(
        self,
        model_path: Optional[str] = None,
        action_dim: int = 7,
        chunk_size: int = 1,
        device: str = "cuda",
        dtype: str = "bf16",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        unnorm_key: Optional[str] = None,
        hf_token: Optional[str] = None,
    ):
        self.model_path = model_path or self.DEFAULT_MODEL_ID
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.device = device
        self.dtype = dtype
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.unnorm_key = unnorm_key or self.DEFAULT_UNNORM_KEY
        self.hf_token = hf_token

        self.processor, self.model = self._load_real_model()

        # 冻结所有参数（active inference 路线下 VLA 始终冻结）
        for p in self.model.parameters():
            p.requires_grad = False

    def _load_real_model(self):
        """加载 HuggingFace OpenVLA 模型与 processor。

        Raises:
            ImportError: transformers 未安装；提示 ``uv sync --extra vla``。
            RuntimeError: 量化时 bitsandbytes 未安装；提示 ``uv sync --extra quant``。
        """
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor  # type: ignore
        except ImportError as e:
            raise ImportError(
                "transformers 未安装。请运行：uv sync --extra vla"
            ) from e

        torch_dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[self.dtype]

        load_kwargs: dict = {
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        if self.hf_token:
            load_kwargs["token"] = self.hf_token

        if self.load_in_8bit or self.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig  # type: ignore
            except ImportError as e:
                raise RuntimeError(
                    "bitsandbytes 量化需要：uv sync --extra quant"
                ) from e
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=self.load_in_8bit,
                load_in_4bit=self.load_in_4bit,
            )
            # 8bit/4bit 时 transformers 会自动 device_map
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = self.device

        processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True, token=self.hf_token
        )
        model = AutoModelForVision2Seq.from_pretrained(self.model_path, **load_kwargs)
        model.eval()
        return processor, model

    @staticmethod
    def _build_prompt(instruction: str) -> str:
        """构造 OpenVLA 标准 prompt 模板。

        参考 OpenVLA 官方 README:
        https://github.com/openvla/openvla
        """
        return f"In: What action should the robot take to {instruction.strip().lower()}?\nOut:"

    @staticmethod
    def _tensor_to_pil(rgb: torch.Tensor):
        """[B,3,H,W] (float in [0,1]) → list of PIL Images."""
        from PIL import Image  # 已在依赖中

        rgb_np = (rgb.detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
        # [B, 3, H, W] → [B, H, W, 3]
        rgb_np = rgb_np.transpose(0, 2, 3, 1)
        return [Image.fromarray(img) for img in rgb_np]

    @torch.no_grad()
    def predict(self, observation: dict, instruction: str) -> torch.Tensor:
        """OpenVLA 推理。

        Args:
            observation: ``{"rgb": Tensor[B,3,H,W] in [0,1], ...}``；OpenVLA 只用 RGB。
            instruction: 自然语言指令，如 "pick up the red block"。

        Returns:
            ``Tensor[B, chunk_size, 7]``；如 chunk_size > 1，复制单步动作填满。
        """
        rgb = observation["rgb"]
        B = rgb.shape[0]
        images = self._tensor_to_pil(rgb)
        prompt = self._build_prompt(instruction)

        # OpenVLA processor 接受 [PIL Image] + prompt（每个样本同 prompt）
        # 这里逐样本调用 predict_action（OpenVLA 暴露的高层 API），避免手工解 logits
        actions = []
        for img in images:
            inputs = self.processor(prompt, img).to(
                self.device, dtype={"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[self.dtype]
            )
            # OpenVLA 在自定义 modeling 文件里实现了 predict_action
            # 输入：input_ids, attention_mask, pixel_values；输出：np.ndarray (7,)
            action_np = self.model.predict_action(
                **inputs, unnorm_key=self.unnorm_key, do_sample=False
            )
            actions.append(torch.from_numpy(np.asarray(action_np, dtype=np.float32)))

        action = torch.stack(actions, dim=0).to(rgb.device)  # [B, 7]

        # 扩展到 chunk_size：OpenVLA 原生不支持 action chunking，T>1 时复制
        if self.chunk_size > 1:
            action = action.unsqueeze(1).expand(-1, self.chunk_size, -1).clone()
        else:
            action = action.unsqueeze(1)  # [B, 1, 7]
        return action

    def parameters(self):
        return self.model.parameters()


class FlowerVLAAdapter(VLAAdapter):
    """FLOWER VLA adapter for CALVIN ABC->D checkpoints."""

    def __init__(
        self,
        flower_checkpoint_dir: Optional[str] = None,
        flower_code_path: Optional[str] = None,
        action_dim: int = 7,
        chunk_size: int = 1,
        device: str = "cuda",
    ):
        repo_root = Path(__file__).resolve().parents[2]
        self.checkpoint_dir = Path(flower_checkpoint_dir or repo_root / "data" / "flower_calvin_abc").expanduser()
        if not self.checkpoint_dir.is_absolute():
            self.checkpoint_dir = repo_root / self.checkpoint_dir
        self.flower_code_path = Path(flower_code_path).expanduser() if flower_code_path else None
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.device = device
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32).view(1, 1, 3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32).view(1, 1, 3, 1, 1)
        self.model = self._load_model()

    def _load_model(self):
        config_path = self.checkpoint_dir / "config.yaml"
        weights_path = self.checkpoint_dir / "model.safetensors"
        missing = [str(p) for p in (config_path, weights_path) if not p.exists()]
        if missing:
            raise RuntimeError(f"FLOWER checkpoint files missing: {missing}")

        if self.flower_code_path is not None:
            if not self.flower_code_path.exists():
                raise RuntimeError(f"FLOWER code path does not exist: {self.flower_code_path}")
            code_path = str(self.flower_code_path)
            if code_path not in sys.path:
                sys.path.insert(0, code_path)

        try:
            import hydra
            from omegaconf import OmegaConf
        except Exception as exc:
            raise RuntimeError(
                "FLOWER adapter requires the external FLOWER repo dependencies plus safetensors. "
                "Pass --flower-code-path <flower_vla_calvin_repo> and install its environment."
            ) from exc

        try:
            cfg = OmegaConf.load(config_path)
            if OmegaConf.select(cfg, "model.load_pretrained") is not None:
                cfg.model.load_pretrained = True
            if OmegaConf.select(cfg, "model.pretrained_model_path") is not None:
                cfg.model.pretrained_model_path = str(weights_path)
            model = hydra.utils.instantiate(cfg.model)
            self._load_flower_compat_weights(model, weights_path)
            model.to(self.device)
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            return model
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load FLOWER VLA from {self.checkpoint_dir}. "
                "Verify that --flower-code-path points to the official FLOWER code and matches this checkpoint."
            ) from exc

    @staticmethod
    def _strip_state_prefix(key: str) -> str:
        for prefix in ("model.", "module.", "ema_model.", "_orig_mod."):
            if key.startswith(prefix):
                return key[len(prefix):]
        return key

    @staticmethod
    def _load_flower_compat_weights(model, weights_path: Path) -> None:
        if weights_path.suffix != ".safetensors":
            return
        try:
            from safetensors.torch import load_file
        except Exception:
            return
        state_dict = load_file(str(weights_path), device="cpu")
        remapped = {}
        key_map = {
            "vlm.language_final_logits_bias": "vlm.language_model.final_logits_bias",
            "vlm.language_shared.weight": "vlm.language_model.model.shared.weight",
        }
        model_state = model.state_dict()
        for old_key, new_key in key_map.items():
            if old_key in state_dict and new_key in model_state and state_dict[old_key].shape == model_state[new_key].shape:
                remapped[new_key] = state_dict[old_key]
        if remapped:
            model.load_state_dict(remapped, strict=False)

    def _image_to_tensor(self, image, pad: int = 10) -> torch.Tensor:
        if isinstance(image, np.ndarray):
            tensor = torch.from_numpy(image)
        elif isinstance(image, torch.Tensor):
            tensor = image.detach().cpu()
        else:
            raise TypeError(f"Unsupported FLOWER image type: {type(image)!r}")
        if tensor.dim() == 3 and tensor.shape[-1] == 3:
            tensor = tensor.permute(2, 0, 1)
        if tensor.dim() != 3:
            raise ValueError(f"Expected image shape [H,W,3] or [3,H,W], got {tuple(tensor.shape)}")
        tensor = tensor.byte().unsqueeze(0)  # [1, 3, H, W] uint8
        tensor = torchvision.transforms.functional.resize(
            tensor, size=[224, 224], antialias=True
        )  # [1, 3, 224, 224] uint8 — matches official torchvision.transforms.Resize
        tensor = tensor.float().div(255)  # ScaleImageTensor
        tensor = tensor.unsqueeze(1).to(self.device)  # [1, 1, 3, 224, 224]
        return (tensor - self.mean.to(self.device)) / self.std.to(self.device)

    def _build_obs(self, observation: dict) -> dict:
        raw_obs = observation.get("raw_calvin_obs")
        if raw_obs is None:
            raise ValueError("FLOWER VLA requires observation['raw_calvin_obs'] from CALVINWrapper.")
        rgb_obs = raw_obs.get("rgb_obs")
        if not isinstance(rgb_obs, dict):
            raise ValueError("FLOWER VLA requires raw_calvin_obs['rgb_obs'] with static and gripper cameras.")
        static = rgb_obs.get("rgb_static")
        gripper = rgb_obs.get("rgb_gripper")
        if static is None or gripper is None:
            raise ValueError("FLOWER VLA requires both rgb_static and rgb_gripper observations.")
        return {
            "rgb_obs": {
                "rgb_static": self._image_to_tensor(static, pad=10),
                "rgb_gripper": self._image_to_tensor(gripper, pad=4),
            }
        }

    @torch.no_grad()
    def predict(self, observation: dict, instruction: str) -> torch.Tensor:
        model_obs = self._build_obs(observation)
        goal = {"lang_text": instruction}
        if not hasattr(self.model, "step"):
            raise RuntimeError("Loaded FLOWER model does not expose step(obs, goal); check FLOWER code version.")
        action = self.model.step(model_obs, goal)
        if isinstance(action, dict):
            action = action.get("action", action.get("actions"))
        if isinstance(action, tuple):
            action = action[0]
        if action is None:
            raise RuntimeError("FLOWER model returned no action.")
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if action.dim() == 1:
            action = action.view(1, 1, -1)
        elif action.dim() == 2:
            action = action.unsqueeze(0) if action.shape[-1] == self.action_dim else action.unsqueeze(1)
        elif action.dim() > 3:
            action = action.view(action.shape[0], -1, action.shape[-1])
        action = action[..., :self.action_dim]
        if action.shape[1] > self.chunk_size:
            action = action[:, :self.chunk_size, :]
        elif action.shape[1] < self.chunk_size:
            pad = action[:, -1:, :].expand(-1, self.chunk_size - action.shape[1], -1)
            action = torch.cat([action, pad], dim=1)
        return action

    def reset(self) -> None:
        if hasattr(self.model, "reset"):
            self.model.reset()

    def parameters(self):
        return self.model.parameters()


class CalvinPolicyAdapter(VLAAdapter):
    """CALVIN-native policy adapter.

    Wraps an official CALVIN policy checkpoint so CE-AIS can use an
    action-compatible base policy instead of zero-shot OpenVLA actions.
    """

    def __init__(
        self,
        calvin_policy_ckpt: Optional[str] = None,
        calvin_train_folder: Optional[str] = None,
        calvin_dataset_path: Optional[str] = None,
        action_dim: int = 7,
        chunk_size: int = 1,
        device: str = "cuda",
    ):
        if not calvin_policy_ckpt or not calvin_train_folder:
            raise ValueError(
                "CALVIN policy requested but no checkpoint/train folder was provided. "
                "Pass --calvin-policy-ckpt and --calvin-train-folder."
            )
        if not str(device).startswith("cuda"):
            raise ValueError("CALVIN policy adapter currently requires a CUDA device, e.g. --device cuda:0.")

        self.calvin_policy_ckpt = str(Path(calvin_policy_ckpt))
        self.calvin_train_folder = str(Path(calvin_train_folder))
        self.calvin_dataset_path = str(Path(calvin_dataset_path)) if calvin_dataset_path else None
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.device = device
        self.observation_space_keys = None
        self.transforms = None
        self.proprio_state = None
        self.model = self._load_model()

    def _device_id(self) -> int:
        return int(self.device.split(":", 1)[1]) if ":" in self.device else 0

    def _load_model(self):
        if self.calvin_dataset_path is None:
            raise ValueError("CALVIN policy requested but no dataset path was provided. Pass --calvin-dataset-path.")
        try:
            from pydoc import locate

            import calvin_agent
            import hydra
            from omegaconf import OmegaConf
        except Exception as exc:
            raise RuntimeError(
                "Unable to import official CALVIN policy dependencies. "
                "Ensure calvin_agent, hydra, and omegaconf are installed/importable before using --vla-type calvin. "
                f"train_folder={self.calvin_train_folder}, checkpoint={self.calvin_policy_ckpt}"
            ) from exc

        try:
            train_cfg_path = Path(self.calvin_train_folder) / ".hydra" / "config.yaml"
            cfg = OmegaConf.load(train_cfg_path)
            lang_folder = cfg.datamodule.datasets.lang_dataset.lang_folder
            calvin_models_dir = Path(calvin_agent.__file__).resolve().parents[1]
            datasets_conf_dir = calvin_models_dir / "conf" / "datamodule" / "datasets"

            if hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
                hydra.core.global_hydra.GlobalHydra.instance().clear()
            with hydra.initialize_config_dir(config_dir=str(datasets_conf_dir), version_base="1.1"):
                datasets_cfg = hydra.compose(
                    "vision_lang.yaml", overrides=["lang_dataset.lang_folder=" + str(lang_folder)]
                )
            cfg.datamodule.datasets = datasets_cfg
            cfg.datamodule.root_data_dir = self.calvin_dataset_path
            data_module = hydra.utils.instantiate(cfg.datamodule, num_workers=0)
            data_module.prepare_data()
            data_module.setup()
            dataset = data_module.val_dataloader().dataset.datasets["lang"]
            self.observation_space_keys = dataset.observation_space
            self.transforms = dataset.transforms
            self.proprio_state = dataset.proprio_state

            model_cls = locate(cfg.model._target_)
            if model_cls is None:
                raise RuntimeError(f"Unable to locate CALVIN model class: {cfg.model._target_}")
            model = model_cls.load_from_checkpoint(self.calvin_policy_ckpt)
            model.load_lang_embeddings(dataset.abs_datasets_dir / dataset.lang_folder / "embeddings.npy")
            model.freeze()
            if cfg.model.action_decoder.get("load_action_bounds", False):
                model.action_decoder._setup_action_bounds(cfg.datamodule.root_data_dir, None, None, True)
            model = model.cuda(self._device_id())
        except Exception as exc:
            raise RuntimeError(
                "Failed to load CALVIN policy. "
                f"train_folder={self.calvin_train_folder}, checkpoint={self.calvin_policy_ckpt}, "
                f"dataset_path={self.calvin_dataset_path}"
            ) from exc
        model.eval()
        return model

    def _transform_observation(self, raw_obs: dict) -> dict:
        if self.observation_space_keys is None or self.transforms is None or self.proprio_state is None:
            raise RuntimeError("CALVIN observation transforms are not initialized.")
        from calvin_agent.datasets.utils.episode_utils import process_depth, process_rgb, process_state

        device = torch.device(self.device)
        state_obs = process_state(raw_obs, self.observation_space_keys, self.transforms, self.proprio_state)
        rgb_obs = process_rgb(raw_obs["rgb_obs"], self.observation_space_keys, self.transforms)
        depth_obs = process_depth(raw_obs["depth_obs"], self.observation_space_keys, self.transforms)
        state_obs["robot_obs"] = state_obs["robot_obs"].to(device).unsqueeze(0)
        rgb_obs.update({"rgb_obs": {k: v.to(device).unsqueeze(0) for k, v in rgb_obs["rgb_obs"].items()}})
        depth_obs.update({"depth_obs": {k: v.to(device).unsqueeze(0) for k, v in depth_obs["depth_obs"].items()}})
        return {
            **rgb_obs,
            **state_obs,
            **depth_obs,
            "robot_obs_raw": torch.from_numpy(raw_obs["robot_obs"]).to(device),
        }

    @torch.no_grad()
    def predict(self, observation: dict, instruction: str) -> torch.Tensor:
        raw_obs = observation.get("raw_calvin_obs")
        if raw_obs is None:
            raise ValueError("CALVIN policy requires observation['raw_calvin_obs'] from CALVINWrapper.")
        model_obs = self._transform_observation(raw_obs)
        action = self.model.step(model_obs, instruction)
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        elif action.dim() > 2:
            action = action[:, 0, :]
        action = action[:, :self.action_dim]
        if self.chunk_size > 1:
            action = action.unsqueeze(1).expand(-1, self.chunk_size, -1).clone()
        else:
            action = action.unsqueeze(1)
        return action

    def reset(self) -> None:
        if hasattr(self.model, "reset"):
            self.model.reset()

    def parameters(self):
        if hasattr(self.model, "parameters"):
            return self.model.parameters()
        return iter(())


class RoboVLMsAdapter(VLAAdapter):
    """RoboVLMs (Kosmos-PH) adapter for CALVIN ABC→D."""

    def __init__(
        self,
        robovlms_checkpoint_dir: Optional[str] = None,
        robovlms_code_path: Optional[str] = None,
        action_dim: int = 7,
        chunk_size: int = 1,
        device: str = "cuda",
    ):
        repo_root = Path(__file__).resolve().parents[2]
        self.checkpoint_dir = Path(robovlms_checkpoint_dir or repo_root / "data" / "robovlms").expanduser()
        self.code_path = Path(robovlms_code_path or repo_root / "external" / "RoboVLMs").expanduser()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.device = device
        self._custom_model = self._load_model()

    def _load_model(self):
        import json as _json
        config_path = self.checkpoint_dir / "configs" / "kosmos_ph_calvin_abc.json"
        ckpt_path = self.checkpoint_dir / "checkpoints" / "kosmos_ph_calvin_abc.pt"
        if not config_path.exists():
            raise RuntimeError(f"RoboVLMs config not found: {config_path}")
        if not ckpt_path.exists():
            raise RuntimeError(f"RoboVLMs checkpoint not found: {ckpt_path}")

        code_path_str = str(self.code_path)
        if code_path_str not in sys.path:
            sys.path.insert(0, code_path_str)

        configs = _json.loads(config_path.read_text())
        hf_cache = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
        kosmos_local = Path(hf_cache) / "hub" / "kosmos-2-patch14-224"
        if kosmos_local.exists():
            configs["model_path"] = str(kosmos_local)
            configs["vlm"]["pretrained_model_name_or_path"] = str(kosmos_local)
            configs["tokenizer"]["pretrained_model_name_or_path"] = str(kosmos_local)
        else:
            configs["model_path"] = "microsoft/kosmos-2-patch14-224"
            configs["vlm"]["pretrained_model_name_or_path"] = "microsoft/kosmos-2-patch14-224"
            configs["tokenizer"]["pretrained_model_name_or_path"] = "microsoft/kosmos-2-patch14-224"

        # Patch transformers kosmos2 with RoboVLMs' custom version
        import importlib
        import transformers
        patch_src = self.code_path / "tools" / "modeling_kosmos2.py"
        if patch_src.exists():
            import shutil
            dst = Path(transformers.__path__[0]) / "models" / "kosmos2" / "modeling_kosmos2.py"
            shutil.copy2(str(patch_src), str(dst))
            if "transformers.models.kosmos2.modeling_kosmos2" in sys.modules:
                del sys.modules["transformers.models.kosmos2.modeling_kosmos2"]
            from transformers.models.kosmos2 import modeling_kosmos2 as _k2mod
            if hasattr(_k2mod, "Kosmos2ForConditionalGeneration"):
                transformers.Kosmos2ForConditionalGeneration = _k2mod.Kosmos2ForConditionalGeneration

        try:
            import importlib.util
            wrapper_path = self.code_path / "eval" / "calvin" / "model_wrapper.py"
            spec = importlib.util.spec_from_file_location("model_wrapper", str(wrapper_path))
            model_wrapper_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_wrapper_mod)
            CustomModel = model_wrapper_mod.CustomModel
        except Exception as exc:
            raise RuntimeError(
                "RoboVLMs eval code not importable. "
                f"Ensure {self.code_path} is the RoboVLMs repo root."
            ) from exc

        model = CustomModel(
            ckpt_path=str(ckpt_path),
            configs=configs,
            device=self.device,
            raw_calvin=True,
        )
        return model

    @torch.no_grad()
    def predict(self, observation: dict, instruction: str) -> torch.Tensor:
        raw_obs = observation.get("raw_calvin_obs")
        if raw_obs is None:
            raise ValueError("RoboVLMs requires observation['raw_calvin_obs'].")
        rgb_obs = raw_obs.get("rgb_obs")
        if not isinstance(rgb_obs, dict):
            raise ValueError("RoboVLMs requires raw_calvin_obs['rgb_obs'].")

        obs = {"rgb_obs": rgb_obs}
        if "robot_obs" in raw_obs:
            obs["robot_obs"] = raw_obs["robot_obs"]

        action = self._custom_model.step(obs, instruction)

        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action)
        action = action.float().to(self.device)
        if action.dim() == 1:
            action = action.view(1, 1, -1)
        elif action.dim() == 2:
            action = action.unsqueeze(0)
        return action[..., :self.action_dim]

    def reset(self) -> None:
        self._custom_model.reset()

    def parameters(self):
        return self._custom_model.policy.parameters()


def build_vla_adapter(config: dict) -> VLAAdapter:
    """工厂函数：根据配置构造 VLA adapter。

    config 字段：
        type: "openvla" | "proxy" | "calvin" | "flower"
        model_path / dtype / load_in_8bit / load_in_4bit / unnorm_key / device
        calvin_policy_ckpt / calvin_train_folder / calvin_dataset_path
        flower_checkpoint_dir / flower_code_path
        action_dim / chunk_size
    """
    vla_type = config.get("type", "proxy")
    common = {
        "action_dim": config.get("action_dim", 7),
        "chunk_size": config.get("chunk_size", 1),
        "device": config.get("device", "cuda"),
    }
    if vla_type == "openvla":
        return OpenVLAAdapter(
            model_path=config.get("model_path"),
            dtype=config.get("dtype", "bf16"),
            load_in_8bit=config.get("load_in_8bit", False),
            load_in_4bit=config.get("load_in_4bit", False),
            unnorm_key=config.get("unnorm_key"),
            hf_token=config.get("hf_token"),
            **common,
        )
    elif vla_type == "proxy":
        return ProxyVLAAdapter(seed=config.get("seed", 42), **common)
    elif vla_type == "flower":
        return FlowerVLAAdapter(
            flower_checkpoint_dir=config.get("flower_checkpoint_dir"),
            flower_code_path=config.get("flower_code_path"),
            **common,
        )
    elif vla_type == "calvin":
        return CalvinPolicyAdapter(
            calvin_policy_ckpt=config.get("calvin_policy_ckpt"),
            calvin_train_folder=config.get("calvin_train_folder"),
            calvin_dataset_path=config.get("calvin_dataset_path"),
            **common,
        )
    elif vla_type == "robovlms":
        return RoboVLMsAdapter(
            robovlms_checkpoint_dir=config.get("robovlms_checkpoint_dir"),
            robovlms_code_path=config.get("robovlms_code_path"),
            **common,
        )
    else:
        raise ValueError(f"Unknown vla type: {vla_type}. Expected one of {SUPPORTED_VLA_TYPES}")
