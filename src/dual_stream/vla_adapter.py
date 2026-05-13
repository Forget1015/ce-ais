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
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


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


def build_vla_adapter(config: dict) -> VLAAdapter:
    """工厂函数：根据配置构造 VLA adapter。

    config 字段：
        type: "openvla" | "proxy"
        model_path / dtype / load_in_8bit / load_in_4bit / unnorm_key / device
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
    else:
        raise ValueError(f"Unknown vla type: {vla_type}")
