"""基线对比实验框架。

定义统一的基线方法接口，支持运行多种基线方法并生成对比结果汇总表格。

支持的基线方法:
- frozen_openvla: 冻结 OpenVLA（零样本基座）
- pdf: PDF（CVPR 2026 后置免微调修复）
- tt_vla: TT-VLA（在线强化学习伪奖励）
- ada_world_policy: AdaWorldPolicy（端到端在线微调）
"""

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("ce-ais")


@dataclass
class BaselineResult:
    """单个基线方法的评估结果。"""

    method_name: str
    chain_success_rate: Dict[int, float] = field(default_factory=dict)
    single_task_rate: Dict[str, float] = field(default_factory=dict)
    avg_steps: float = 0.0
    latency_ms: float = 0.0
    trajectory_jerk: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


class BaselineMethod(ABC):
    """基线方法抽象基类。

    所有基线方法需实现统一的 predict 接口。
    """

    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config

    @abstractmethod
    def predict(
        self, observation: dict, instruction: str
    ) -> torch.Tensor:
        """生成动作预测。

        Args:
            observation: {"rgb", "depth", "pose"} 多模态观测。
            instruction: 语言指令。

        Returns:
            action: 动作张量。
        """
        ...

    def setup(self) -> None:
        """初始化模型和资源（可选覆盖）。"""
        pass

    def teardown(self) -> None:
        """释放资源（可选覆盖）。"""
        pass


class FrozenOpenVLABaseline(BaselineMethod):
    """冻结 OpenVLA 零样本基座基线。"""

    def __init__(self, config: dict):
        super().__init__("frozen_openvla", config)
        self._adapter = None

    def setup(self) -> None:
        logger.info("Setting up Frozen OpenVLA baseline")
        from src.dual_stream.vla_adapter import build_vla_adapter

        adapter_config = {
            "type": self.config.get("vla_type", "openvla"),
            "model_path": self.config.get("model_path"),
            "device": self.config.get("device", "cuda"),
            "dtype": self.config.get("dtype", "bf16"),
            "load_in_8bit": self.config.get("load_in_8bit", False),
            "load_in_4bit": self.config.get("load_in_4bit", False),
            "unnorm_key": self.config.get("unnorm_key"),
            "hf_token": self.config.get("hf_token"),
            "action_dim": self.config.get("action_dim", 7),
            "chunk_size": self.config.get("chunk_size", 1),
            "calvin_policy_ckpt": self.config.get("calvin_policy_ckpt"),
            "calvin_train_folder": self.config.get("calvin_train_folder"),
            "calvin_dataset_path": self.config.get("calvin_dataset_path"),
        }
        self._adapter = build_vla_adapter(adapter_config)

    def predict(
        self, observation: dict, instruction: str
    ) -> torch.Tensor:
        if self._adapter is None:
            raise RuntimeError("Call setup() before predict()")
        return self._adapter.predict(observation, instruction)


class FrozenCalvinPolicyBaseline(BaselineMethod):
    """冻结 CALVIN-native policy 基线。"""

    def __init__(self, config: dict):
        super().__init__("frozen_calvin_policy", config)
        self._adapter = None

    def setup(self) -> None:
        logger.info("Setting up Frozen CALVIN policy baseline")
        from src.dual_stream.vla_adapter import build_vla_adapter

        adapter_config = {
            "type": "calvin",
            "device": self.config.get("device", "cuda"),
            "action_dim": self.config.get("action_dim", 7),
            "chunk_size": self.config.get("chunk_size", 1),
            "calvin_policy_ckpt": self.config.get("calvin_policy_ckpt"),
            "calvin_train_folder": self.config.get("calvin_train_folder"),
            "calvin_dataset_path": self.config.get("calvin_dataset_path"),
        }
        self._adapter = build_vla_adapter(adapter_config)

    def predict(self, observation: dict, instruction: str) -> torch.Tensor:
        if self._adapter is None:
            raise RuntimeError("Call setup() before predict()")
        return self._adapter.predict(observation, instruction)

    def reset_task(self) -> None:
        if self._adapter is not None and hasattr(self._adapter, "reset"):
            self._adapter.reset()


class PDFBaseline(BaselineMethod):
    """PDF（后置免微调修复）基线。

    参考 arXiv 2604.18107: 不微调 VLA 基座，而是在输出端
    追加轻量扰动模块，利用不确定性驱动的延迟反馈修正动作。
    """

    def __init__(self, config: dict):
        super().__init__("pdf", config)
        self._adapter = None
        self._perturb_net = None
        self._optimizer = None
        self._prev_obs_feat = None
        self._prev_delta = None

    def setup(self) -> None:
        from src.dual_stream.vla_adapter import build_vla_adapter

        device = self.config.get("device", "cuda")
        action_dim = self.config.get("action_dim", 7)
        adapter_config = {
            "type": self.config.get("vla_type", "openvla"),
            "model_path": self.config.get("model_path"),
            "device": device,
            "dtype": self.config.get("dtype", "bf16"),
            "load_in_8bit": self.config.get("load_in_8bit", False),
            "load_in_4bit": self.config.get("load_in_4bit", False),
            "unnorm_key": self.config.get("unnorm_key"),
            "hf_token": self.config.get("hf_token"),
            "action_dim": action_dim,
            "chunk_size": self.config.get("chunk_size", 1),
            "calvin_policy_ckpt": self.config.get("calvin_policy_ckpt"),
            "calvin_train_folder": self.config.get("calvin_train_folder"),
            "calvin_dataset_path": self.config.get("calvin_dataset_path"),
        }
        self._adapter = build_vla_adapter(adapter_config)

        feat_dim = 64
        self._obs_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(3, feat_dim), nn.ReLU(),
        ).to(device)

        # 扰动模块: obs_feat + action + uncertainty → delta
        self._perturb_net = nn.Sequential(
            nn.Linear(feat_dim + action_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        ).to(device)

        self._optimizer = torch.optim.Adam(
            list(self._perturb_net.parameters()) + list(self._obs_encoder.parameters()),
            lr=1e-4,
        )
        self._prev_obs_feat = None
        self._prev_delta = None

    def predict(self, observation: dict, instruction: str) -> torch.Tensor:
        if self._adapter is None:
            raise RuntimeError("Call setup() before predict()")
        device = self.config.get("device", "cuda")

        base_action = self._adapter.predict(observation, instruction)  # [B,T,7]
        B = base_action.shape[0]
        a_flat = base_action[:, 0, :]  # [B, 7]

        rgb = observation["rgb"]
        obs_feat = self._obs_encoder(rgb.mean(dim=1, keepdim=True).expand(-1, 3, -1, -1)
                                      if rgb.shape[1] != 3 else rgb)

        # 不确定性: K 次带噪声前向取方差
        K = 3
        noisy_actions = []
        for _ in range(K):
            noisy_rgb = rgb + 0.05 * torch.randn_like(rgb)
            obs_n = {**observation, "rgb": noisy_rgb.clamp(0, 1)}
            a_n = self._adapter.predict(obs_n, instruction)[:, 0, :]
            noisy_actions.append(a_n)
        uncertainty = torch.stack(noisy_actions).var(dim=0).mean(dim=-1, keepdim=True)  # [B,1]

        # 延迟反馈更新
        if self._prev_obs_feat is not None and self._prev_delta is not None:
            feat_diff = (obs_feat.detach() - self._prev_obs_feat).norm(dim=-1, keepdim=True)
            loss = -(feat_diff * self._prev_delta.detach().norm(dim=-1, keepdim=True)).mean()
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

        inp = torch.cat([obs_feat, a_flat, uncertainty], dim=-1)
        delta = self._perturb_net(inp) * 0.1  # [B, 7]
        corrected = a_flat + delta

        self._prev_obs_feat = obs_feat.detach()
        self._prev_delta = delta.detach()

        return corrected.unsqueeze(1)  # [B, 1, 7]

    def teardown(self) -> None:
        self._adapter = None
        self._perturb_net = None
        self._obs_encoder = None
        self._optimizer = None
        torch.cuda.empty_cache()


class TTVLABaseline(BaselineMethod):
    """TT-VLA（在线强化学习伪奖励）基线。

    使用步骤级进度分类器构建密集伪奖励，通过 REINFORCE
    在测试阶段在线更新动作残差适配器。
    """

    def __init__(self, config: dict):
        super().__init__("tt_vla", config)
        self._adapter = None
        self._residual = None
        self._progress_net = None
        self._optimizer = None
        self._prev_progress = 0.0
        self._prev_log_prob = None

    def setup(self) -> None:
        from src.dual_stream.vla_adapter import build_vla_adapter

        device = self.config.get("device", "cuda")
        action_dim = self.config.get("action_dim", 7)
        adapter_config = {
            "type": self.config.get("vla_type", "openvla"),
            "model_path": self.config.get("model_path"),
            "device": device,
            "dtype": self.config.get("dtype", "bf16"),
            "load_in_8bit": self.config.get("load_in_8bit", False),
            "load_in_4bit": self.config.get("load_in_4bit", False),
            "unnorm_key": self.config.get("unnorm_key"),
            "hf_token": self.config.get("hf_token"),
            "action_dim": action_dim,
            "chunk_size": self.config.get("chunk_size", 1),
            "calvin_policy_ckpt": self.config.get("calvin_policy_ckpt"),
            "calvin_train_folder": self.config.get("calvin_train_folder"),
            "calvin_dataset_path": self.config.get("calvin_dataset_path"),
        }
        self._adapter = build_vla_adapter(adapter_config)

        feat_dim = 64
        self._obs_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(3, feat_dim), nn.ReLU(),
        ).to(device)

        # 残差适配器: obs_feat + base_action → delta
        self._residual = nn.Sequential(
            nn.Linear(feat_dim + action_dim + 16, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        ).to(device)

        # 进度估计器: obs_feat → progress ∈ [0, 1]
        self._progress_net = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        ).to(device)

        self._optimizer = torch.optim.Adam(
            list(self._residual.parameters())
            + list(self._obs_encoder.parameters())
            + list(self._progress_net.parameters()),
            lr=3e-4,
        )
        self._prev_progress = 0.0
        self._prev_log_prob = None

    @staticmethod
    def _hash_instruction(instruction: str, dim: int = 16) -> torch.Tensor:
        h = np.zeros(dim, dtype=np.float32)
        for i, ch in enumerate(instruction.encode("utf-8")):
            h[i % dim] += float(ch)
        h = h / (np.linalg.norm(h) + 1e-8)
        return torch.from_numpy(h)

    def predict(self, observation: dict, instruction: str) -> torch.Tensor:
        if self._adapter is None:
            raise RuntimeError("Call setup() before predict()")
        device = self.config.get("device", "cuda")

        base_action = self._adapter.predict(observation, instruction)
        B = base_action.shape[0]
        a_flat = base_action[:, 0, :]  # [B, 7]

        rgb = observation["rgb"]
        obs_feat = self._obs_encoder(rgb.mean(dim=1, keepdim=True).expand(-1, 3, -1, -1)
                                      if rgb.shape[1] != 3 else rgb)

        instr_feat = self._hash_instruction(instruction).to(device).unsqueeze(0).expand(B, -1)
        inp = torch.cat([obs_feat, a_flat, instr_feat], dim=-1)

        delta_mean = self._residual(inp) * 0.1
        noise_std = 0.02
        noise = noise_std * torch.randn_like(delta_mean)
        delta = delta_mean + noise

        # log-probability for REINFORCE
        log_prob = -0.5 * ((noise / noise_std) ** 2).sum(dim=-1)

        # 进度估计
        progress = self._progress_net(obs_feat.detach()).squeeze(-1)  # [B]

        # REINFORCE 更新（利用上一步的 log_prob 和当前进度增量）
        if self._prev_log_prob is not None:
            reward = (progress.mean().item() - self._prev_progress)
            pg_loss = -(self._prev_log_prob * reward).mean()
            prog_target = torch.ones_like(progress) * max(self._prev_progress + 0.01, 0.5)
            prog_loss = F.mse_loss(progress, prog_target.detach())
            loss = pg_loss + 0.1 * prog_loss
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

        self._prev_progress = progress.mean().item()
        self._prev_log_prob = log_prob.mean()

        corrected = a_flat + delta.detach()
        return corrected.unsqueeze(1)

    def teardown(self) -> None:
        self._adapter = None
        self._residual = None
        self._progress_net = None
        self._obs_encoder = None
        self._optimizer = None
        self._prev_log_prob = None
        torch.cuda.empty_cache()


class AdaWorldPolicyBaseline(BaselineMethod):
    """AdaWorldPolicy（端到端在线微调）基线。

    耦合小型世界模型（MLP）与冻结 VLA，世界模型预测下一帧
    观测特征，在线通过预测误差更新策略残差和世界模型参数。
    """

    def __init__(self, config: dict):
        super().__init__("ada_world_policy", config)
        self._adapter = None
        self._world_model = None
        self._policy_residual = None
        self._wm_optimizer = None
        self._pol_optimizer = None
        self._prev_feat = None
        self._prev_pred_feat = None

    def setup(self) -> None:
        from src.dual_stream.vla_adapter import build_vla_adapter

        device = self.config.get("device", "cuda")
        action_dim = self.config.get("action_dim", 7)
        adapter_config = {
            "type": self.config.get("vla_type", "openvla"),
            "model_path": self.config.get("model_path"),
            "device": device,
            "dtype": self.config.get("dtype", "bf16"),
            "load_in_8bit": self.config.get("load_in_8bit", False),
            "load_in_4bit": self.config.get("load_in_4bit", False),
            "unnorm_key": self.config.get("unnorm_key"),
            "hf_token": self.config.get("hf_token"),
            "action_dim": action_dim,
            "chunk_size": self.config.get("chunk_size", 1),
            "calvin_policy_ckpt": self.config.get("calvin_policy_ckpt"),
            "calvin_train_folder": self.config.get("calvin_train_folder"),
            "calvin_dataset_path": self.config.get("calvin_dataset_path"),
        }
        self._adapter = build_vla_adapter(adapter_config)

        feat_dim = 64
        self._obs_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(3, feat_dim), nn.ReLU(),
        ).to(device)

        # 世界模型: feat + action → predicted_next_feat
        self._world_model = nn.Sequential(
            nn.Linear(feat_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feat_dim),
        ).to(device)

        # 策略残差: feat + prediction_error → delta_action
        self._policy_residual = nn.Sequential(
            nn.Linear(feat_dim + feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        ).to(device)

        self._wm_optimizer = torch.optim.Adam(
            list(self._world_model.parameters()) + list(self._obs_encoder.parameters()),
            lr=1e-3,
        )
        self._pol_optimizer = torch.optim.Adam(
            self._policy_residual.parameters(), lr=3e-4,
        )
        self._prev_feat = None
        self._prev_pred_feat = None

    def predict(self, observation: dict, instruction: str) -> torch.Tensor:
        if self._adapter is None:
            raise RuntimeError("Call setup() before predict()")
        device = self.config.get("device", "cuda")
        action_dim = self.config.get("action_dim", 7)

        base_action = self._adapter.predict(observation, instruction)
        B = base_action.shape[0]
        a_flat = base_action[:, 0, :]  # [B, 7]

        rgb = observation["rgb"]
        obs_feat = self._obs_encoder(rgb.mean(dim=1, keepdim=True).expand(-1, 3, -1, -1)
                                      if rgb.shape[1] != 3 else rgb)

        pred_error = torch.zeros(B, obs_feat.shape[-1], device=device)

        # 在线更新：用上一步的世界模型预测 vs 当前真实特征
        if self._prev_feat is not None and self._prev_pred_feat is not None:
            pred_error = (obs_feat.detach() - self._prev_pred_feat).detach()

            wm_loss = F.mse_loss(self._prev_pred_feat, obs_feat.detach())
            self._wm_optimizer.zero_grad()
            wm_loss.backward()
            self._wm_optimizer.step()

        # 策略残差修正
        inp = torch.cat([obs_feat.detach(), pred_error], dim=-1)
        delta = self._policy_residual(inp) * 0.1
        corrected = a_flat + delta

        # 世界模型预测下一步特征
        wm_inp = torch.cat([obs_feat.detach(), corrected.detach()], dim=-1)
        self._prev_pred_feat = self._world_model(wm_inp)
        self._prev_feat = obs_feat.detach()

        return corrected.unsqueeze(1)

    def teardown(self) -> None:
        self._adapter = None
        self._world_model = None
        self._policy_residual = None
        self._obs_encoder = None
        self._wm_optimizer = None
        self._pol_optimizer = None
        torch.cuda.empty_cache()


# 基线方法注册表
BASELINE_REGISTRY: Dict[str, type] = {
    "frozen_openvla": FrozenOpenVLABaseline,
    "frozen_calvin_policy": FrozenCalvinPolicyBaseline,
    "pdf": PDFBaseline,
    "tt_vla": TTVLABaseline,
    "ada_world_policy": AdaWorldPolicyBaseline,
}


class BaselineFramework:
    """基线对比实验框架。

    统一管理多种基线方法的运行和结果对比。

    Args:
        config: 评估配置字典。
        output_dir: 结果输出目录。
    """

    def __init__(self, config: dict, output_dir: str = "logs"):
        self.config = config
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._results: Dict[str, BaselineResult] = {}

    def create_baseline(
        self, method_name: str, method_config: Optional[dict] = None
    ) -> BaselineMethod:
        """创建基线方法实例。

        Args:
            method_name: 基线方法名称。
            method_config: 方法特定配置。

        Returns:
            BaselineMethod 实例。
        """
        if method_name not in BASELINE_REGISTRY:
            raise ValueError(
                f"Unknown baseline: {method_name}. "
                f"Available: {list(BASELINE_REGISTRY.keys())}"
            )
        cfg = method_config or self.config
        return BASELINE_REGISTRY[method_name](cfg)

    def run_baseline(
        self,
        method: BaselineMethod,
        env_wrapper,
        task_chains: List[List[str]],
        max_steps_per_task: int = 300,
    ) -> BaselineResult:
        """运行单个基线方法的评估。

        Args:
            method: 基线方法实例。
            env_wrapper: CALVINWrapper 环境封装。
            task_chains: 任务链列表。
            max_steps_per_task: 每个任务最大步数。

        Returns:
            BaselineResult 评估结果。
        """
        method.setup()
        chain_results: Dict[int, List[bool]] = {}
        task_results: Dict[str, List[bool]] = {}
        latencies: List[float] = []

        for chain in task_chains:
            result = env_wrapper.run_chain_evaluation(
                policy_fn=method.predict,
                task_chain=chain,
                max_steps_per_task=max_steps_per_task,
                policy_reset_fn=getattr(method, "reset_task", None),
            )

            chain_len = result["chain_length"]
            if chain_len not in chain_results:
                chain_results[chain_len] = []
            chain_results[chain_len].append(result["chain_success"])

            for tr in result["task_results"]:
                task_name = tr["task"]
                if task_name not in task_results:
                    task_results[task_name] = []
                task_results[task_name].append(tr["success"])

        # 计算成功率
        chain_rates = {
            l: sum(r) / len(r) for l, r in chain_results.items() if r
        }
        task_rates = {
            t: sum(r) / len(r) for t, r in task_results.items() if r
        }

        baseline_result = BaselineResult(
            method_name=method.name,
            chain_success_rate=chain_rates,
            single_task_rate=task_rates,
        )

        self._results[method.name] = baseline_result
        method.teardown()

        logger.info("Baseline %s evaluation complete", method.name)
        return baseline_result

    def run_all_baselines(
        self,
        env_wrapper,
        task_chains: List[List[str]],
        baseline_names: Optional[List[str]] = None,
        max_steps_per_task: int = 300,
    ) -> Dict[str, BaselineResult]:
        """运行所有（或指定的）基线方法。

        Args:
            env_wrapper: CALVINWrapper 环境封装。
            task_chains: 任务链列表。
            baseline_names: 要运行的基线名称列表。None 表示全部。
            max_steps_per_task: 每个任务最大步数。

        Returns:
            {method_name: BaselineResult} 字典。
        """
        names = baseline_names or list(BASELINE_REGISTRY.keys())
        for name in names:
            method = self.create_baseline(name)
            self.run_baseline(
                method, env_wrapper, task_chains, max_steps_per_task
            )
        return self._results

    def generate_comparison_table(
        self, filename: str = "baseline_comparison.json"
    ) -> str:
        """生成对比结果汇总表格。

        Returns:
            输出文件路径。
        """
        table = {}
        for name, result in self._results.items():
            table[name] = {
                "chain_success_rate": result.chain_success_rate,
                "single_task_rate": result.single_task_rate,
                "avg_steps": result.avg_steps,
                "latency_ms": result.latency_ms,
                "trajectory_jerk": result.trajectory_jerk,
            }

        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(table, f, indent=2, ensure_ascii=False)

        logger.info("Baseline comparison saved to %s", path)
        return path

    @staticmethod
    def list_available_baselines() -> List[str]:
        """列出所有可用的基线方法名称。"""
        return list(BASELINE_REGISTRY.keys())
