"""非对称双流推理拓扑编排。

编排 VLA 主策略流和 CE-WM 裁判流的并行推理：
VLA 生成候选动作 → Encoder 编码观测 → CE-WM 能量评估 + 不确定性估计
→ 双向门控 → EFE 偏转 → 输出校正动作

关键约束: 所有模型参数 requires_grad=False
"""

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.dual_stream.vla_adapter import VLAAdapter
from src.steering.bilateral_gating import BilateralGating
from src.steering.efe_steering import EFESteering

logger = logging.getLogger("ce-ais")


class DualStreamTopology:
    """非对称双流推理拓扑。

    编排完整推理流程，确保所有模型参数绝对冻结。

    Args:
        vla_adapter: VLA 插件适配器。
        encoder: 对比编码器。
        ce_wm: 因果能量世界模型。
        steering: EFE 偏转模块。
        gating: 双向门控模块。
        mc_samples: MC-Dropout 采样次数。
    """

    def __init__(
        self,
        vla_adapter: VLAAdapter,
        encoder: nn.Module,
        ce_wm: nn.Module,
        steering: EFESteering,
        gating: BilateralGating,
        mc_samples: int = 5,
        compile_ce_wm: bool = True,
    ):
        self.vla = vla_adapter
        self.encoder = encoder
        self.ce_wm = ce_wm
        self.steering = steering
        self.gating = gating
        self.mc_samples = mc_samples

        # 上一次动作（用于安全回退）
        self.last_action: Optional[torch.Tensor] = None

        # 绝对冻结所有参数
        self._freeze_all()

        # torch.compile 加速 CE-WM 推理
        if compile_ce_wm:
            try:
                self.ce_wm = torch.compile(self.ce_wm, mode="reduce-overhead")
                logger.info("CE-WM compiled with torch.compile (reduce-overhead)")
            except Exception as e:
                logger.warning("torch.compile failed, using eager mode: %s", e)

    def _freeze_all(self) -> None:
        """冻结所有模型参数，确保 requires_grad=False。"""
        for model in [self.encoder, self.ce_wm]:
            for param in model.parameters():
                param.requires_grad = False

        # VLA 适配器参数冻结
        for param in self.vla.parameters():
            param.requires_grad = False

    def step(
        self,
        observation: dict,
        language_instruction: str,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """单步推理。

        Args:
            observation: {"rgb", "depth", "pose"} 多模态观测。
            language_instruction: 语言指令。

        Returns:
            action: 校正后的动作张量。
            info: 诊断信息字典。
        """
        # 1. 主策略流: VLA 生成候选动作
        a_init = self.vla.predict(observation, language_instruction)

        # 2. 编码: 观测 → 潜变量
        z_t = self.encoder.encode(observation)

        # 3. 裁判流: CE-WM 能量评估 + 不确定性
        T = a_init.shape[1]
        z_seq = z_t.unsqueeze(1).expand(-1, T, -1)

        with torch.no_grad():
            energy_before = self.ce_wm(z_seq, a_init.detach())

        uncertainty = self.ce_wm.get_uncertainty(
            z_seq, a_init.detach(), n_samples=self.mc_samples
        )

        # 4. 双向门控
        gating_lambda = self.gating.compute_lambda(uncertainty)

        # 5. EFE 偏转
        a_star = self.steering.steer(
            a_init, z_t, self.ce_wm, gating_lambda
        )

        # 偏转后能量
        with torch.no_grad():
            energy_after = self.ce_wm(z_seq, a_star)

        self.last_action = a_star.detach()

        info = {
            "energy_before": energy_before.detach().cpu(),
            "energy_after": energy_after.detach().cpu(),
            "uncertainty": uncertainty.detach().cpu(),
            "gating_lambda": gating_lambda.detach().cpu(),
        }

        return a_star.detach(), info

    def safe_step(
        self,
        observation: dict,
        language_instruction: str,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """带安全回退的推理步骤。

        当任何组件出现异常时，优雅降级至 VLA 原始输出。

        Args:
            observation: {"rgb", "depth", "pose"} 多模态观测。
            language_instruction: 语言指令。

        Returns:
            action: 动作张量。
            info: 诊断信息字典。
        """
        # 尝试获取 VLA 动作
        try:
            a_init = self.vla.predict(observation, language_instruction)
        except Exception as e:
            logger.error("VLA prediction failed: %s", e)
            if self.last_action is not None:
                return self.last_action, {"fallback": "action_hold", "error": str(e)}
            raise

        # 尝试完整偏转流程
        try:
            z_t = self.encoder.encode(observation)

            # 编码器输出异常检测
            if z_t.norm() < 1e-8:
                raise ValueError("Encoder output near-zero")

            T = a_init.shape[1]
            z_seq = z_t.unsqueeze(1).expand(-1, T, -1)

            uncertainty = self.ce_wm.get_uncertainty(
                z_seq, a_init.detach(), n_samples=self.mc_samples
            )
            gating_lambda = self.gating.compute_lambda(uncertainty)

            a_star = self.steering.steer(
                a_init, z_t, self.ce_wm, gating_lambda
            )

            # 检查偏转结果有效性
            if torch.isnan(a_star).any() or torch.isinf(a_star).any():
                raise ValueError("Steering produced NaN/Inf")

            self.last_action = a_star.detach()

            with torch.no_grad():
                energy_after = self.ce_wm(z_seq, a_star)

            return a_star.detach(), {
                "status": "steered",
                "uncertainty": uncertainty.detach().cpu(),
                "gating_lambda": gating_lambda.detach().cpu(),
                "energy_after": energy_after.detach().cpu(),
            }

        except Exception as e:
            logger.error("Steering failed, falling back to VLA: %s", e)
            self.last_action = a_init.detach()
            return a_init.detach(), {
                "fallback": "vla_prior",
                "error": str(e),
            }
