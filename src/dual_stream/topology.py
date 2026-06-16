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
        compile_ce_wm: bool = False,
    ):
        self.vla = vla_adapter
        self.encoder = encoder
        self.ce_wm = ce_wm
        self.steering = steering
        self.gating = gating
        self.mc_samples = mc_samples

        # 上一次动作（用于安全回退）
        self.last_action: Optional[torch.Tensor] = None
        self.reset_diagnostics()

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
        """单步推理。"""
        return self.safe_step(observation, language_instruction)

    def reset(self) -> None:
        """重置跨子任务状态。"""
        self.last_action = None
        if hasattr(self.vla, "reset"):
            self.vla.reset()
        if hasattr(self.gating, "reset"):
            self.gating.reset()

    def reset_diagnostics(self) -> None:
        self._diag = {
            "steps_total": 0,
            "accepted_count": 0,
            "rejected_count": 0,
            "abstained_uncertainty_count": 0,
            "vla_prior_count": 0,
            "fallback_count": 0,
            "action_delta_inf_sum": 0.0,
            "energy_before_sum": 0.0,
            "energy_after_sum": 0.0,
            "uncertainty_sum": 0.0,
            "gating_lambda_sum": 0.0,
        }
        self._step_log = []

    def get_diagnostics(self) -> Dict[str, Any]:
        if not hasattr(self, "_diag"):
            self.reset_diagnostics()
        diag = dict(self._diag)
        total = max(int(diag["steps_total"]), 1)
        diag.update({
            "accepted_rate": diag["accepted_count"] / total,
            "rejected_rate": diag["rejected_count"] / total,
            "abstained_uncertainty_rate": diag["abstained_uncertainty_count"] / total,
            "vla_prior_rate": diag["vla_prior_count"] / total,
            "fallback_rate": diag["fallback_count"] / total,
            "action_delta_inf_mean": diag["action_delta_inf_sum"] / total,
            "energy_before_mean": diag["energy_before_sum"] / total,
            "energy_after_mean": diag["energy_after_sum"] / total,
            "uncertainty_mean": diag["uncertainty_sum"] / total,
            "gating_lambda_mean": diag["gating_lambda_sum"] / total,
        })
        # Per-step distributions for hyperparameter tuning
        if hasattr(self, "_step_log") and self._step_log:
            import numpy as _np
            uncertainties = [s["uncertainty"] for s in self._step_log if s.get("uncertainty") is not None]
            energy_deltas = [s["energy_before"] - s["energy_after"] for s in self._step_log
                           if s.get("energy_before") is not None and s.get("energy_after") is not None]
            lambdas = [s["gating_lambda"] for s in self._step_log if s.get("gating_lambda") is not None]
            action_deltas = [s["action_delta_inf"] for s in self._step_log if s.get("action_delta_inf") is not None]

            if uncertainties:
                u = _np.array(uncertainties)
                diag["uncertainty_percentiles"] = {
                    "p10": float(_np.percentile(u, 10)),
                    "p25": float(_np.percentile(u, 25)),
                    "p50": float(_np.percentile(u, 50)),
                    "p75": float(_np.percentile(u, 75)),
                    "p90": float(_np.percentile(u, 90)),
                    "p95": float(_np.percentile(u, 95)),
                    "max": float(u.max()),
                }
            if energy_deltas:
                ed = _np.array(energy_deltas)
                diag["energy_delta_percentiles"] = {
                    "p10": float(_np.percentile(ed, 10)),
                    "p50": float(_np.percentile(ed, 50)),
                    "p90": float(_np.percentile(ed, 90)),
                    "mean": float(ed.mean()),
                }
            if lambdas:
                la = _np.array(lambdas)
                diag["gating_lambda_percentiles"] = {
                    "p50": float(_np.percentile(la, 50)),
                    "p90": float(_np.percentile(la, 90)),
                    "p95": float(_np.percentile(la, 95)),
                }
            if action_deltas:
                ad = _np.array(action_deltas)
                diag["action_delta_percentiles"] = {
                    "p50": float(_np.percentile(ad, 50)),
                    "p90": float(_np.percentile(ad, 90)),
                    "p95": float(_np.percentile(ad, 95)),
                    "max": float(ad.max()),
                }
            # Intervention outcome stats
            intervened_steps = [s for s in self._step_log if s.get("accepted")]
            passthrough_steps = [s for s in self._step_log if not s.get("accepted")]
            diag["intervention_count"] = len(intervened_steps)
            diag["passthrough_count"] = len(passthrough_steps)
        return diag

    def safe_step(
        self,
        observation: dict,
        language_instruction: str,
        a_init: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """带安全回退的推理步骤。"""
        if not hasattr(self, "_diag"):
            self.reset_diagnostics()

        if a_init is None:
            try:
                a_init = self.vla.predict(observation, language_instruction)
            except Exception as e:
                logger.error("VLA prediction failed: %s", e)
                info = self._make_info(status="fallback", error=str(e), reject_reason="vla_error")
                self._record_diagnostics(info)
                if self.last_action is not None:
                    return self.last_action, info
                raise

        try:
            z_t = self.encoder.encode(observation)
            if z_t.norm() < 1e-8:
                raise ValueError("Encoder output near-zero")

            T = a_init.shape[1]
            z_seq = z_t.unsqueeze(1).expand(-1, T, -1)
            with torch.no_grad():
                energy_before = self.ce_wm(z_seq, a_init.detach())

            uncertainty = self.ce_wm.get_uncertainty(
                z_seq, a_init.detach(), n_samples=self.mc_samples
            )
            threshold = getattr(self.gating, "hard_uncertainty_threshold", None)
            if threshold is not None and torch.any(uncertainty > float(threshold)):
                info = self._make_info(
                    status="abstained_uncertainty",
                    accepted=False,
                    intervened=False,
                    reject_reason="uncertainty_threshold",
                    energy_before=energy_before,
                    energy_after=energy_before,
                    uncertainty=uncertainty,
                    gating_lambda=torch.zeros_like(uncertainty),
                    action_delta_inf=0.0,
                )
                self.last_action = a_init.detach()
                self._record_diagnostics(info)
                return a_init.detach(), info

            gating_lambda = self.gating.compute_lambda(uncertainty)
            min_lambda = float(getattr(self.gating, "min_lambda", 0.0) or 0.0)
            if torch.max(gating_lambda).item() <= min_lambda:
                info = self._make_info(
                    status="vla_prior",
                    accepted=False,
                    intervened=False,
                    reject_reason="low_gating_lambda",
                    energy_before=energy_before,
                    energy_after=energy_before,
                    uncertainty=uncertainty,
                    gating_lambda=gating_lambda,
                    action_delta_inf=0.0,
                )
                self.last_action = a_init.detach()
                self._record_diagnostics(info)
                return a_init.detach(), info

            a_star = self.steering.steer(a_init, z_t, self.ce_wm, gating_lambda)
            action_delta_inf = (a_star - a_init).detach().abs().max().item()
            steered_finite = torch.isfinite(a_star).all().item()
            delta_limit = float(getattr(self.steering, "action_delta_max", float("inf"))) + 1e-8
            delta_safe = action_delta_inf <= delta_limit or not getattr(
                self.steering, "enable_trust_region", True
            )

            if steered_finite:
                with torch.no_grad():
                    energy_after = self.ce_wm(z_seq, a_star.detach())
            else:
                energy_after = energy_before

            margin = float(getattr(self.steering, "accept_energy_margin", 0.0))
            energy_improved = torch.all(energy_after <= energy_before - margin).item()
            accept_reject = getattr(self.steering, "enable_accept_reject", True)
            accepted = bool(steered_finite and delta_safe and (energy_improved or not accept_reject))

            if accepted:
                action = a_star.detach()
                status = "accepted"
                reject_reason = None
            else:
                action = a_init.detach()
                status = "rejected"
                if not steered_finite:
                    reject_reason = "non_finite_action"
                elif not delta_safe:
                    reject_reason = "trust_region_violation"
                else:
                    reject_reason = "energy_not_improved"

            self.last_action = action
            info = self._make_info(
                status=status,
                accepted=accepted,
                intervened=accepted,
                reject_reason=reject_reason,
                energy_before=energy_before,
                energy_after=energy_after,
                uncertainty=uncertainty,
                gating_lambda=gating_lambda,
                action_delta_inf=action_delta_inf,
            )
            self._record_diagnostics(info)
            return action, info

        except Exception as e:
            logger.error("Steering failed, falling back to VLA: %s", e)
            self.last_action = a_init.detach()
            info = self._make_info(
                status="fallback",
                accepted=False,
                intervened=False,
                reject_reason="exception",
                error=str(e),
                action_delta_inf=0.0,
            )
            self._record_diagnostics(info)
            return a_init.detach(), info

    def _make_info(self, **kwargs: Any) -> Dict[str, Any]:
        info = {
            "status": kwargs.get("status", "fallback"),
            "accepted": bool(kwargs.get("accepted", False)),
            "intervened": bool(kwargs.get("intervened", False)),
            "fallback": kwargs.get("status") == "fallback",
            "reject_reason": kwargs.get("reject_reason"),
            "energy_before": self._to_cpu_tensor(kwargs.get("energy_before")),
            "energy_after": self._to_cpu_tensor(kwargs.get("energy_after")),
            "uncertainty": self._to_cpu_tensor(kwargs.get("uncertainty")),
            "gating_lambda": self._to_cpu_tensor(kwargs.get("gating_lambda")),
            "action_delta_inf": float(kwargs.get("action_delta_inf", 0.0) or 0.0),
        }
        if kwargs.get("error") is not None:
            info["error"] = kwargs["error"]
        return info

    def _record_diagnostics(self, info: Dict[str, Any]) -> None:
        if not hasattr(self, "_diag"):
            self.reset_diagnostics()
        self._diag["steps_total"] += 1
        status = info.get("status")
        if status == "accepted":
            self._diag["accepted_count"] += 1
        elif status == "rejected":
            self._diag["rejected_count"] += 1
        elif status == "abstained_uncertainty":
            self._diag["abstained_uncertainty_count"] += 1
        elif status == "vla_prior":
            self._diag["vla_prior_count"] += 1
        elif status == "fallback":
            self._diag["fallback_count"] += 1

        self._diag["action_delta_inf_sum"] += float(info.get("action_delta_inf") or 0.0)
        self._diag["energy_before_sum"] += self._to_float(info.get("energy_before")) or 0.0
        self._diag["energy_after_sum"] += self._to_float(info.get("energy_after")) or 0.0
        self._diag["uncertainty_sum"] += self._to_float(info.get("uncertainty")) or 0.0
        self._diag["gating_lambda_sum"] += self._to_float(info.get("gating_lambda")) or 0.0

        if hasattr(self, "_step_log"):
            self._step_log.append({
                "status": status,
                "accepted": info.get("accepted", False),
                "uncertainty": self._to_float(info.get("uncertainty")),
                "gating_lambda": self._to_float(info.get("gating_lambda")),
                "energy_before": self._to_float(info.get("energy_before")),
                "energy_after": self._to_float(info.get("energy_after")),
                "action_delta_inf": float(info.get("action_delta_inf") or 0.0),
            })

    @staticmethod
    def _to_cpu_tensor(value: Any) -> Optional[torch.Tensor]:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        return torch.tensor(float(value))

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return float(value.detach().float().mean().cpu().item())
        return float(value)
