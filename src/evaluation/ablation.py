"""消融实验框架。

通过配置文件切换消融变体，无需修改核心推理代码。

支持的消融实验:
- no_gating: 剥离双向门控（固定引导强度）
- mse_energy: MSE 重建替换能量判别
- mamba1_backbone: Mamba-1 替换 Mamba-3
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ce-ais")


# 预定义消融变体
ABLATION_VARIANTS = {
    "no_gating": {
        "description": "剥离双向门控（固定引导强度 λ=λ_max）",
        "config_file": "configs/ablation/no_gating.yaml",
    },
    "mse_energy": {
        "description": "MSE 重建误差替换能量判别",
        "config_file": "configs/ablation/mse_energy.yaml",
    },
    "mamba1_backbone": {
        "description": "Mamba-1 替换 Mamba-3 时序引擎",
        "config_file": "configs/ablation/mamba1_backbone.yaml",
    },
}


@dataclass
class AblationResult:
    """单个消融变体的评估结果。"""

    variant_name: str
    description: str = ""
    chain_success_rate: Dict[int, float] = field(default_factory=dict)
    single_task_rate: Dict[str, float] = field(default_factory=dict)
    avg_steps: float = 0.0
    latency_ms: float = 0.0
    trajectory_jerk: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


class AblationFramework:
    """消融实验框架。

    通过配置文件切换消融变体，统一管理消融实验的运行和结果对比。

    Args:
        base_config: 基础配置字典（完整 CE-AIS 配置）。
        output_dir: 结果输出目录。
    """

    def __init__(self, base_config: dict, output_dir: str = "logs"):
        self.base_config = base_config
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._results: Dict[str, AblationResult] = {}

    def list_variants(self) -> Dict[str, str]:
        """列出所有可用的消融变体及其描述。"""
        return {
            name: info["description"]
            for name, info in ABLATION_VARIANTS.items()
        }

    def get_variant_config(self, variant_name: str) -> dict:
        """获取消融变体的配置文件路径。

        Args:
            variant_name: 消融变体名称。

        Returns:
            配置文件路径。
        """
        if variant_name not in ABLATION_VARIANTS:
            raise ValueError(
                f"Unknown ablation variant: {variant_name}. "
                f"Available: {list(ABLATION_VARIANTS.keys())}"
            )
        return ABLATION_VARIANTS[variant_name]

    def run_ablation(
        self,
        variant_name: str,
        eval_fn,
        config_override: Optional[dict] = None,
    ) -> AblationResult:
        """运行单个消融实验。

        Args:
            variant_name: 消融变体名称。
            eval_fn: 评估函数，接收配置字典，返回评估结果字典。
            config_override: 额外的配置覆盖。

        Returns:
            AblationResult 评估结果。
        """
        variant_info = self.get_variant_config(variant_name)

        # 加载消融配置（通过 ConfigManager 的继承机制）
        from src.config.config_manager import ConfigManager

        config_path = variant_info["config_file"]
        if os.path.exists(config_path):
            cm = ConfigManager(config_path=config_path)
            ablation_config = cm.config
        else:
            # 配置文件不存在时使用基础配置
            ablation_config = dict(self.base_config)
            logger.warning(
                "Ablation config %s not found, using base config",
                config_path,
            )

        if config_override:
            from src.config.config_manager import deep_merge
            ablation_config = deep_merge(ablation_config, config_override)

        logger.info(
            "Running ablation: %s (%s)",
            variant_name,
            variant_info["description"],
        )

        # 运行评估
        eval_result = eval_fn(ablation_config)

        result = AblationResult(
            variant_name=variant_name,
            description=variant_info["description"],
            chain_success_rate=eval_result.get("chain_success_rate", {}),
            single_task_rate=eval_result.get("single_task_rate", {}),
            avg_steps=eval_result.get("avg_steps", 0.0),
            latency_ms=eval_result.get("latency_ms", 0.0),
            trajectory_jerk=eval_result.get("trajectory_jerk", 0.0),
            extra=eval_result.get("extra", {}),
        )

        self._results[variant_name] = result
        return result

    def run_all_ablations(
        self,
        eval_fn,
        variant_names: Optional[List[str]] = None,
    ) -> Dict[str, AblationResult]:
        """运行所有（或指定的）消融实验。

        Args:
            eval_fn: 评估函数。
            variant_names: 要运行的变体名称列表。None 表示全部。

        Returns:
            {variant_name: AblationResult} 字典。
        """
        names = variant_names or list(ABLATION_VARIANTS.keys())
        for name in names:
            self.run_ablation(name, eval_fn)
        return self._results

    def generate_comparison_table(
        self, filename: str = "ablation_comparison.json"
    ) -> str:
        """生成消融结果对比表格。

        Returns:
            输出文件路径。
        """
        table = {}
        for name, result in self._results.items():
            table[name] = {
                "description": result.description,
                "chain_success_rate": result.chain_success_rate,
                "single_task_rate": result.single_task_rate,
                "avg_steps": result.avg_steps,
                "latency_ms": result.latency_ms,
                "trajectory_jerk": result.trajectory_jerk,
            }

        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(table, f, indent=2, ensure_ascii=False)

        logger.info("Ablation comparison saved to %s", path)
        return path
