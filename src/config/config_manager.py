"""层级化 YAML 配置管理器。

支持:
- YAML 配置文件加载
- 配置继承 (base → experiment) 通过 inherit_from 字段
- 命令行参数覆盖 (--key.subkey=value 格式)
- 运行时配置快照保存
"""

import copy
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    """深度合并两个字典，override 中的值覆盖 base 中的对应值。

    对于嵌套字典递归合并；对于非字典值直接覆盖。

    Args:
        base: 基础配置字典。
        override: 覆盖配置字典。

    Returns:
        合并后的新字典。
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def set_nested(config: dict, dotted_key: str, value: Any) -> None:
    """通过点分隔键设置嵌套字典中的值。

    Args:
        config: 配置字典。
        dotted_key: 点分隔的键路径，如 "ce_wm.d_model"。
        value: 要设置的值。
    """
    keys = dotted_key.split(".")
    current = config
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def get_nested(config: dict, dotted_key: str, default: Any = None) -> Any:
    """通过点分隔键获取嵌套字典中的值。

    Args:
        config: 配置字典。
        dotted_key: 点分隔的键路径。
        default: 键不存在时的默认值。

    Returns:
        对应的值，或 default。
    """
    keys = dotted_key.split(".")
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _auto_cast(value: str) -> Any:
    """尝试将字符串值自动转换为合适的 Python 类型。

    Args:
        value: 字符串值。

    Returns:
        转换后的值（int、float、bool 或原始字符串）。
    """
    # bool
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    # int
    try:
        return int(value)
    except ValueError:
        pass
    # float
    try:
        return float(value)
    except ValueError:
        pass
    return value


def parse_overrides(args: List[str]) -> Dict[str, Any]:
    """解析命令行覆盖参数列表。

    支持格式: --key.subkey=value 或 key.subkey=value

    Args:
        args: 命令行参数列表。

    Returns:
        解析后的键值对字典。
    """
    overrides: Dict[str, Any] = {}
    for arg in args:
        stripped = arg.lstrip("-")
        if "=" not in stripped:
            continue
        key, raw_value = stripped.split("=", 1)
        overrides[key] = _auto_cast(raw_value)
    return overrides


class ConfigManager:
    """层级化配置管理器。

    支持:
    - YAML 配置文件加载
    - 配置继承 (base → experiment)
    - 命令行参数覆盖
    - 运行时配置快照保存
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ):
        """初始化配置管理器。

        Args:
            config_path: YAML 配置文件路径。为 None 时使用空字典。
            overrides: 命令行覆盖参数字典 {dotted_key: value}。
        """
        if config_path is not None:
            self.config = self._load_with_inheritance(config_path)
        else:
            self.config = {}

        if overrides:
            self._apply_overrides(overrides)

    def _load_with_inheritance(self, path: str) -> dict:
        """加载 YAML 配置文件，递归处理 inherit_from 继承链。

        Args:
            path: YAML 文件路径。

        Returns:
            合并后的配置字典。
        """
        path = os.path.abspath(path)
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        if "inherit_from" in config:
            inherit_path = config.pop("inherit_from")
            # 相对路径基于当前配置文件所在目录解析
            if not os.path.isabs(inherit_path):
                inherit_path = os.path.join(os.path.dirname(path), inherit_path)
            base = self._load_with_inheritance(inherit_path)
            config = deep_merge(base, config)

        return config

    def _apply_overrides(self, overrides: Dict[str, Any]) -> None:
        """应用命令行参数覆盖。

        Args:
            overrides: {dotted_key: value} 字典。
        """
        for key, value in overrides.items():
            set_nested(self.config, key, value)

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """获取配置值。

        Args:
            dotted_key: 点分隔的键路径，如 "ce_wm.d_model"。
            default: 键不存在时的默认值。

        Returns:
            配置值。
        """
        return get_nested(self.config, dotted_key, default)

    def save_snapshot(self, log_dir: str, filename: Optional[str] = None) -> str:
        """保存运行时配置快照至指定目录。

        Args:
            log_dir: 日志目录路径。
            filename: 快照文件名。默认为 config_snapshot.yaml。

        Returns:
            快照文件的完整路径。
        """
        os.makedirs(log_dir, exist_ok=True)
        if filename is None:
            filename = "config_snapshot.yaml"
        snapshot_path = os.path.join(log_dir, filename)
        with open(snapshot_path, "w", encoding="utf-8") as f:
            yaml.dump(
                self.config,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
        return snapshot_path

    def to_dict(self) -> dict:
        """返回配置字典的深拷贝。"""
        return copy.deepcopy(self.config)
