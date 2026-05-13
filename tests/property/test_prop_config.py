"""配置管理属性测试。

Feature: ce-ais-framework
Property 13: 配置继承覆盖正确性
Property 3: 配置驱动组件实例化
"""

import os
import tempfile

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from src.config.config_manager import ConfigManager, deep_merge, parse_overrides, set_nested
from src.config.schema import CEWMConfig, EncoderConfig


# ======================================================================
# Property 13: 配置继承覆盖正确性
# ======================================================================


# --- 策略定义 ---

# 生成任意嵌套配置字典（深度 ≤ 3）
leaf_values = st.one_of(
    st.integers(min_value=-1000, max_value=1000),
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N"))),
    st.booleans(),
)

config_dict_strategy = st.recursive(
    st.dictionaries(
        keys=st.text(min_size=1, max_size=8, alphabet="abcdefghijklmnopqrstuvwxyz"),
        values=leaf_values,
        min_size=1,
        max_size=5,
    ),
    lambda children: st.dictionaries(
        keys=st.text(min_size=1, max_size=8, alphabet="abcdefghijklmnopqrstuvwxyz"),
        values=st.one_of(children, leaf_values),
        min_size=1,
        max_size=5,
    ),
    max_leaves=20,
)


def _all_keys(d: dict, prefix: str = "") -> set:
    """递归收集字典中所有叶子键路径。"""
    keys = set()
    for k, v in d.items():
        full = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            keys.update(_all_keys(v, full))
        else:
            keys.add(full)
    return keys


def _get_by_path(d: dict, path: str):
    """通过点分隔路径获取值。"""
    parts = path.split(".")
    cur = d
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


@settings(max_examples=100)
@given(base=config_dict_strategy, override=config_dict_strategy)
def test_deep_merge_preserves_base_keys(base, override):
    """合并后应包含基础配置所有键。

    Feature: ce-ais-framework, Property 13: 配置继承覆盖正确性
    """
    merged = deep_merge(base, override)

    base_keys = _all_keys(base)
    merged_keys = _all_keys(merged)

    # 基础配置的所有键路径应在合并结果中存在
    for key in base_keys:
        assert key in merged_keys or any(
            key.startswith(mk + ".") or mk.startswith(key + ".") for mk in merged_keys
        ), f"Base key '{key}' missing from merged config"


@settings(max_examples=100)
@given(base=config_dict_strategy, override=config_dict_strategy)
def test_deep_merge_override_takes_precedence(base, override):
    """子配置显式指定的键值应覆盖基础配置。

    Feature: ce-ais-framework, Property 13: 配置继承覆盖正确性
    """
    merged = deep_merge(base, override)

    override_leaf_keys = _all_keys(override)
    for key in override_leaf_keys:
        expected = _get_by_path(override, key)
        actual = _get_by_path(merged, key)
        assert actual == expected, (
            f"Override key '{key}': expected {expected}, got {actual}"
        )


# --- 命令行覆盖测试 ---

@settings(max_examples=100)
@given(
    key_parts=st.lists(
        st.text(min_size=1, max_size=6, alphabet="abcdefghijklmnopqrstuvwxyz"),
        min_size=1,
        max_size=4,
    ),
    value=st.one_of(
        st.integers(min_value=-100, max_value=100),
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        st.text(min_size=1, max_size=10, alphabet="abcdefghijklmnopqrstuvwxyz"),
    ),
)
def test_cli_override_updates_nested_config(key_parts, value):
    """命令行覆盖参数 --key.subkey=value 应正确更新嵌套配置。

    Feature: ce-ais-framework, Property 13: 配置继承覆盖正确性
    """
    dotted_key = ".".join(key_parts)
    config = {}
    set_nested(config, dotted_key, value)

    # 验证值已正确设置
    current = config
    for part in key_parts[:-1]:
        assert part in current
        current = current[part]
    assert current[key_parts[-1]] == value


@settings(max_examples=100)
@given(
    key_parts=st.lists(
        st.text(min_size=1, max_size=6, alphabet="abcdefghijklmnopqrstuvwxyz"),
        min_size=1,
        max_size=3,
    ),
    value=st.one_of(
        st.integers(min_value=-100, max_value=100),
        st.just("true"),
        st.just("false"),
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False).map(str),
    ),
)
def test_parse_overrides_format(key_parts, value):
    """parse_overrides 应正确解析 --key.subkey=value 格式。

    Feature: ce-ais-framework, Property 13: 配置继承覆盖正确性
    """
    dotted_key = ".".join(key_parts)
    arg = f"--{dotted_key}={value}"
    result = parse_overrides([arg])
    assert dotted_key in result


@settings(max_examples=100)
@given(base=config_dict_strategy, override=config_dict_strategy)
def test_yaml_inheritance_via_files(base, override):
    """通过 YAML 文件的 inherit_from 继承应正确合并。

    Feature: ce-ais-framework, Property 13: 配置继承覆盖正确性
    """
    import yaml

    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = os.path.join(tmpdir, "base.yaml")
        child_path = os.path.join(tmpdir, "child.yaml")

        with open(base_path, "w") as f:
            yaml.dump(base, f)

        child_with_inherit = dict(override)
        child_with_inherit["inherit_from"] = "base.yaml"
        with open(child_path, "w") as f:
            yaml.dump(child_with_inherit, f)

        mgr = ConfigManager(config_path=child_path)
        merged = mgr.to_dict()

        # override 中的叶子键应覆盖 base
        for key in _all_keys(override):
            expected = _get_by_path(override, key)
            actual = _get_by_path(merged, key)
            assert actual == expected, (
                f"Inherited key '{key}': expected {expected}, got {actual}"
            )


# ======================================================================
# Property 3: 配置驱动组件实例化
# ======================================================================


@settings(max_examples=100, deadline=None)
@given(
    latent_dim=st.sampled_from([32, 64, 128, 256]),
)
def test_encoder_instantiation_from_config(latent_dim):
    """对于不同 latent_dim，编码器应能成功实例化且前向传播不抛异常。

    Feature: ce-ais-framework, Property 3: 配置驱动组件实例化
    """
    from src.encoders.contrastive_encoder import ContrastiveEncoder

    config = EncoderConfig(
        backbone_type="resnet18",
        pose_dim=7,
        visual_dim=512,
        latent_dim=latent_dim,
        temperature=0.07,
        image_size=[64, 64],
    )
    encoder = ContrastiveEncoder(config)
    encoder.eval()

    B = 2
    rgb = torch.randn(B, 3, 64, 64)
    depth = torch.randn(B, 1, 64, 64)
    pose = torch.randn(B, 7)

    with torch.no_grad():
        z = encoder(rgb, depth, pose)

    assert z.shape == (B, latent_dim)
    assert torch.isfinite(z).all()


@settings(max_examples=100, deadline=None)
@given(
    latent_dim=st.sampled_from([32, 64, 128]),
    d_model=st.sampled_from([64, 128]),
    n_layers=st.sampled_from([1, 2]),
)
def test_cewm_instantiation_from_config(latent_dim, d_model, n_layers):
    """对于不同配置参数，CE-WM 应能成功实例化且前向传播不抛异常。

    Feature: ce-ais-framework, Property 3: 配置驱动组件实例化
    """
    from src.world_model.ce_wm import CausalEnergyWorldModel

    config = CEWMConfig(
        d_model=d_model,
        d_state=16,
        n_layers=n_layers,
        expand_factor=2,
        mimo_groups=2,
        action_dim=7,
        latent_dim=latent_dim,
        dropout=0.1,
    )
    model = CausalEnergyWorldModel(config)
    model.eval()

    B, T = 2, 4
    z_seq = torch.randn(B, T, latent_dim)
    a_seq = torch.randn(B, T, 7)

    with torch.no_grad():
        energy = model(z_seq, a_seq)

    assert energy.shape == (B,)
    assert torch.isfinite(energy).all()
