"""序列化 round-trip 属性测试。

Feature: ce-ais-framework
Property 4: 序列化 round-trip
"""

import os
import tempfile

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from src.config.schema import CEWMConfig, EncoderConfig
from src.utils.checkpoint import CheckpointManager


# --- 策略定义 ---

leaf_values = st.one_of(
    st.integers(min_value=-1000, max_value=1000),
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N"))),
    st.booleans(),
)

config_dict_strategy = st.dictionaries(
    keys=st.text(min_size=1, max_size=8, alphabet="abcdefghijklmnopqrstuvwxyz"),
    values=st.one_of(
        leaf_values,
        st.dictionaries(
            keys=st.text(min_size=1, max_size=8, alphabet="abcdefghijklmnopqrstuvwxyz"),
            values=leaf_values,
            min_size=0,
            max_size=5,
        ),
    ),
    min_size=1,
    max_size=10,
)


# ======================================================================
# Property 4: 序列化 round-trip — 模型状态
# ======================================================================


@settings(max_examples=100, deadline=None)
@given(
    epoch=st.integers(min_value=0, max_value=1000),
    global_step=st.integers(min_value=0, max_value=100000),
    latent_dim=st.sampled_from([32, 64, 128]),
)
def test_encoder_state_roundtrip(epoch, global_step, latent_dim):
    """编码器模型状态保存后再加载应产生完全等价的对象。

    Feature: ce-ais-framework, Property 4: 序列化 round-trip
    """
    from src.encoders.contrastive_encoder import ContrastiveEncoder

    config = EncoderConfig(
        backbone_type="resnet18",
        latent_dim=latent_dim,
        image_size=[64, 64],
    )

    model = ContrastiveEncoder(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 执行一步前向+反向以产生非零优化器状态
    rgb = torch.randn(2, 3, 64, 64)
    depth = torch.randn(2, 1, 64, 64)
    pose = torch.randn(2, 7)
    z = model(rgb, depth, pose)
    z.sum().backward()
    optimizer.step()

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_mgr = CheckpointManager(checkpoint_dir=tmpdir, prefix="enc")
        ckpt_mgr.save(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            global_step=global_step,
        )

        # 创建新模型并加载
        model2 = ContrastiveEncoder(config)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)

        latest = ckpt_mgr.find_latest()
        assert latest is not None

        state = ckpt_mgr.load(
            latest, model2, optimizer=optimizer2, map_location="cpu"
        )

        assert state["epoch"] == epoch
        assert state["global_step"] == global_step

        # 验证模型参数完全一致
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), model2.named_parameters()
        ):
            assert n1 == n2
            assert torch.equal(p1.data, p2.data), f"Parameter {n1} mismatch"

        # 验证优化器状态一致
        sd1 = optimizer.state_dict()
        sd2 = optimizer2.state_dict()
        assert len(sd1["param_groups"]) == len(sd2["param_groups"])


@settings(max_examples=100, deadline=None)
@given(
    epoch=st.integers(min_value=0, max_value=1000),
    global_step=st.integers(min_value=0, max_value=100000),
)
def test_cewm_state_roundtrip(epoch, global_step):
    """CE-WM 模型状态保存后再加载应产生完全等价的对象。

    Feature: ce-ais-framework, Property 4: 序列化 round-trip
    """
    from src.world_model.ce_wm import CausalEnergyWorldModel

    config = CEWMConfig(
        d_model=64, d_state=16, n_layers=2, expand_factor=2,
        mimo_groups=2, action_dim=7, latent_dim=32, dropout=0.1,
    )
    model = CausalEnergyWorldModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 一步前向+反向
    z_seq = torch.randn(2, 4, 32)
    a_seq = torch.randn(2, 4, 7)
    energy = model(z_seq, a_seq)
    energy.sum().backward()
    optimizer.step()

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_mgr = CheckpointManager(checkpoint_dir=tmpdir, prefix="cewm")
        ckpt_mgr.save(
            epoch=epoch, model=model, optimizer=optimizer,
            global_step=global_step,
        )

        model2 = CausalEnergyWorldModel(config)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)

        latest = ckpt_mgr.find_latest()
        state = ckpt_mgr.load(
            latest, model2, optimizer=optimizer2, map_location="cpu"
        )

        assert state["epoch"] == epoch
        assert state["global_step"] == global_step

        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), model2.named_parameters()
        ):
            assert torch.equal(p1.data, p2.data), f"Parameter {n1} mismatch"


# ======================================================================
# Property 4: 序列化 round-trip — 配置字典
# ======================================================================


@settings(max_examples=100)
@given(config_data=config_dict_strategy)
def test_config_dict_roundtrip(config_data):
    """配置字典保存后再加载应键值完全一致。

    Feature: ce-ais-framework, Property 4: 序列化 round-trip
    """
    import yaml

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "config.yaml")

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)

        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)

        assert loaded == config_data, (
            f"Config roundtrip mismatch:\n  original: {config_data}\n  loaded: {loaded}"
        )
