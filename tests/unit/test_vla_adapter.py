"""VLA adapter 单元测试。

测试 ProxyVLAAdapter 的基本功能：
- 输出 shape [B, T, 7]
- 数值在 [-1, 1] 范围
- 不同 instruction 产生不同输出
- 相同输入产生确定性输出
"""

import torch
import pytest

from src.dual_stream.vla_adapter import (
    ProxyVLAAdapter,
    OpenVLAAdapter,
    build_vla_adapter,
)


class TestProxyVLAAdapter:

    def setup_method(self):
        self.adapter = ProxyVLAAdapter(action_dim=7, chunk_size=1, device="cpu")

    def test_output_shape(self):
        obs = {
            "rgb": torch.rand(2, 3, 224, 224),
            "depth": torch.zeros(2, 1, 224, 224),
            "pose": torch.zeros(2, 7),
        }
        action = self.adapter.predict(obs, "pick up the red block")
        assert action.shape == (2, 1, 7)

    def test_output_range(self):
        obs = {
            "rgb": torch.rand(4, 3, 224, 224),
            "depth": torch.zeros(4, 1, 224, 224),
            "pose": torch.zeros(4, 7),
        }
        action = self.adapter.predict(obs, "move to the table")
        assert action.min() >= -1.0
        assert action.max() <= 1.0

    def test_different_instructions_different_output(self):
        obs = {
            "rgb": torch.rand(1, 3, 224, 224),
            "depth": torch.zeros(1, 1, 224, 224),
            "pose": torch.zeros(1, 7),
        }
        a1 = self.adapter.predict(obs, "pick up the red block")
        a2 = self.adapter.predict(obs, "push the green button")
        assert not torch.allclose(a1, a2, atol=1e-4)

    def test_deterministic(self):
        obs = {
            "rgb": torch.rand(1, 3, 224, 224),
            "depth": torch.zeros(1, 1, 224, 224),
            "pose": torch.zeros(1, 7),
        }
        a1 = self.adapter.predict(obs, "pick up the red block")
        a2 = self.adapter.predict(obs, "pick up the red block")
        assert torch.allclose(a1, a2)

    def test_different_images_different_output(self):
        obs1 = {
            "rgb": torch.zeros(1, 3, 224, 224),
            "depth": torch.zeros(1, 1, 224, 224),
            "pose": torch.zeros(1, 7),
        }
        obs2 = {
            "rgb": torch.ones(1, 3, 224, 224),
            "depth": torch.zeros(1, 1, 224, 224),
            "pose": torch.zeros(1, 7),
        }
        a1 = self.adapter.predict(obs1, "pick up the red block")
        a2 = self.adapter.predict(obs2, "pick up the red block")
        assert not torch.allclose(a1, a2, atol=1e-4)

    def test_chunk_size_greater_than_1(self):
        adapter = ProxyVLAAdapter(action_dim=7, chunk_size=4, device="cpu")
        obs = {
            "rgb": torch.rand(2, 3, 224, 224),
            "depth": torch.zeros(2, 1, 224, 224),
            "pose": torch.zeros(2, 7),
        }
        action = adapter.predict(obs, "pick up the red block")
        assert action.shape == (2, 4, 7)

    def test_frozen_parameters(self):
        for p in self.adapter.parameters():
            assert not p.requires_grad


class TestBuildVLAAdapter:

    def test_build_proxy(self):
        config = {"type": "proxy", "action_dim": 7, "chunk_size": 1, "device": "cpu"}
        adapter = build_vla_adapter(config)
        assert isinstance(adapter, ProxyVLAAdapter)

    def test_build_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown vla type"):
            build_vla_adapter({"type": "nonexistent", "device": "cpu"})


class TestOpenVLAAdapter:

    def test_prompt_template(self):
        prompt = OpenVLAAdapter._build_prompt("pick up the red block")
        assert prompt == "In: What action should the robot take to pick up the red block?\nOut:"

    def test_prompt_strips_whitespace(self):
        prompt = OpenVLAAdapter._build_prompt("  Pick Up The Block  ")
        assert prompt == "In: What action should the robot take to pick up the block?\nOut:"
