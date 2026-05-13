"""编码器单元测试：backbone + ContrastiveEncoder。"""

import pytest
import torch

from src.config.schema import EncoderConfig
from src.encoders.backbone import build_backbone
from src.encoders.contrastive_encoder import ContrastiveEncoder


# ── backbone 工厂函数 ──────────────────────────────────────────


class TestBuildBackbone:
    def test_resnet18_output_shape(self):
        backbone = build_backbone("resnet18")
        x = torch.randn(2, 3, 200, 200)
        out = backbone(x)
        assert out.shape == (2, 512)

    def test_vit_small_output_shape(self):
        backbone = build_backbone("vit_small")
        x = torch.randn(2, 3, 200, 200)
        out = backbone(x)
        assert out.shape == (2, 384)

    def test_unsupported_backbone_raises(self):
        with pytest.raises(ValueError, match="Unsupported backbone type"):
            build_backbone("mobilenet")


# ── ContrastiveEncoder ─────────────────────────────────────────


class TestContrastiveEncoder:
    @pytest.fixture
    def default_encoder(self):
        config = EncoderConfig(backbone_type="resnet18", latent_dim=128)
        return ContrastiveEncoder(config)

    def test_forward_output_shape(self, default_encoder):
        B = 4
        rgb = torch.randn(B, 3, 200, 200)
        depth = torch.randn(B, 1, 200, 200)
        pose = torch.randn(B, 7)
        z = default_encoder(rgb, depth, pose)
        assert z.shape == (B, 128)

    def test_output_l2_normalized(self, default_encoder):
        rgb = torch.randn(2, 3, 200, 200)
        depth = torch.randn(2, 1, 200, 200)
        pose = torch.randn(2, 7)
        z = default_encoder(rgb, depth, pose)
        norms = z.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_encode_interface(self, default_encoder):
        obs = {
            "rgb": torch.randn(2, 3, 200, 200),
            "depth": torch.randn(2, 1, 200, 200),
            "pose": torch.randn(2, 7),
        }
        z = default_encoder.encode(obs)
        assert z.shape == (2, 128)

    def test_vit_small_backbone(self):
        config = EncoderConfig(backbone_type="vit_small", latent_dim=64)
        encoder = ContrastiveEncoder(config)
        rgb = torch.randn(2, 3, 200, 200)
        depth = torch.randn(2, 1, 200, 200)
        pose = torch.randn(2, 7)
        z = encoder(rgb, depth, pose)
        assert z.shape == (2, 64)

    def test_different_image_resolution(self):
        config = EncoderConfig(
            backbone_type="resnet18",
            latent_dim=128,
            image_size=[128, 128],
        )
        encoder = ContrastiveEncoder(config)
        rgb = torch.randn(2, 3, 128, 128)
        depth = torch.randn(2, 1, 128, 128)
        pose = torch.randn(2, 7)
        z = encoder(rgb, depth, pose)
        assert z.shape == (2, 128)

    def test_different_pose_dim(self):
        config = EncoderConfig(
            backbone_type="resnet18",
            pose_dim=14,
            latent_dim=128,
        )
        encoder = ContrastiveEncoder(config)
        rgb = torch.randn(2, 3, 200, 200)
        depth = torch.randn(2, 1, 200, 200)
        pose = torch.randn(2, 14)
        z = encoder(rgb, depth, pose)
        assert z.shape == (2, 128)
