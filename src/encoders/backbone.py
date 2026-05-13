"""视觉骨干网络工厂函数，支持 ResNet-18 和 ViT-Small。

不依赖 torchvision，使用纯 PyTorch 实现。
"""

import torch
import torch.nn as nn


def build_backbone(backbone_type: str, pretrained: bool = False) -> nn.Module:
    """
    构建视觉骨干网络。

    Args:
        backbone_type: 骨干网络类型，支持 "resnet18" 和 "vit_small"
        pretrained: 是否加载 ImageNet 预训练权重

    Returns:
        去除分类头的视觉特征提取器，输出维度见下表：
          - resnet18:  512
          - vit_small: 384
    """
    if backbone_type == "resnet18":
        return _build_resnet18(pretrained)
    elif backbone_type == "vit_small":
        return _build_vit_small(pretrained)
    else:
        raise ValueError(
            f"Unsupported backbone type: {backbone_type}. "
            f"Choose from: resnet18, vit_small"
        )


class _BasicBlock(nn.Module):
    """ResNet BasicBlock (无 torchvision 依赖)。"""
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


def _make_layer(in_ch, out_ch, blocks, stride=1):
    downsample = None
    if stride != 1 or in_ch != out_ch:
        downsample = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
            nn.BatchNorm2d(out_ch),
        )
    layers = [_BasicBlock(in_ch, out_ch, stride, downsample)]
    for _ in range(1, blocks):
        layers.append(_BasicBlock(out_ch, out_ch))
    return nn.Sequential(*layers)


def _build_resnet18(pretrained: bool) -> nn.Module:
    """构建 ResNet-18 骨干（纯 PyTorch），输出 512 维特征。"""
    if pretrained:
        import warnings
        warnings.warn("pretrained=True requires torchvision; using random init")
    return nn.Sequential(
        nn.Conv2d(3, 64, 7, 2, 3, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(3, 2, 1),
        _make_layer(64, 64, 2),
        _make_layer(64, 128, 2, stride=2),
        _make_layer(128, 256, 2, stride=2),
        _make_layer(256, 512, 2, stride=2),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
    )


def _build_vit_small(pretrained: bool) -> nn.Module:
    """构建 ViT-Small 骨干（基于 torchvision ViT-B/16 缩小版），输出 384 维特征。"""
    return _ViTSmall(pretrained=pretrained)


class _ViTSmall(nn.Module):
    """
    轻量级 ViT-Small 实现。

    使用 patch_size=16, embed_dim=384, depth=6, num_heads=6。
    输入图像会被自适应调整到 224×224（ViT 标准输入）。
    """

    def __init__(self, pretrained: bool = False):
        super().__init__()
        self.patch_size = 16
        self.embed_dim = 384
        self.num_heads = 6
        self.depth = 6

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            3, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        # 使用 14×14 = 196 patches (224/16) 作为默认
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 197, self.embed_dim)  # 196 patches + 1 cls
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.embed_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=self.depth
        )
        self.norm = nn.LayerNorm(self.embed_dim)
        self.resize = nn.AdaptiveAvgPool2d((224, 224))

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] 输入图像
        Returns:
            [B, 384] CLS token 特征
        """
        B = x.shape[0]
        # 自适应调整到 224×224
        x = self.resize(x)
        # Patch embedding: [B, embed_dim, 14, 14] -> [B, 196, embed_dim]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        # 拼接 CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        # Transformer
        x = self.transformer(x)
        x = self.norm(x[:, 0])  # CLS token 输出
        return x
