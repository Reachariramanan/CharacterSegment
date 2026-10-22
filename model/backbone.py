"""SlimResNet18: a ResNet-18-topology backbone at 1/4 stock channel width,
with NO spatial downsampling (stride 1 throughout -- see BackboneConfig
docstring for why). 4 stages x 2 BasicBlocks, mirroring stock ResNet-18
depth, channels (16, 32, 64, 128) instead of (64, 128, 256, 512).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.config import BackboneConfig


class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: tuple[int, int] = (1, 1)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = None
        if in_ch != out_ch or stride != (1, 1):
            self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.shortcut is None else self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class SlimResNet18(nn.Module):
    def __init__(self, cfg: BackboneConfig | None = None):
        super().__init__()
        cfg = cfg or BackboneConfig()
        c1, c2, c3, c4 = cfg.stage_channels

        self.stem = nn.Sequential(
            nn.Conv2d(cfg.in_channels, c1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )

        self.stage1 = self._make_stage(c1, c1, cfg.blocks_per_stage)
        self.stage2 = self._make_stage(c1, c2, cfg.blocks_per_stage)
        self.stage3 = self._make_stage(c2, c3, cfg.blocks_per_stage)
        self.stage4 = self._make_stage(c3, c4, cfg.blocks_per_stage)

        self.out_channels = c4

    @staticmethod
    def _make_stage(in_ch: int, out_ch: int, n_blocks: int) -> nn.Sequential:
        blocks = [BasicBlock(in_ch, out_ch)]
        for _ in range(n_blocks - 1):
            blocks.append(BasicBlock(out_ch, out_ch))
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_channels, H, W) -> (B, out_channels, H, W). Spatial dims unchanged."""
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x
