"""LineSlimResNet18: same topology/BasicBlock as SlimResNet18, but
downsamples width (stride (1,2)) at the stem and at the first block of
stages 2-4, while leaving height (the page's y-axis -- where line boundaries
live) untouched. See LineBackboneConfig for why this differs from the
akshara backbone's fully resolution-preserving design.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.backbone import BasicBlock
from model.config import LineBackboneConfig


class LineSlimResNet18(nn.Module):
    def __init__(self, cfg: LineBackboneConfig | None = None):
        super().__init__()
        cfg = cfg or LineBackboneConfig()
        c1, c2, c3, c4 = cfg.stage_channels
        w_stride = (1, cfg.width_downsample_per_stage)

        self.stem = nn.Sequential(
            nn.Conv2d(cfg.in_channels, c1, kernel_size=7, stride=w_stride, padding=3, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )

        self.stage1 = self._make_stage(c1, c1, cfg.blocks_per_stage, stride=(1, 1))
        self.stage2 = self._make_stage(c1, c2, cfg.blocks_per_stage, stride=w_stride)
        self.stage3 = self._make_stage(c2, c3, cfg.blocks_per_stage, stride=w_stride)
        self.stage4 = self._make_stage(c3, c4, cfg.blocks_per_stage, stride=w_stride)

        self.out_channels = c4

    @staticmethod
    def _make_stage(in_ch: int, out_ch: int, n_blocks: int, stride: tuple[int, int]) -> nn.Sequential:
        blocks = [BasicBlock(in_ch, out_ch, stride=stride)]
        for _ in range(n_blocks - 1):
            blocks.append(BasicBlock(out_ch, out_ch))
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_channels, H, W) -> (B, out_channels, H, W/16). Height unchanged."""
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x
