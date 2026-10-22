"""Dataclasses defining the BoundaryModel's dimensions. Kept separate from the
module code so the size budget is visible/tunable in one place.

Size budget (fp32, verified by tests/test_model_size.py):
  SlimResNet18 backbone   ~2.79 MB
  Row-collapse pool       ~0
  S4D-lite SSM stack      ~1.06 MB
  Boundary head           ~0
  Diffusion self-correct  ~0.70 MB
  ------------------------------
  Total                   ~4.55 MB   (budget ceiling: 88 MB; ~19x headroom)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BackboneConfig:
    in_channels: int = 1  # grayscale
    stage_channels: tuple[int, int, int, int] = (16, 32, 64, 128)
    blocks_per_stage: int = 2
    # Deliberate deviation from stock ResNet-18: NO spatial downsampling
    # anywhere (stride 1 throughout, "same" padding). Segments can be as
    # narrow as ~12px, so any downsampling would blur boundary localization;
    # keeping the feature map at input resolution means the 1D sequence fed
    # to the SSM maps 1:1 to pixel columns -- boundary logits and the
    # gaussian-bump training targets live in the same coordinate system with
    # no upsampling/interpolation step needed.
    preserve_resolution: bool = True


@dataclass
class SSMConfig:
    d_model: int = 128     # must match backbone's final stage channels
    d_state: int = 64
    n_layers: int = 4
    ffn_expand: int = 2    # GLU FFN hidden = d_model * ffn_expand * 2 (gate+value)


@dataclass
class DiffusionConfig:
    window_px: int = 256
    stride_px: int = 128   # 50% overlap
    n_steps: int = 4       # T
    hidden_channels: int = 128
    kernel_size: int = 5
    dilations: tuple[int, int, int] = (1, 2, 4)
    timestep_embed_dim: int = 16
    bump_sigma_px: float = 1.5  # gaussian-bump target width for training


@dataclass
class ModelConfig:
    backbone: BackboneConfig = None
    ssm: SSMConfig = None
    diffusion: DiffusionConfig = None
    size_budget_bytes: int = 88 * 1024 * 1024

    def __post_init__(self):
        if self.backbone is None:
            self.backbone = BackboneConfig()
        if self.ssm is None:
            self.ssm = SSMConfig()
        if self.diffusion is None:
            self.diffusion = DiffusionConfig()


@dataclass
class LineBackboneConfig:
    """Backbone for page-level LINE segmentation (y-axis), as opposed to
    BackboneConfig which is for per-line AKSHARA segmentation (x-axis).

    Here the axis of interest (page height / y-position of line boundaries)
    is preserved at full resolution (stride 1), but the axis that gets
    collapsed away (page width) is aggressively downsampled -- unlike the
    akshara backbone, this collapse axis is NOT small (~900px, a whole page
    width vs. a ~20-45px line height), so leaving it undownsampled would
    make full-page feature maps too large for a 4GB-VRAM budget. 4 stride-
    (1,2) downsampling stages give a 16x reduction along width only.
    """
    in_channels: int = 1
    stage_channels: tuple[int, int, int, int] = (16, 32, 64, 128)
    blocks_per_stage: int = 2
    width_downsample_per_stage: int = 2  # applied at stem + first block of stages 2-4


@dataclass
class LineModelConfig:
    line_backbone: LineBackboneConfig = None
    ssm: SSMConfig = None
    diffusion: DiffusionConfig = None
    size_budget_bytes: int = 88 * 1024 * 1024

    def __post_init__(self):
        if self.line_backbone is None:
            self.line_backbone = LineBackboneConfig()
        if self.ssm is None:
            self.ssm = SSMConfig()
        if self.diffusion is None:
            # Line gaps are wider/coarser than akshara gaps; a full page
            # height (~1300px) is small enough for the FFT-based SSM+
            # diffusion to process in one shot, so no sliding window is
            # needed here (window_px set larger than any expected page).
            self.diffusion = DiffusionConfig(
                window_px=4096, stride_px=4096, bump_sigma_px=3.0
            )
