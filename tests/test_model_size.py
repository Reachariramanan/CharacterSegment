"""Verifies the BoundaryModel's in-memory state_dict size stays within the
88MB budget, and does a basic forward-pass shape/sanity check. No disk
writes -- everything happens in memory (random init, no trained weights)."""

from __future__ import annotations

import torch

from model.boundary_model import BoundaryModel
from model.config import ModelConfig


def _state_dict_size_bytes(model: torch.nn.Module) -> int:
    return sum(t.numel() * t.element_size() for t in model.state_dict().values())


def test_model_size_under_budget():
    cfg = ModelConfig()
    model = BoundaryModel(cfg)
    size_bytes = _state_dict_size_bytes(model)
    size_mb = size_bytes / (1024 * 1024)
    print(f"BoundaryModel state_dict size: {size_mb:.2f} MB (budget: {cfg.size_budget_bytes / (1024*1024):.0f} MB)")
    assert size_bytes < cfg.size_budget_bytes, f"{size_mb:.2f} MB exceeds budget"


def test_forward_pass_shapes():
    model = BoundaryModel()
    model.eval()
    x = torch.randn(2, 1, 32, 128)  # (B, 1, H, W) -- 2 line crops, 32px tall, 128px wide
    out = model(x)
    assert out["logits"].shape == (2, 128)
    assert out["refined"].shape == (2, 128)
    assert out["hidden"].shape == (2, model.cfg.ssm.d_model, 128)


def test_predict_boundaries_wide_line():
    model = BoundaryModel()
    model.eval()
    x = torch.randn(1, 1, 32, 600)  # wider than one diffusion window (256px) -> sliding window path
    boundaries = model.predict_boundaries(x)
    assert boundaries[0] == 0
    assert boundaries[-1] == 599
    assert boundaries == sorted(boundaries)


if __name__ == "__main__":
    test_model_size_under_budget()
    test_forward_pass_shapes()
    test_predict_boundaries_wide_line()
    print("All checks passed.")
