"""CLI: print BoundaryModel's in-memory state_dict size against the budget.

Usage:
    python -m scripts.verify_model_size
"""

from __future__ import annotations

from model.boundary_model import BoundaryModel
from model.config import ModelConfig


def main() -> None:
    cfg = ModelConfig()
    model = BoundaryModel(cfg)
    size_bytes = sum(t.numel() * t.element_size() for t in model.state_dict().values())
    size_mb = size_bytes / (1024 * 1024)
    budget_mb = cfg.size_budget_bytes / (1024 * 1024)

    n_params = sum(t.numel() for t in model.parameters())
    print(f"Total parameters: {n_params:,}")
    print(f"State dict size:  {size_mb:.2f} MB")
    print(f"Budget:           {budget_mb:.0f} MB")
    print(f"Status:           {'OK' if size_bytes < cfg.size_budget_bytes else 'OVER BUDGET'}")


if __name__ == "__main__":
    main()
