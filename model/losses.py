"""Loss terms for training BoundaryModel:
  L = lambda1 * focal(peak presence) + lambda2 * soft-argmax localization
      + lambda3 * diffusion x0-MSE (from DiffusionSelfCorrector.training_loss)
      + lambda4 * ink avoidance (don't predict a boundary through a character)

Starting weights lambda1=1.0, lambda2=0.5, lambda3=1.0, lambda4=0.5.
"""

from __future__ import annotations

import torch


def focal_bce_loss(
    pred_prob: torch.Tensor, target: torch.Tensor, gamma: float = 2.0, pos_weight: float = 15.0
) -> torch.Tensor:
    """pred_prob, target: (B, L) in [0, 1]. Continuous-target focal BCE:
    down-weights positions the model already gets right (pt close to 1),
    matching the standard focal-loss purpose of focusing on hard examples --
    here, positions near a boundary where the model is uncertain.

    pos_weight adds the class-imbalance half of standard alpha-balanced
    focal loss (the focusing term `(1-pt)^gamma` alone only down-weights
    EASY examples of either class -- with boundary targets covering under
    ~5% of positions, "always predict near-zero" is an easy negative that
    stays easy after focusing too, so without this weight the model can
    reach a deceptively low loss without ever committing to a sharp peak at
    the sparse true boundaries)."""
    eps = 1e-6
    pred_prob = pred_prob.clamp(eps, 1 - eps)
    pt = target * pred_prob + (1 - target) * (1 - pred_prob)
    focal_weight = (1 - pt) ** gamma
    alpha_weight = 1.0 + pos_weight * target
    bce = -(target * torch.log(pred_prob) + (1 - target) * torch.log(1 - pred_prob))
    return (alpha_weight * focal_weight * bce).mean()


def soft_argmax_localization_loss(
    pred_signal: torch.Tensor,
    boundaries_batch: list[list[int]],
    window_radius: int = 15,
    sharpen: float = 10.0,
) -> torch.Tensor:
    """pred_signal: (B, L) in [0, 1]. boundaries_batch[i]: interior boundary
    x-positions (ints) for example i (edges 0/L-1 excluded -- those are
    structural, not predicted). For each true boundary, take a soft-argmax
    over a local window of the predicted signal and penalize its distance
    from the true position -- this is what gives sub-pixel-ish localization
    on top of the coarser peak-presence signal from the focal loss."""
    device = pred_signal.device
    length = pred_signal.shape[-1]
    positions = torch.arange(length, device=device, dtype=torch.float32)

    all_sq_errors = []
    for i, boundaries in enumerate(boundaries_batch):
        for b in boundaries:
            lo = max(0, b - window_radius)
            hi = min(length, b + window_radius + 1)
            window_vals = pred_signal[i, lo:hi]
            window_pos = positions[lo:hi]
            weights = torch.softmax(window_vals * sharpen, dim=0)
            soft_pos = (weights * window_pos).sum()
            all_sq_errors.append((soft_pos - b) ** 2)

    if not all_sq_errors:
        return torch.zeros((), device=device)
    return torch.stack(all_sq_errors).mean()


def ink_avoidance_loss(pred_prob: torch.Tensor, energy_col: torch.Tensor) -> torch.Tensor:
    """pred_prob, energy_col: (B, L). energy_col is the per-column SDT energy
    (negative = deep inside a character's ink, ~0 at its edge, positive in
    the background gap between characters -- see
    seam_extraction.compute_normalized_energy_channel and
    dataset.py:_add_line_windows for how it's built per training window).

    Penalizes predicted boundary-presence probability sitting on ink: at a
    background column (energy >= 0) this term is exactly zero regardless of
    pred_prob, so it never fights the model for predicting real boundaries
    in the gaps between characters -- it only pushes probability mass OUT of
    columns that are actually inside a character, which is what "boundaries
    should not pass through the blacks" means directly translated into a
    gradient signal. Same shape as seam_energy_loss.energy_avoidance_loss
    (used by the curved separator models), adapted here for a dense
    per-column probability instead of a single sampled path position."""
    return (pred_prob * torch.relu(-energy_col)).mean()


def total_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    boundaries_batch: list[list[int]],
    diffusion_mse: torch.Tensor,
    energy_col: torch.Tensor | None = None,
    lambda1: float = 1.0,
    lambda2: float = 0.5,
    lambda3: float = 1.0,
    lambda4: float = 0.5,
) -> dict[str, torch.Tensor]:
    """logits: (B, L) raw SSM boundary-head output; target: (B, L) gaussian-
    bump targets in [0,1]; diffusion_mse: scalar from
    DiffusionSelfCorrector.training_loss (computed separately, since it needs
    the model's hidden features -- see train.py); energy_col: (B, L) per-
    column ink energy (see ink_avoidance_loss), or None to skip that term
    (e.g. for callers/tests that don't have it available)."""
    pred_prob = torch.sigmoid(logits)
    l_focal = focal_bce_loss(pred_prob, target)
    l_loc = soft_argmax_localization_loss(pred_prob, boundaries_batch)
    l_ink = ink_avoidance_loss(pred_prob, energy_col) if energy_col is not None else torch.zeros((), device=logits.device)
    l_total = lambda1 * l_focal + lambda2 * l_loc + lambda3 * diffusion_mse + lambda4 * l_ink
    return {
        "total": l_total,
        "focal": l_focal,
        "localization": l_loc,
        "diffusion": diffusion_mse,
        "ink": l_ink,
    }
