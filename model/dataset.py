"""Turns saved annotation JSON files into (image_window, target_bumps) training
examples. Not exercised this session (no annotations exist yet for a fresh
project) but ready to run the moment page_0002.json (or later pages) has been
through the UI.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from backend.services.seam_extraction import compute_normalized_energy_channel
from model.config import DiffusionConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"


def gaussian_bumps(boundaries_px: list[int], length: int, sigma: float) -> np.ndarray:
    """Sum of narrow Gaussian bumps centered at each interior boundary
    (edges at 0 and length-1 are structural, not real boundaries to predict)."""
    signal = np.zeros(length, dtype=np.float32)
    xs = np.arange(length, dtype=np.float32)
    interior = boundaries_px[1:-1] if len(boundaries_px) > 2 else []
    for b in interior:
        signal += np.exp(-0.5 * ((xs - b) / sigma) ** 2)
    return np.clip(signal, 0.0, 1.0)


class BoundaryWindowDataset(Dataset):
    """One example = one sliding window (width `window_px`, stride
    `stride_px`) cropped from one annotated text line, paired with a
    Gaussian-bump target signal built from that line's boundaries_px."""

    def __init__(
        self,
        page_ids: list[str],
        cfg: DiffusionConfig | None = None,
        max_lines: int | None = None,
        require_user_correction: bool = False,
    ):
        """max_lines: only use the first N lines per page (matches how the
        UI is actually being used -- a human corrects top-to-bottom, so only
        that prefix is trustworthy ground truth; a line needing zero
        corrections, e.g. a short title line, is still trustworthy as long
        as it falls within the reviewed prefix).
        require_user_correction: additionally skip any line with zero
        user-corrected boundaries. Off by default -- combine with max_lines
        instead when the trust boundary is "reviewed up to line N", since
        this flag can't distinguish "reviewed, needed no change" from
        "never reviewed"."""
        self.cfg = cfg or DiffusionConfig()
        self.examples: list[dict] = []

        for page_id in page_ids:
            ann_path = ANNOTATIONS_DIR / f"{page_id}.json"
            if not ann_path.exists():
                continue
            annotation = json.loads(ann_path.read_text(encoding="utf-8"))
            image_path = PROJECT_ROOT / annotation["image_path"]
            gray_u8 = np.array(Image.open(image_path).convert("L"))
            gray = gray_u8.astype(np.float32) / 255.0
            # SDT energy (negative = inside ink), computed once per page --
            # same signal the curved seam-generator models are conditioned
            # on (see seam_extraction.py) -- reused here purely as an
            # auxiliary training target for `ink_avoidance_loss`, not as a
            # model input, so BoundaryModel's own architecture/inference
            # stays untouched.
            energy = compute_normalized_energy_channel(gray_u8)

            lines = annotation["lines"]
            if max_lines is not None:
                lines = lines[:max_lines]

            for line in lines:
                if require_user_correction and "user" not in line["boundary_source"]:
                    continue
                self._add_line_windows(gray, energy, line)

    def _add_line_windows(self, gray: np.ndarray, energy: np.ndarray, line: dict) -> None:
        # y0/y1 can be non-integer for a split line (curved-seam bounding
        # envelope, e.g. from Math.min/max over an interpolated JS array) --
        # round to the nearest pixel row for cropping.
        y0, y1 = round(line["y0"]), round(line["y1"])
        # A line's boundaries_px[0]/[-1] are its own left/right extent -- by
        # default the whole page width, but can be trimmed inward via the
        # annotator's Scissor tool to exclude scanner noise at the page
        # margins (see frontend's trimLineEdge / getLineX0/getLineX1). Crop
        # to exactly that extent so a trimmed-off margin is structurally
        # absent from training, not merely down-weighted -- the model never
        # sees it as a training pixel at all. boundaries are reindexed
        # relative to x0 to match the now-narrower crop's own column 0.
        x0, x1 = round(line["boundaries_px"][0]), round(line["boundaries_px"][-1])
        boundaries = [b - x0 for b in line["boundaries_px"]]
        line_crop = gray[y0:y1, x0:x1]
        # Worst-case (most-negative, i.e. deepest-in-ink) energy per column
        # across the line's own height -- collapses the 2D energy crop to a
        # 1D per-column signal comparable to the model's own per-column
        # boundary-presence probability (see model/losses.py:ink_avoidance_loss).
        energy_col = energy[y0:y1, x0:x1].min(axis=0)
        line_width = line_crop.shape[1]

        window = self.cfg.window_px
        stride = self.cfg.stride_px

        start = 0
        while start < line_width:
            end = min(start + window, line_width)
            local_boundaries = [b - start for b in boundaries if start <= b <= end]
            if len(local_boundaries) >= 2:  # need at least both edges present
                self.examples.append(
                    {
                        "crop": line_crop[:, start:end],
                        "energy_col": energy_col[start:end],
                        "boundaries": local_boundaries,
                        "width": end - start,
                    }
                )
            if end == line_width:
                break
            start += stride

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        crop = torch.from_numpy(ex["crop"]).unsqueeze(0)  # (1, H, W)
        target = gaussian_bumps(ex["boundaries"], ex["width"], self.cfg.bump_sigma_px)
        target = torch.from_numpy(target)  # (W,)
        energy_col = torch.from_numpy(ex["energy_col"].astype(np.float32))  # (W,)
        return crop, target, energy_col
