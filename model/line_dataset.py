"""Builds training examples for LineSegmentationModel from annotation JSON.

Only lines up to `max_lines` are trusted (matches how the annotation UI is
actually being used: a human corrects akshara boundaries line-by-line, top
to bottom, so only the already-corrected prefix is reliable ground truth).
The page is CROPPED to just past the last trusted line's y1, so the
untrusted/unverified remainder of the page never enters the target mask as
a false "not a line" region.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"


def _line_coverage_per_row(line: dict, height: int, width: int) -> np.ndarray:
    """Fraction of the line's width that covers each row y, in [0,1].

    For a straight (non-split) line this is just 1.0 for y0<=y<=y1 and 0
    elsewhere -- identical to the old flat-rectangle target. For a SPLIT
    line (top_seam_px/bottom_seam_px present, one y-value per column) this
    is column-aware: row y counts as covered only at the columns where the
    line's own curve actually reaches it.

    This matters because a split line's flat bounding envelope (y0=min(top
    curve), y1=max(bottom curve)) can legitimately OVERLAP a neighboring
    split line's envelope even when the real per-column curves never cross
    -- e.g. curve A dips low on the left, curve B (the line below A) rises
    high on the right, so min(A)..max(A) and min(B)..max(B) overlap in y
    even though at any given column A is strictly above B. Building the
    target from the flat envelope was writing target=1 for BOTH lines
    across their whole overlapping y-range, erasing the real gap between
    them entirely -- diagnosed directly: several split-line pairs on this
    page had negative "gaps" (overlapping envelopes) once inspected, and the
    line-segmentation model trained on that target collapsed those whole
    regions into one line, no matter how many epochs it got, because there
    was no dip left in the target signal to learn."""
    if "top_seam_px" in line and "bottom_seam_px" in line:
        top = np.asarray(line["top_seam_px"], dtype=np.float32)
        bottom = np.asarray(line["bottom_seam_px"], dtype=np.float32)
        ys = np.arange(height, dtype=np.float32)[:, None]  # (height, 1)
        covered = (ys >= top[None, :]) & (ys <= bottom[None, :])  # (height, width)
        return covered.mean(axis=1).astype(np.float32)

    coverage = np.zeros(height, dtype=np.float32)
    y0, y1 = round(line["y0"]), round(line["y1"])
    coverage[y0 : y1 + 1] = 1.0
    return coverage


class LineSegmentationDataset(Dataset):
    def __init__(
        self,
        page_ids: list[str],
        max_lines: int | None = None,
        edge_sigma_px: float = 0.5,
        crop_margin_px: int = 20,
    ):
        # edge_sigma_px was originally 2.0, which seemed like a reasonable
        # "soft edge" smoothing choice in isolation, but several real
        # inter-line gaps on this document are only ~2-3px tall -- smoothing
        # with sigma=2.0 destroys the dip almost entirely (down to only
        # ~0.62 at the gap center for a true 2px gap, never near 0), so the
        # model was faithfully learning a target that had already erased the
        # very signal it needed to detect tight gaps -- diagnosed by
        # comparing the trained model's raw output against this target
        # construction directly. sigma=0.5 keeps a true 2px gap's target
        # down near ~0.11, comfortably separable from a 0.5 threshold.
        self.examples: list[dict] = []

        for page_id in page_ids:
            ann_path = ANNOTATIONS_DIR / f"{page_id}.json"
            if not ann_path.exists():
                continue
            annotation = json.loads(ann_path.read_text(encoding="utf-8"))
            image = np.array(Image.open(PROJECT_ROOT / annotation["image_path"]).convert("L"))
            image = image.astype(np.float32) / 255.0

            lines = annotation["lines"]
            if max_lines is not None:
                lines = lines[:max_lines]
            if not lines:
                continue

            # y0/y1 can be non-integer for a split line (curved-seam bounding
            # envelope) -- round to the nearest pixel row throughout.
            y_end = round(min(image.shape[0], lines[-1]["y1"] + crop_margin_px))
            crop = image[:y_end, :]

            width = image.shape[1]
            target = np.zeros(y_end, dtype=np.float32)
            edges: list[int] = []
            for line in lines:
                coverage = _line_coverage_per_row(line, y_end, width)
                target = np.maximum(target, coverage)  # max, not sum -- overlapping
                # coverage from two lines should never exceed "covered", not stack past 1.0
                edges.extend([round(line["y0"]), round(line["y1"])])
            target = gaussian_filter1d(target, sigma=edge_sigma_px)
            target = np.clip(target, 0.0, 1.0)

            self.examples.append({"crop": crop, "target": target, "edges": sorted(edges)})

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        crop = torch.from_numpy(ex["crop"]).unsqueeze(0)  # (1, H, W)
        target = torch.from_numpy(ex["target"])  # (H,)
        return crop, target, ex["edges"]
