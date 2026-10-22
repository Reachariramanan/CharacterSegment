"""Builds training examples for CharacterSeparatorModel: one example per
INTERIOR akshara boundary (excludes the two structural line edges).

There is currently no curved ground truth for character separators -- the
vertical split-line editor only supports straight cuts -- so every
example's target path is a flat vertical line at the boundary's x-position,
same as how LINE separators started before any curved split-line data
existed. This is intentional: the model/training infra is built now so it's
ready to learn real curvature the moment the annotation tool grows a
curved-boundary mode for characters too (see the "seam generation" design
discussion).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from backend.services.seam_extraction import compute_normalized_energy_channel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"


class CharacterSeamDataset(Dataset):
    def __init__(self, page_ids: list[str], margin_px: int = 15):
        self.margin_px = margin_px
        self.examples: list[dict] = []

        for page_id in page_ids:
            ann_path = ANNOTATIONS_DIR / f"{page_id}.json"
            if not ann_path.exists():
                continue
            annotation = json.loads(ann_path.read_text(encoding="utf-8"))
            gray_u8 = np.array(Image.open(PROJECT_ROOT / annotation["image_path"]).convert("L"))
            image = gray_u8.astype(np.float32) / 255.0
            energy = compute_normalized_energy_channel(gray_u8)  # once per page
            page_width = image.shape[1]

            for line in annotation["lines"]:
                y0, y1 = round(line["y0"]), round(line["y1"])
                if y1 <= y0:
                    continue
                boundaries = line["boundaries_px"]
                for bx in boundaries[1:-1]:  # interior only -- skip structural edges
                    self._add_boundary(image, energy, page_width, y0, y1, bx)

    def _add_boundary(
        self,
        image: np.ndarray,
        energy: np.ndarray,
        page_width: int,
        y0: int,
        y1: int,
        anchor: float,
    ) -> None:
        strip_x0 = round(anchor - self.margin_px)
        strip_x1 = round(anchor + self.margin_px)
        if strip_x0 < 0 or strip_x1 > page_width:
            return  # too close to page edge for this margin -- skip
        strip_width = strip_x1 - strip_x0
        line_height = y1 - y0

        strip = np.stack(
            [image[y0:y1, strip_x0:strip_x1], energy[y0:y1, strip_x0:strip_x1]], axis=0
        )  # (2, H, W_strip)

        x1_norm = np.full(line_height, 0.5, dtype=np.float32)  # flat -- no curved ground truth yet
        self.examples.append(
            {
                "strip": strip,
                "x1_norm": x1_norm,
                "strip_x0_global": strip_x0,
                "strip_width": strip_width,
            }
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        strip = torch.from_numpy(ex["strip"])  # (2, H, W_strip)
        x1_norm = torch.from_numpy(ex["x1_norm"])  # (H,)
        x0_norm = torch.full_like(x1_norm, 0.5)  # flat prior at strip center
        return strip, x1_norm, x0_norm
