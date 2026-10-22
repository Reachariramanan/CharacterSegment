"""Builds training examples for LineSeparatorModel: one example per line
EDGE (top and bottom), since edges are trained independently (no forced
shared identity between adjacent lines -- see the "seam generation" design
discussion for why: this document has real, variable-size whitespace gaps
between most lines, so forcing lines to share a single boundary object would
be wrong; the actual fix for the earlier overlap bug is an ordering
constraint enforced at generation time, not shared identity here).

Each example crops a horizontal STRIP of the page (width = the line's own
boundaries_px[0]..[-1] extent -- the whole page by default, narrower if
trimmed via the annotator's Scissor tool; height = 2*margin_px) centered on
the edge's flat y0/y1 anchor, and provides:
  - the strip image crop, stacked with a normalized SDT energy channel (see
    seam_extraction.compute_normalized_energy_channel) -- a directly-computed
    "where the ink is" signal, since a tiny network trained on only a
    handful of curved examples shouldn't have to discover that from scratch
  - the true seam path in strip-local [0,1] coordinates (from top_seam_px/
    bottom_seam_px where the line was manually split, else a flat line at
    the anchor -- i.e. most examples are currently flat, a handful curved;
    see the design discussion's "data scarcity for real curves" gap)
  - a flat prior path at the strip center (0.5), matching the anchor
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


class SeamPathDataset(Dataset):
    def __init__(self, page_ids: list[str], margin_px: int = 40):
        self.margin_px = margin_px
        self.examples: list[dict] = []

        for page_id in page_ids:
            ann_path = ANNOTATIONS_DIR / f"{page_id}.json"
            if not ann_path.exists():
                continue
            annotation = json.loads(ann_path.read_text(encoding="utf-8"))
            gray_u8 = np.array(Image.open(PROJECT_ROOT / annotation["image_path"]).convert("L"))
            image = gray_u8.astype(np.float32) / 255.0
            energy = compute_normalized_energy_channel(gray_u8)  # computed once per page, not per edge
            height, width = image.shape

            for line in annotation["lines"]:
                self._add_edge(image, energy, height, width, line, "top_seam_px", "y0")
                self._add_edge(image, energy, height, width, line, "bottom_seam_px", "y1")

    def _add_edge(
        self,
        image: np.ndarray,
        energy: np.ndarray,
        page_height: int,
        width: int,
        line: dict,
        seam_key: str,
        scalar_key: str,
    ) -> None:
        anchor = line[scalar_key]
        strip_y0 = round(anchor - self.margin_px)
        strip_y1 = round(anchor + self.margin_px)
        if strip_y0 < 0 or strip_y1 > page_height:
            return  # too close to page edge for this margin -- skip rather than pad oddly
        strip_height = strip_y1 - strip_y0

        # A line's boundaries_px[0]/[-1] are its own left/right extent -- by
        # default the whole page width, but can be trimmed inward via the
        # annotator's Scissor tool to exclude scanner noise at the page
        # margins (see frontend's trimLineEdge / getLineX0/getLineX1) --
        # crop this edge's strip to match, same as BoundaryWindowDataset
        # does for the akshara-boundary model (see model/dataset.py), so the
        # two training pipelines agree on what's actually "in" the line
        # instead of the seam always spanning edge-to-edge regardless.
        x0, x1 = round(line["boundaries_px"][0]), round(line["boundaries_px"][-1])
        strip_width = x1 - x0

        if seam_key in line:
            true_y = np.asarray(line[seam_key][x0:x1], dtype=np.float32)
        else:
            true_y = np.full(strip_width, float(anchor), dtype=np.float32)

        y1_norm = (true_y - strip_y0) / strip_height
        if y1_norm.min() < -0.5 or y1_norm.max() > 1.5:
            return  # curve deviates further than this margin can represent -- skip (needs a bigger margin)

        strip = np.stack(
            [image[strip_y0:strip_y1, x0:x1], energy[strip_y0:strip_y1, x0:x1]], axis=0
        )  # (2, H, W_line)
        self.examples.append(
            {
                "strip": strip,
                "y1_norm": np.clip(y1_norm, 0.0, 1.0),
                "strip_y0_global": strip_y0,
                "strip_height": strip_height,
            }
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        strip = torch.from_numpy(ex["strip"])  # (2, H_strip, W)
        y1_norm = torch.from_numpy(ex["y1_norm"])  # (W,)
        y0_norm = torch.full_like(y1_norm, 0.5)  # flat prior at strip center
        return strip, y1_norm, y0_norm
