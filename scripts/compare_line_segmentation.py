"""Compares three line-segmentation approaches on the same page, so their
performance can be judged side by side:

  1. cv_assist   -- classical horizontal projection-profile line detector
  2. seam         -- minimum-energy-seam DP (Eq. 4, SDT), windowed around
                     the cv_assist anchors (see seam_extraction.py's
                     docstring for why the unconstrained global version
                     drifts badly on this document's table layout)
  3. model        -- the trained LineSegmentationModel checkpoint

Produces one overlay PNG with each method's line bands in a different
color/style, plus a text report of each method's (y0, y1) per line so
differences are easy to read off directly.

Usage:
    python -m scripts.compare_line_segmentation --page page_0002 --max-lines 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from backend.services.cv_assist import CvAssistConfig, _binarize_sauvola, _detect_text_lines, propose_boundaries
from backend.services.seam_extraction import extract_line_bands_windowed
from model.config import LineModelConfig
from model.line_segmenter import LineSegmentationModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "data" / "images"
DEBUG_DIR = IMAGES_DIR / "debug"
CHECKPOINT_PATH = PROJECT_ROOT / "model" / "checkpoints" / "line_segmenter_pageonly.pt"


def run_cv_assist(gray: np.ndarray, max_lines: int | None) -> list[tuple[int, int]]:
    proposals = propose_boundaries(gray, CvAssistConfig())
    bands = [(p.y0, p.y1) for p in proposals]
    return bands[:max_lines] if max_lines else bands


def run_model(gray: np.ndarray, crop_height: int) -> list[tuple[int, int]]:
    if not CHECKPOINT_PATH.exists():
        print(f"[model] no checkpoint at {CHECKPOINT_PATH}, skipping")
        return []
    model = LineSegmentationModel(LineModelConfig())
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
    model.eval()

    crop = gray[:crop_height, :].astype(np.float32) / 255.0
    x = torch.from_numpy(crop).unsqueeze(0).unsqueeze(0)
    bands = model.predict_line_bands(x)
    return bands


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page", type=str, default="page_0002")
    parser.add_argument("--max-lines", type=int, default=10, help="restrict comparison to the first N cv_assist-detected lines")
    parser.add_argument("--seam-window-margin", type=int, default=25)
    args = parser.parse_args()

    png_path = IMAGES_DIR / f"{args.page}.png"
    gray = np.array(Image.open(png_path).convert("L"))

    cv_bands = run_cv_assist(gray, args.max_lines)
    print(f"cv_assist: {len(cv_bands)} lines")
    for i, (y0, y1) in enumerate(cv_bands):
        print(f"  L{i+1:02d}: y0={y0} y1={y1} height={y1-y0}")

    seam_bands = extract_line_bands_windowed(gray, cv_bands, window_margin_px=args.seam_window_margin)
    print(f"\nseam (windowed): {len(seam_bands)} lines")
    for b in seam_bands:
        print(f"  {b['line_id']}: y0={b['y0']} y1={b['y1']} seam_mean_y={np.mean(b['seam_px']):.1f} seam_std={np.std(b['seam_px']):.1f}")

    crop_height = cv_bands[-1][1] + 20 if cv_bands else gray.shape[0]
    model_bands = run_model(gray, crop_height)
    print(f"\nmodel (learned): {len(model_bands)} lines")
    for i, (y0, y1) in enumerate(model_bands):
        print(f"  M{i+1:02d}: y0={y0} y1={y1} height={y1-y0}")

    # --- Overlay ---
    crop_h = min(gray.shape[0], crop_height + 40)
    overlay = Image.open(png_path).convert("RGB").crop((0, 0, gray.shape[1], crop_h))
    draw = ImageDraw.Draw(overlay)
    width = overlay.width

    for y0, y1 in cv_bands:
        draw.rectangle([2, y0, width - 2, y1], outline=(0, 200, 0), width=1)  # green

    for b in seam_bands:
        seam = b["seam_px"]
        pts = [(x, int(seam[x])) for x in range(0, len(seam), 2)]
        draw.line(pts, fill=(220, 0, 0), width=1)  # red

    for y0, y1 in model_bands:
        draw.rectangle([6, y0, width - 6, y1], outline=(0, 120, 255), width=1)  # blue

    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DEBUG_DIR / f"{args.page}_line_comparison.png"
    overlay.save(out_path)
    print(f"\nLegend: green=cv_assist box, red=seam-carving medial seam, blue=learned model box")
    print(f"Wrote overlay: {out_path}")


if __name__ == "__main__":
    main()
