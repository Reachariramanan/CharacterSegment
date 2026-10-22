"""CLI: run the classical-CV assist proposal algorithm on a rendered page and
save a debug overlay PNG (green = detected line bands, red = proposed
boundaries) so proposals can be visually sanity-checked.

Usage:
    python -m scripts.run_cv_assist --page page_0002
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from backend.services.cv_assist import CvAssistConfig, propose_boundaries

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "data" / "images"
DEBUG_DIR = PROJECT_ROOT / "data" / "images" / "debug"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page", type=str, required=True, help="page id, e.g. page_0002")
    args = parser.parse_args()

    png_path = IMAGES_DIR / f"{args.page}.png"
    gray_img = Image.open(png_path).convert("L")
    gray = np.array(gray_img)

    cfg = CvAssistConfig()
    proposals = propose_boundaries(gray, cfg)

    total_boundaries = sum(len(p.boundaries_px) for p in proposals)
    print(f"Detected {len(proposals)} text lines, {total_boundaries} total boundary points.")
    for p in proposals:
        n_segments = max(0, len(p.boundaries_px) - 1)
        print(f"  {p.line_id}: y=[{p.y0},{p.y1}]  segments={n_segments}  boundaries={p.boundaries_px}")

    # Overlay for visual sanity check.
    overlay = gray_img.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    for p in proposals:
        draw.rectangle([0, p.y0, overlay.width - 1, p.y1], outline=(0, 200, 0), width=1)
        for x in p.boundaries_px:
            draw.line([(x, p.y0), (x, p.y1)], fill=(220, 0, 0), width=1)

    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DEBUG_DIR / f"{args.page}_cv_assist_overlay.png"
    overlay.save(out_path)
    print(f"Wrote overlay: {out_path}")

    # Also dump raw proposals as JSON for inspection / feeding into annotation store later.
    proposals_json = [
        {"line_id": p.line_id, "y0": p.y0, "y1": p.y1, "boundaries_px": p.boundaries_px}
        for p in proposals
    ]
    json_path = DEBUG_DIR / f"{args.page}_cv_assist_proposals.json"
    json_path.write_text(json.dumps(proposals_json, indent=2), encoding="utf-8")
    print(f"Wrote proposals: {json_path}")


if __name__ == "__main__":
    main()
