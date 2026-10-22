"""Exports every saved-annotation page into an Ultralytics YOLO-seg dataset
with two classes:
  0 "line"      -- one polygon per text line, spanning its full left/right
                   edge (boundaries_px[0]/[-1]) and top/bottom seam.
  1 "character" -- one polygon per INTERIOR akshara segment, i.e. the strip
                   between each consecutive pair of boundaries_px -- this is
                   the actual per-akshara detection target ("it has to find
                   the characters").
Both reuse the same top/bottom seam curve (or flat y0/y1) already drawn as
the line band and boundary cuts in the UI -- see frontend/app.js's
getLineX0/getLineX1/getTopY/getBottomY for the identical convention on the
editor side. A "character" instance can heavily overlap its parent "line"
instance in image space -- that's expected and fine for a multi-class
detector (different classes, unambiguous).

Re-run fresh on every "Train YOLO" (model/train_yolo.py) so the dataset
always reflects whatever's currently saved, the same "training only ever
sees saved annotations" rule the other models already follow (see
backend/services/annotation_store.list_annotated_pages). No extra trust
filter beyond that (e.g. requiring "user"-sourced boundaries) -- matches
how model/train.py and model/character_seam_dataset.py already treat every
saved boundary as trainable ground truth, CV-proposed or not.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "data" / "images"
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"
DEFAULT_OUT_DIR = PROJECT_ROOT / "model" / "yolo_dataset"

CLASS_NAMES = ["line", "character"]
LINE_CLASS_IDX = 0
CHARACTER_CLASS_IDX = 1

N_POLY_SAMPLES_LINE = 24  # points sampled along a line's top/bottom curve -- keeps label files compact while still tracing a wavy baseline
N_POLY_SAMPLES_CHAR = 8  # characters are much narrower -- fewer samples needed to trace their (usually near-flat) span
MIN_SPAN_PX = 3  # a span this narrow (in x) isn't a usable polygon


def _annotated_page_ids() -> list[str]:
    return sorted(p.stem for p in ANNOTATIONS_DIR.glob("page_*.json"))


def _span_polygon(
    line: dict[str, Any], x0_raw: float, x1_raw: float, width: int, height: int, n_samples: int
) -> list[float] | None:
    """One YOLO-seg polygon for the x-span [x0_raw, x1_raw] of `line`:
    normalized (x,y) pairs walking the top edge left->right then the bottom
    edge right->left, using the line's own top_seam_px/bottom_seam_px (or
    flat y0/y1 if it's never been split/curved). None if the span is too
    narrow to be a usable instance."""
    x0 = max(0, min(round(x0_raw), width - 1))
    x1 = max(0, min(round(x1_raw), width - 1))
    if x1 - x0 < MIN_SPAN_PX:
        return None

    top_seam = line.get("top_seam_px")
    bottom_seam = line.get("bottom_seam_px")
    top_y = float(line["y0"])
    bottom_y = float(line["y1"])

    n_samples = max(3, min(n_samples, x1 - x0 + 1))
    xs = np.linspace(x0, x1, n_samples).round().astype(int)
    xs = np.clip(xs, 0, width - 1)

    def y_at(seam, flat_y, x):
        return float(seam[x]) if seam else flat_y

    top_pts = [(x, y_at(top_seam, top_y, x)) for x in xs]
    bottom_pts = [(x, y_at(bottom_seam, bottom_y, x)) for x in xs[::-1]]

    coords: list[float] = []
    for px, py in top_pts + bottom_pts:
        coords.append(round(px / width, 6))
        coords.append(round(max(0.0, min(1.0, py / height)), 6))  # normalize BEFORE clamping to [0,1] -- clamping the raw pixel value first collapsed every y to ~1/height
    return coords


def _line_polygon(line: dict[str, Any], width: int, height: int) -> list[float] | None:
    b = line["boundaries_px"]
    return _span_polygon(line, b[0], b[-1], width, height, N_POLY_SAMPLES_LINE)


def _character_polygons(line: dict[str, Any], width: int, height: int) -> list[list[float]]:
    b = line["boundaries_px"]
    polys = []
    for i in range(len(b) - 1):
        poly = _span_polygon(line, b[i], b[i + 1], width, height, N_POLY_SAMPLES_CHAR)
        if poly is not None:
            polys.append(poly)
    return polys


def export_yolo_dataset(out_dir: Path = DEFAULT_OUT_DIR) -> Path:
    """Rebuilds out_dir from scratch from every saved-annotation page.
    Returns the path to the dataset's data.yaml (what YOLO's `data=` wants)."""
    images_out = out_dir / "images" / "train"
    labels_out = out_dir / "labels" / "train"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    n_pages, n_lines, n_chars = 0, 0, 0
    for page_id in _annotated_page_ids():
        meta = json.loads((IMAGES_DIR / f"{page_id}.meta.json").read_text(encoding="utf-8"))
        width, height = meta["width_px"], meta["height_px"]
        annotation = json.loads((ANNOTATIONS_DIR / f"{page_id}.json").read_text(encoding="utf-8"))

        label_lines = []
        for line in annotation["lines"]:
            line_poly = _line_polygon(line, width, height)
            if line_poly is not None:
                label_lines.append(f"{LINE_CLASS_IDX} " + " ".join(f"{c:.6f}" for c in line_poly))
                n_lines += 1
            for char_poly in _character_polygons(line, width, height):
                label_lines.append(f"{CHARACTER_CLASS_IDX} " + " ".join(f"{c:.6f}" for c in char_poly))
                n_chars += 1
        if not label_lines:
            continue

        shutil.copy(IMAGES_DIR / f"{page_id}.png", images_out / f"{page_id}.png")
        (labels_out / f"{page_id}.txt").write_text("\n".join(label_lines) + "\n", encoding="utf-8")
        n_pages += 1

    if n_pages == 0:
        raise RuntimeError(
            "No annotated pages with at least one usable line -- nothing to export for YOLO training. "
            "Initialize CV proposals + Save on at least one page first."
        )

    data_yaml = out_dir / "data.yaml"
    yaml_path_str = str(out_dir.resolve()).replace("\\", "/")
    names_yaml = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    data_yaml.write_text(
        f"path: {yaml_path_str}\n"
        "train: images/train\n"
        "val: images/train\n"  # no held-out split at this data scale -- same tradeoff the rest of this project's models already make
        "names:\n"
        f"{names_yaml}\n",
        encoding="utf-8",
    )
    print(f"Exported {n_lines} line + {n_chars} character instances across {n_pages} pages to {out_dir}")
    return data_yaml


if __name__ == "__main__":
    export_yolo_dataset()
