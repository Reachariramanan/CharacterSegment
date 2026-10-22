"""Inference wrapper for YOLO26n-seg line-segmentation -- both the stock
pretrained (COCO) checkpoint and our own fine-tuned one, so the UI can show
"Check original YOLO" / "Check finetuned YOLO" previews side by side.
Mirrors model_runtime.py's lazy-load-and-cache-per-checkpoint pattern for
the custom models.

"original" is a sanity-check / before-and-after reference, not a real line
predictor -- the stock weights only know COCO's 80 classes, none of which
is "line", so whatever it fires on is incidental. "finetuned" is
model/train_yolo.py's output, trained on the "line" class exported by
model/yolo_dataset_export.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "model" / "checkpoints"
PRETRAINED_PATH = CHECKPOINT_DIR / "yolo26n-seg_pretrained.pt"
FINETUNED_PATH = CHECKPOINT_DIR / "yolo_seg_finetuned.pt"
IMAGES_DIR = PROJECT_ROOT / "data" / "images"

VARIANT_PATHS = {"original": PRETRAINED_PATH, "finetuned": FINETUNED_PATH}

_models: dict[str, Any] = {}  # "original" | "finetuned" -> loaded ultralytics YOLO, cached until reload()


def is_available(variant: str) -> bool:
    return VARIANT_PATHS[variant].exists()


def _load(variant: str):
    if variant in _models:
        return _models[variant]
    path = VARIANT_PATHS[variant]
    if not path.exists():
        return None
    from ultralytics import YOLO

    model = YOLO(str(path))
    _models[variant] = model
    return model


def reload(variant: str) -> None:
    """Drops the cached model so the next request picks up a freshly
    trained checkpoint -- called by training_runtime after a YOLO run."""
    _models.pop(variant, None)


def run_yolo_guide(page_id: str, variant: str) -> dict[str, Any]:
    """Runs YOLO-seg inference on a page. Returns per-instance polygons in
    page-pixel coordinates (frontend overlay draws these directly, same
    coordinate space as boundaries_px/top_seam_px elsewhere)."""
    model = _load(variant)
    if model is None:
        raise FileNotFoundError(f"No YOLO checkpoint for variant={variant!r} yet.")

    png_path = IMAGES_DIR / f"{page_id}.png"
    if not png_path.exists():
        raise FileNotFoundError(f"No rendered page image for {page_id}: {png_path}")

    results = model.predict(str(png_path), verbose=False)[0]

    instances: list[dict[str, Any]] = []
    if results.masks is not None:
        polys = results.masks.xy  # list of (N,2) numpy arrays, pixel coords
        boxes = results.boxes
        classes = boxes.cls.tolist() if boxes is not None else [0.0] * len(polys)
        confs = boxes.conf.tolist() if boxes is not None else [None] * len(polys)
        names = results.names
        for poly, cls, conf in zip(polys, classes, confs):
            instances.append(
                {
                    "class_name": names.get(int(cls), str(int(cls))),
                    "confidence": round(float(conf), 4) if conf is not None else None,
                    "polygon_px": [[round(float(x), 1), round(float(y), 1)] for x, y in poly],
                }
            )

    return {"page_id": page_id, "variant": variant, "instances": instances}
