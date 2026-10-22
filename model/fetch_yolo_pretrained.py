"""One-time fetch of Ultralytics' pretrained YOLO26n-seg (nano, COCO-pretrained)
checkpoint into model/checkpoints/ -- the "already learnt data" fine-tuning
starts from (see model/train_yolo.py). Nano, not small/medium, matches this
machine's tight memory budget (same reasoning as the rest of this project's
small custom models).

Usage:
    python -m model.fetch_yolo_pretrained
"""

from __future__ import annotations

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "model" / "checkpoints"
OUT_PATH = CHECKPOINT_DIR / "yolo26n-seg_pretrained.pt"
MODEL_NAME = "yolo26n-seg.pt"


def main() -> None:
    if OUT_PATH.exists():
        print(f"Already have {OUT_PATH}")
        return

    from ultralytics import YOLO

    # Ultralytics resolves a bare name like "yolo26n-seg.pt" by downloading
    # it (once) into its own cache dir if not already present there.
    model = YOLO(MODEL_NAME)
    downloaded_path = Path(model.ckpt_path) if getattr(model, "ckpt_path", None) else Path(MODEL_NAME)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if downloaded_path.exists():
        shutil.copy(downloaded_path, OUT_PATH)
    else:
        model.save(str(OUT_PATH))
    print(f"Saved pretrained YOLO26n-seg weights to {OUT_PATH}")


if __name__ == "__main__":
    main()
