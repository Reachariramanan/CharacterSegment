"""Transfer-learning fine-tune of Ultralytics YOLO26n-seg on the "line"
instance-segmentation dataset exported by model/yolo_dataset_export.py.

Unlike the rest of model/train_*.py this wraps the `ultralytics` package
rather than a hand-rolled training loop -- YOLO26 is a full pretrained
detector/segmenter (COCO-pretrained weights), not something to reimplement.
Same CLI/--resume/--status-file conventions as the other scripts so it
plugs into the existing subprocess launcher (backend/services/training_runtime.py)
unchanged.

"Starting from the beginning" (per the user's request) means starting from
the pretrained COCO weights (PRETRAINED_PATH) so the model reuses what it
already learned about generic shapes/edges, rather than random init --
--resume, when given an existing fine-tuned checkpoint, continues from that
instead (same "never restart from scratch once you have a checkpoint"
convention as Train boundaries / Train lines).

Usage:
    python -m model.train_yolo --epochs 100
    python -m model.train_yolo --epochs 30 --resume model/checkpoints/yolo_seg_finetuned.pt
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "model" / "checkpoints"
PRETRAINED_PATH = CHECKPOINT_DIR / "yolo26n-seg_pretrained.pt"
DEFAULT_OUT_PATH = CHECKPOINT_DIR / "yolo_seg_finetuned.pt"
RUNS_DIR = CHECKPOINT_DIR / "_yolo_runs"

# YOLO26n-seg's top-level module list (model.model.model) has 24 entries,
# indices 0-23 -- verified directly against the pretrained checkpoint:
# 0-22 are the backbone/neck (Conv/C3k2/SPPF/C2PSA/Upsample/Concat blocks),
# 23 is the Segment26 detection+mask head. Ultralytics' `freeze=N` freezes
# layer indices [0, N) (see ultralytics/engine/trainer.py's freeze_list),
# so freeze=23 leaves ONLY index 23 (the head) trainable -- "don't train
# the entire model, only the last layer" per the user's explicit request.
# This also directly helps the OOM/small-dataset situation: far fewer
# trainable params (and no backbone gradients to store) means less memory
# and less overfitting risk on ~3 pages of data.
FREEZE_LAYERS = 23


def train_yolo_model(
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 4,
    freeze: int = FREEZE_LAYERS,
    resume: Path | None = None,
    out_path: Path = DEFAULT_OUT_PATH,
    status_file: Path | None = None,
) -> Path:
    from model.training_status import now_iso, write_status
    from model.yolo_dataset_export import export_yolo_dataset

    data_yaml = export_yolo_dataset()

    from ultralytics import YOLO

    start_weights = resume if (resume is not None and resume.exists()) else PRETRAINED_PATH
    if not start_weights.exists():
        raise FileNotFoundError(
            f"No pretrained YOLO26n-seg weights at {start_weights} -- run "
            "`python -m model.fetch_yolo_pretrained` first."
        )
    model = YOLO(str(start_weights))

    def on_epoch_end(trainer):
        epoch = trainer.epoch + 1
        loss = trainer.loss.item() if trainer.loss is not None else None
        write_status(status_file, running=True, epoch=epoch, total_epochs=epochs, loss=loss)

    model.add_callback("on_train_epoch_end", on_epoch_end)

    run_name = "train"
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        freeze=freeze,
        project=str(RUNS_DIR),
        name=run_name,
        exist_ok=True,
        verbose=False,
        plots=False,
        # Dataset is a handful of pages -- multiple Windows spawn-mode
        # DataLoader worker processes add startup fragility (each has to
        # re-import torch/ultralytics) for zero benefit at this size.
        workers=0,
    )

    best = RUNS_DIR / run_name / "weights" / "best.pt"
    if not best.exists():
        raise RuntimeError(f"Training finished but no best.pt found at {best}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # --pages is accepted (and ignored) only for CLI-shape parity with the
    # other model/train_*.py scripts -- training_runtime.py always passes
    # it, but YOLO always exports from every saved-annotation page itself
    # (see yolo_dataset_export.py) rather than a page subset.
    parser.add_argument("--pages", nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    # 640, not YOLO's larger presets -- a 4GB card (this machine's RTX 500
    # Ada) hit a genuine CUDA out-of-memory crash at imgsz=960 on a
    # near-full-resolution manuscript page, specifically from YOLO26's
    # semantic-mask auxiliary loss (a dense NxHxW tensor, much heavier than
    # the sparse box/instance losses) -- verified directly (see
    # model/checkpoints/_training_status/yolo_seg.log from that run).
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    # Freeze the backbone/neck (indices 0..freeze-1 of 24), train only the
    # remaining layers -- default 23 trains ONLY the final Segment head.
    # Pass 0 to fine-tune the whole model instead.
    parser.add_argument("--freeze", type=int, default=FREEZE_LAYERS)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--status-file", type=Path, default=None)
    args = parser.parse_args()

    from model.training_status import now_iso, write_status

    write_status(args.status_file, running=True, epoch=0, total_epochs=args.epochs, loss=None,
                 done=False, error=None, started_at=now_iso(), finished_at=None)

    try:
        out_path = train_yolo_model(
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            freeze=args.freeze,
            resume=args.resume,
            out_path=args.out,
            status_file=args.status_file,
        )
    except Exception as e:
        write_status(args.status_file, running=False, done=False, error=str(e), finished_at=now_iso())
        raise

    print(f"Saved checkpoint: {out_path}")
    write_status(args.status_file, running=False, done=True, finished_at=now_iso())


if __name__ == "__main__":
    main()
