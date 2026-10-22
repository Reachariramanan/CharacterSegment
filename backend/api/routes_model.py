from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.services import annotation_store, model_runtime, training_runtime, yolo_runtime

router = APIRouter()


@router.post("/api/pages/{page_id}/model-guide")
def model_guide(page_id: str):
    if not model_runtime.is_available():
        raise HTTPException(
            status_code=501,
            detail=(
                "No trained model checkpoint yet at model/checkpoints/boundary_model.pt. "
                "Annotate enough pages and run model/train.py first."
            ),
        )
    try:
        return model_runtime.run_model_guide(page_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


# Different default epoch counts per model: the line-segmenter trains on a
# single whole-page example per step (vs. the boundary model's many small
# windows), and was observed to need meaningfully more epochs to open up
# every inter-line gap on a full page -- 150 converged to a degenerate
# "almost everything is one giant line" solution, 300 resolved correctly.
# character_separator/line_separator mirror their own CLI defaults (see
# train_character_separator.py / train_line_separator.py). yolo_seg is
# transfer learning from COCO-pretrained weights (not random init), so it
# converges in far fewer epochs than the from-scratch custom models.
DEFAULT_EPOCHS = {"boundary": 100, "line_segmenter": 300, "character_separator": 400, "line_separator": 500, "yolo_seg": 100}

# Fewer epochs when resuming from an existing checkpoint ("Update model"):
# the weights are already close to a good solution, so this is a fine-tune
# pass on whatever's new in the annotation, not training from scratch.
DEFAULT_EPOCHS_RESUME = {"boundary": 30, "line_segmenter": 60, "character_separator": 80, "line_separator": 100, "yolo_seg": 30}

# Checkpoint each model_type resumes from when resume=True -- the same file
# model_runtime.py/yolo_runtime.py serves inference from, so "Update model"/
# "Train YOLO" always continue from whatever's currently live. For yolo_seg
# this is the FINE-TUNED checkpoint specifically -- when it doesn't exist
# yet (never trained), resuming=False below and training starts from the
# pretrained COCO checkpoint instead (see model/train_yolo.py), never from
# a random init.
RESUME_CHECKPOINTS = {
    "boundary": model_runtime.CHECKPOINT_PATH,
    "line_segmenter": model_runtime.LINE_CHECKPOINT_PATH,
    "character_separator": model_runtime.CHARACTER_SEPARATOR_CHECKPOINT_PATH,
    "line_separator": model_runtime.LINE_SEPARATOR_CHECKPOINT_PATH,
    "yolo_seg": yolo_runtime.FINETUNED_PATH,
}

VALID_MODEL_TYPES = ("boundary", "line_segmenter", "character_separator", "line_separator", "yolo_seg")


@router.post("/api/pages/{page_id}/retrain")
def retrain(
    page_id: str,
    model_type: str = "boundary",
    epochs: int | None = None,
    max_lines: int | None = None,
    batch_size: int | None = None,
    resume: bool = False,
):
    """Kicks off training in a background thread and returns immediately;
    poll GET .../retrain/status?model_type=... for progress. model_type is
    "boundary" (per-line akshara x-boundaries), "line_segmenter" (page-
    level line y-bands), or "character_separator" (curved, ink-avoiding
    akshara seam generation -- see model/character_separator_model.py).
    batch_size only applies to "boundary" -- a bigger batch means fewer
    distinct-shape batches per epoch, which matters a lot here since each
    new tensor shape costs a cuFFT replan (see model/ssm.py).
    epochs defaults per model_type (see DEFAULT_EPOCHS / DEFAULT_EPOCHS_RESUME)
    if not given. resume=True loads the currently-served checkpoint's weights
    first and fine-tunes them (falls back to training from scratch if no
    checkpoint exists yet).

    Trains on every page that has a *saved* annotation (not just the page
    currently open in the editor, and never on a page that's only ever seen
    unsaved/auto CV-assist proposals) -- page_id here just identifies which
    tab triggered the request, for status polling."""
    if model_type not in VALID_MODEL_TYPES:
        raise HTTPException(status_code=400, detail=f"Unknown model_type: {model_type}")

    pages = annotation_store.list_annotated_pages()
    if not pages:
        raise HTTPException(
            status_code=400,
            detail="No pages have a saved annotation yet -- use 'Initialize CV proposals' then Save on at least one page first.",
        )

    resume_path = RESUME_CHECKPOINTS[model_type] if resume else None
    resuming = resume_path is not None and resume_path.exists()

    if epochs is None:
        epochs = DEFAULT_EPOCHS_RESUME[model_type] if resuming else DEFAULT_EPOCHS[model_type]

    try:
        training_runtime.start_training(
            model_type=model_type,
            pages=pages,
            max_lines=max_lines,
            epochs=epochs,
            batch_size=batch_size,
            resume=resume_path,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))

    return {"status": "started", "model_type": model_type, "epochs": epochs, "resumed": resuming, "pages": pages}


@router.get("/api/pages/{page_id}/retrain/status")
def retrain_status(page_id: str, model_type: str = "boundary"):
    if model_type not in VALID_MODEL_TYPES:
        raise HTTPException(status_code=400, detail=f"Unknown model_type: {model_type}")
    return training_runtime.get_status(model_type)
