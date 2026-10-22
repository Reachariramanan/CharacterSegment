from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.services import yolo_runtime

router = APIRouter()


@router.get("/api/pages/{page_id}/yolo-guide")
def yolo_guide(page_id: str, variant: str = "finetuned"):
    """variant="original" previews the stock COCO-pretrained YOLO26n-seg
    checkpoint (a sanity check that it loads/runs -- its classes aren't
    ours); variant="finetuned" previews our own checkpoint trained on the
    "line" class by model/train_yolo.py (see the Train YOLO button)."""
    if variant not in ("original", "finetuned"):
        raise HTTPException(status_code=400, detail="variant must be 'original' or 'finetuned'")
    if not yolo_runtime.is_available(variant):
        detail = (
            "No pretrained YOLO26n-seg checkpoint yet -- run `python -m model.fetch_yolo_pretrained` first."
            if variant == "original"
            else "No fine-tuned YOLO checkpoint yet -- use the 'Train YOLO' button first."
        )
        raise HTTPException(status_code=501, detail=detail)
    try:
        return yolo_runtime.run_yolo_guide(page_id, variant)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
