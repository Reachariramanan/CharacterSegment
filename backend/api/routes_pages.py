from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.services.annotation_store import load_meta
from backend.services.cv_assist import CvAssistConfig, propose_boundaries

router = APIRouter()


@router.get("/api/pages")
def list_pages():
    from backend.services.annotation_store import has_annotation, list_pages as _list_pages

    pages = _list_pages()
    return {
        "pages": [{"page_id": p, "has_annotation": has_annotation(p)} for p in pages]
    }


@router.get("/api/pages/{page_id}")
def get_page(page_id: str):
    try:
        meta = load_meta(page_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {
        "page_id": page_id,
        "image_url": f"/data/images/{page_id}.png",
        "width": meta["width_px"],
        "height": meta["height_px"],
        "dpi": meta["dpi"],
    }


@router.get("/api/pages/{page_id}/proposals")
def get_proposals(page_id: str):
    import numpy as np
    from PIL import Image

    from backend.services.annotation_store import IMAGES_DIR

    try:
        load_meta(page_id)  # validates page exists
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    png_path = IMAGES_DIR / f"{page_id}.png"
    gray = np.array(Image.open(png_path).convert("L"))
    proposals = propose_boundaries(gray, CvAssistConfig())

    return {
        "page_id": page_id,
        "lines": [
            {
                "line_id": p.line_id,
                "y0": p.y0,
                "y1": p.y1,
                "boundaries_px": p.boundaries_px,
            }
            for p in proposals
        ],
    }
