from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.services.annotation_store import IMAGES_DIR, load_annotation, load_meta
from backend.services.seam_extraction import extract_line_bands_windowed

router = APIRouter()


@router.get("/api/pages/{page_id}/seams")
def get_seams(page_id: str):
    """Runs the windowed seam-carving line extraction (Eq. 4 SDT dynamic
    program, constrained per-line to avoid the cross-line drift the
    unconstrained global version shows on this document -- see
    seam_extraction.py) anchored on the page's CURRENT annotation line
    bands, so the overlay always reflects whatever's actually loaded/saved
    in the UI right now."""
    import numpy as np
    from PIL import Image

    try:
        load_meta(page_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    annotation = load_annotation(page_id)
    # y0/y1 can be non-integer for a split line (curved-seam bounding
    # envelope) -- round before use as array slice indices downstream.
    anchor_bands = [(round(line["y0"]), round(line["y1"])) for line in annotation["lines"]]

    png_path = IMAGES_DIR / f"{page_id}.png"
    gray = np.array(Image.open(png_path).convert("L"))

    seam_bands = extract_line_bands_windowed(gray, anchor_bands)
    return {
        "page_id": page_id,
        "lines": [
            {"line_id": b["line_id"], "y0": b["y0"], "y1": b["y1"], "seam_px": b["seam_px"]}
            for b in seam_bands
        ],
    }
