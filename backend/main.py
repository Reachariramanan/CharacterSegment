"""FastAPI backend for the Grantha boundary annotator.

Run with:
    uvicorn backend.main:app --reload --port 8000
Then open http://localhost:8000/ in a browser.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.api import routes_annotations, routes_model, routes_pages, routes_seams, routes_yolo

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
DATA_DIR = PROJECT_ROOT / "data"

app = FastAPI(title="Grantha Boundary Annotator")

app.include_router(routes_pages.router)
app.include_router(routes_annotations.router)
app.include_router(routes_model.router)
app.include_router(routes_seams.router)
app.include_router(routes_yolo.router)

# Serve page images/debug overlays directly.
app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")

# Serve frontend static assets (app.js, style.css) under /static.
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))
