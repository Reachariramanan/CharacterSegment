from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from backend.services.annotation_store import (
    list_versions,
    load_annotation,
    load_version,
    save_annotation,
)

router = APIRouter()


@router.get("/api/pages/{page_id}/annotations")
def get_annotations(page_id: str):
    try:
        return load_annotation(page_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/api/pages/{page_id}/annotations")
def post_annotations(page_id: str, body: dict[str, Any]):
    try:
        path = save_annotation(page_id, body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "saved", "path": str(path)}


@router.get("/api/pages/{page_id}/versions")
def get_versions(page_id: str):
    return {"versions": list_versions(page_id)}


@router.get("/api/pages/{page_id}/versions/{version_id}")
def get_version(page_id: str, version_id: str):
    try:
        return load_version(page_id, version_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
