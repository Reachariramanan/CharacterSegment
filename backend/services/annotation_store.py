"""Load/save annotation JSON for a page, matching the schema in the project plan.

A page with no saved annotation yet has none -- callers must not assume one
exists (see load_annotation). The editor has an explicit "Initialize CV
proposals" action for bootstrapping a blank page instead.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from backend.services.cv_assist import CvAssistConfig, propose_boundaries

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
IMAGES_DIR = PROJECT_ROOT / "data" / "images"
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_meta(page_id: str) -> dict[str, Any]:
    meta_path = IMAGES_DIR / f"{page_id}.meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No rendered page found for {page_id}: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def annotation_path(page_id: str) -> Path:
    return ANNOTATIONS_DIR / f"{page_id}.json"


def versions_dir(page_id: str) -> Path:
    return ANNOTATIONS_DIR / f"{page_id}_versions"


def _version_summary(data: dict[str, Any]) -> dict[str, Any]:
    lines = data.get("lines", [])
    n_user = sum(sum(1 for s in l.get("boundary_source", []) if s == "user") for l in lines)
    return {
        "n_lines": len(lines),
        "n_user_corrected": n_user,
        "line_source": data.get("line_source"),
    }


def list_versions(page_id: str) -> list[dict[str, Any]]:
    """Returns version metadata, newest first: {version_id, created_at,
    n_lines, n_user_corrected, line_source}. version_id is the save
    timestamp (sortable, filesystem-safe) and doubles as the filename."""
    vdir = versions_dir(page_id)
    if not vdir.exists():
        return []
    versions = []
    for path in vdir.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        summary = _version_summary(data)
        summary["version_id"] = path.stem
        summary["created_at"] = data.get("updated_at", data.get("created_at"))
        versions.append(summary)
    versions.sort(key=lambda v: v["version_id"], reverse=True)
    return versions


def load_version(page_id: str, version_id: str) -> dict[str, Any]:
    path = versions_dir(page_id) / f"{version_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"No version {version_id} for {page_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_proposals(page_id: str) -> dict[str, Any]:
    """Run the CV-assist pipeline fresh and package it into the annotation schema,
    with every boundary marked provenance="proposed"."""
    meta = load_meta(page_id)
    png_path = IMAGES_DIR / f"{page_id}.png"
    gray = np.array(Image.open(png_path).convert("L"))

    proposals = propose_boundaries(gray, CvAssistConfig())

    lines = []
    for p in proposals:
        n_boundaries = len(p.boundaries_px)
        lines.append(
            {
                "line_id": p.line_id,
                "y0": p.y0,
                "y1": p.y1,
                "boundaries_px": p.boundaries_px,
                "boundary_source": ["proposed"] * n_boundaries,
                "akshara_labels": [None] * max(0, n_boundaries - 1),
            }
        )

    now = _now_iso()
    return {
        "page_id": page_id,
        "source_pdf": meta["source_pdf"],
        "pdf_page_index_1based": meta["pdf_page_index_1based"],
        "image_path": f"data/images/{page_id}.png",
        "dpi": meta["dpi"],
        "image_width_px": meta["width_px"],
        "image_height_px": meta["height_px"],
        "rotation_deg_applied": meta["rotation_deg"],
        "created_at": now,
        "updated_at": now,
        "line_source": "cv_assist_v1",
        "lines": lines,
    }


def load_annotation(page_id: str) -> dict[str, Any]:
    """Load the saved annotation. Raises FileNotFoundError if the page has
    never been saved -- no automatic CV-assist fallback (the editor's
    "Initialize CV proposals" button is the explicit way to bootstrap a
    blank page; see build_proposals)."""
    path = annotation_path(page_id)
    if not path.exists():
        raise FileNotFoundError(f"No saved annotation yet for {page_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def list_pages() -> list[str]:
    """All page_ids that have a rendered image, sorted."""
    return sorted(p.name.removesuffix(".meta.json") for p in IMAGES_DIR.glob("page_*.meta.json"))


def has_annotation(page_id: str) -> bool:
    return annotation_path(page_id).exists()


def list_annotated_pages() -> list[str]:
    """page_ids that have a saved (at least once) annotation -- these are
    the only pages that should ever be fed to training, per the user's
    explicit "whatever the image has initial annots those only will go for
    training" requirement."""
    return [p for p in list_pages() if has_annotation(p)]


def save_annotation(page_id: str, data: dict[str, Any]) -> Path:
    """Validate minimal schema invariants and persist the annotation JSON."""
    if data.get("page_id") != page_id:
        raise ValueError(f"page_id mismatch: url={page_id} body={data.get('page_id')}")

    for line in data.get("lines", []):
        boundaries = line.get("boundaries_px", [])
        sources = line.get("boundary_source", [])
        if len(boundaries) != len(sources):
            raise ValueError(
                f"line {line.get('line_id')}: boundaries_px ({len(boundaries)}) and "
                f"boundary_source ({len(sources)}) length mismatch"
            )
        if boundaries != sorted(boundaries):
            raise ValueError(f"line {line.get('line_id')}: boundaries_px must be sorted ascending")

    data["updated_at"] = _now_iso()
    if "created_at" not in data:
        data["created_at"] = data["updated_at"]

    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = annotation_path(page_id)
    serialized = json.dumps(data, indent=2)
    path.write_text(serialized, encoding="utf-8")

    # Every save is also kept as an immutable, timestamped version snapshot
    # -- the "current" file above always reflects the latest save (so
    # training/model-guide/etc keep working unchanged), while the versions
    # directory lets the UI list and reactivate any prior save.
    vdir = versions_dir(page_id)
    vdir.mkdir(parents=True, exist_ok=True)
    version_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    (vdir / f"{version_id}.json").write_text(serialized, encoding="utf-8")

    return path
