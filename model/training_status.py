"""Tiny JSON status-file writer shared by model/train.py and
model/train_line_segmenter.py, so a subprocess-based training run can report
progress back to the backend (backend/services/training_runtime.py reads
the same file format) without model/ needing to import anything from
backend/.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(path: Path | None, **fields: Any) -> None:
    if path is None:
        return
    current: dict[str, Any] = {}
    if path.exists():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            current = {}
    current.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current), encoding="utf-8")
