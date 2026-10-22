"""Runs model training as a genuinely separate OS process (not an in-process
thread) so a UI-triggered /retrain request doesn't block the FastAPI server
and -- more importantly -- doesn't contend with uvicorn's event loop for the
GIL. An earlier in-process-thread version of this was observed to run ~100x
slower than the identical training loop run as a standalone script (10+
min/epoch inside the server thread vs. ~6s/epoch standalone) -- CPU-bound
Python work in the training loop (dataset shuffling, collate, loss-term
orchestration) apparently starved badly when sharing a thread/GIL with the
server's async event loop, even though PyTorch's CUDA calls themselves
release the GIL. A subprocess has its own interpreter and CUDA context, so
it can't contend with the server at all.

Progress is reported via a small JSON status file per model_type (written by
the training script's progress_callback), which GET /retrain/status reads --
this works across the process boundary where an in-memory dict wouldn't.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STATUS_DIR = PROJECT_ROOT / "model" / "checkpoints" / "_training_status"
STATUS_DIR.mkdir(parents=True, exist_ok=True)

_processes: dict[str, subprocess.Popen] = {}
_log_files: dict[str, Any] = {}  # model_type -> open file handle backing that process's stdout/stderr


def _status_path(model_type: str) -> Path:
    return STATUS_DIR / f"{model_type}.json"


def _log_path(model_type: str) -> Path:
    return STATUS_DIR / f"{model_type}.log"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_status(model_type: str, **fields: Any) -> None:
    path = _status_path(model_type)
    current = {}
    if path.exists():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            current = {}
    current.update(fields)
    path.write_text(json.dumps(current), encoding="utf-8")


def _pid_is_alive(pid: int) -> bool:
    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True, timeout=5
        )
        return str(pid) in result.stdout
    except Exception:
        return True  # can't tell -- don't falsely mark a real job as dead


def get_status(model_type: str) -> dict[str, Any]:
    # Reap a finished process so `running` reflects reality even if the
    # script's own status write raced the process exit.
    proc = _processes.get(model_type)
    if proc is not None and proc.poll() is not None:
        _reap(model_type, proc)

    path = _status_path(model_type)
    if not path.exists():
        return {"running": False}
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"running": False}

    # A status file can say running=True with no live process behind it if
    # the SERVER itself was restarted/killed (e.g. an OS low-memory kill)
    # while a subprocess-based job was in flight -- this server instance
    # never spawned it, so it's not in `_processes` to be reaped normally.
    # Self-heal using the PID recorded at launch.
    if status.get("running", False) and model_type not in _processes:
        pid = status.get("pid")
        if pid is not None and not _pid_is_alive(pid):
            status = {
                "running": False,
                "done": False,
                "error": "Training process was no longer running (likely killed along with the server, e.g. an OS low-memory event) -- restart training if needed.",
                "finished_at": _now_iso(),
                "epoch": status.get("epoch"),
                "total_epochs": status.get("total_epochs"),
                "loss": status.get("loss"),
                "started_at": status.get("started_at"),
            }
            _write_status(model_type, **status)

    return status


MODEL_TYPES = ("boundary", "line_segmenter", "character_separator", "line_separator", "yolo_seg")

TRAIN_MODULES = {
    "boundary": "model.train",
    "line_segmenter": "model.train_line_segmenter",
    "character_separator": "model.train_character_separator",
    "line_separator": "model.train_line_separator",
    "yolo_seg": "model.train_yolo",
}

# character_separator's dataset (CharacterSeamDataset) has no concept of a
# trusted line-prefix -- it builds one example per already-annotated
# boundary directly, so --max-lines isn't a CLI arg train_character_separator.py
# accepts; passing it would crash the subprocess with an argparse error.
# yolo_seg (model/train_yolo.py) always exports from every saved-annotation
# page itself (see model/yolo_dataset_export.py) -- no page-subset concept.
SUPPORTS_MAX_LINES = {"boundary", "line_segmenter"}


def any_running() -> str | None:
    """Returns the model_type currently training, if any -- only one job
    runs at a time regardless of type (they were observed to badly starve
    each other for this 4GB card's GPU when run concurrently)."""
    for model_type in MODEL_TYPES:
        if get_status(model_type).get("running", False):
            return model_type
    return None


def _reap(model_type: str, proc: subprocess.Popen) -> None:
    returncode = proc.returncode
    del _processes[model_type]
    log_file = _log_files.pop(model_type, None)
    if log_file is not None:
        try:
            log_file.close()
        except Exception:
            pass
    status = get_status(model_type)
    if status.get("running", False):
        # Process exited without the script marking itself done -- a crash.
        log_tail = ""
        try:
            log_bytes = _log_path(model_type).read_bytes()
            log_tail = log_bytes[-4000:].decode("utf-8", errors="replace")
        except Exception:
            pass
        _write_status(
            model_type,
            running=False,
            done=False,
            error=f"Training process exited with code {returncode}.\n{log_tail}",
            finished_at=_now_iso(),
        )


def start_training(
    model_type: str,
    pages: list[str],
    max_lines: int | None,
    epochs: int,
    batch_size: int | None = None,
    resume: Path | None = None,
) -> None:
    """Launches training for `model_type` (one of MODEL_TYPES)
    as a subprocess running the corresponding CLI module. Raises
    RuntimeError if ANY training job (any model type) is already
    running. If `resume` is given (and exists), the subprocess loads those
    weights before training instead of starting from a random init --
    this is what lets "Update model" fine-tune the currently-served
    checkpoint rather than discarding it and retraining from scratch."""
    running = any_running()
    if running:
        raise RuntimeError(f"{running} training is already running -- wait for it to finish first")

    _write_status(
        model_type,
        running=True,
        epoch=0,
        total_epochs=epochs,
        loss=None,
        done=False,
        error=None,
        started_at=_now_iso(),
        finished_at=None,
    )

    module = TRAIN_MODULES[model_type]
    cmd = [
        sys.executable,
        "-u",
        "-m",
        module,
        "--pages",
        *pages,
        "--epochs",
        str(epochs),
        "--status-file",
        str(_status_path(model_type)),
    ]
    if max_lines is not None and model_type in SUPPORTS_MAX_LINES:
        cmd += ["--max-lines", str(max_lines)]
    if batch_size is not None and model_type == "boundary":
        cmd += ["--batch-size", str(batch_size)]
    if resume is not None and resume.exists():
        cmd += ["--resume", str(resume)]

    # stdout/stderr go to a log FILE, not a pipe -- a pipe nobody drains
    # while the process is running fills up its OS buffer once the
    # subprocess writes enough output, and the write() blocks forever
    # (observed directly: model.train_yolo wraps ultralytics, which prints
    # a full model summary, AMP-check download progress, and per-epoch
    # progress tables -- far more than the other train_*.py scripts' quiet
    # "every 50 epochs" logging -- and the training subprocess hung at
    # epoch 0 with 0 CPU time, genuinely blocked on a full pipe, not slow).
    # A file has no such limit, and doubles as a persistent log to inspect.
    log_file = open(_log_path(model_type), "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    _processes[model_type] = proc
    _log_files[model_type] = log_file
    _write_status(model_type, pid=proc.pid)

    from backend.services import model_runtime, yolo_runtime

    # Invalidate the relevant module-level cache so the next /model-guide
    # call reloads the freshly-written checkpoint instead of the stale
    # in-memory model.
    if model_type == "boundary":
        model_runtime._model = None
    elif model_type == "line_segmenter":
        model_runtime._line_model = None
    elif model_type == "character_separator":
        model_runtime._character_separator_model = None
    elif model_type == "line_separator":
        model_runtime._line_separator_model = None
    elif model_type == "yolo_seg":
        yolo_runtime.reload("finetuned")
