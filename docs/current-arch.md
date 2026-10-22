# Current architecture

A living snapshot of how this project is actually built, as of 2026-10-22. See `docs/claude.md` for working conventions and `docs/architectures-experimented.md` for the fuller history of every modeling approach tried (including trained-but-unused and considered-but-never-built ones); this file is the "what exists and why" reference for what's actually wired into the live app today.

## What this is

A manuscript annotation + training tool for Grantha-script (South Indian) page scans: a human traces line bands and akshara (character) boundaries on scanned pages through a browser UI, and those corrections become training data for several models that try to automate the same job. It is a FastAPI backend + a vanilla-JS canvas editor (no frontend framework) + PyTorch/Ultralytics models, all running locally on one machine (an RTX 500 Ada laptop GPU, 4GB VRAM — a real, frequently-hit constraint throughout).

## Directory layout

```
backend/        FastAPI app: routes + stateless-ish services
  api/           routes_pages, routes_annotations, routes_model, routes_seams, routes_yolo
  services/      annotation_store, cv_assist, model_runtime, yolo_runtime,
                 training_runtime, seam_extraction, pdf_extract
frontend/        index.html + app.js (single file, no build step) + style.css
model/           custom PyTorch model code, training scripts, dataset classes
data/
  images/        page_NNNN.png + .meta.json, rendered from the source PDF
  annotations/   page_NNNN.json (current) + page_NNNN_versions/*.json (every past save)
scripts/         one-off CLIs: extract_page, extract_all_pages, run_cv_assist, etc.
tests/           a handful of pytest size/shape checks
docs/            this file + claude.md
```

## Data model

`data/annotations/page_NNNN.json`:
```jsonc
{
  "page_id": "page_0002", "image_width_px": 913, "image_height_px": 1297,
  "line_source": "cv_assist_v1",
  "lines": [{
    "line_id": "L01", "y0": 106, "y1": 123,
    "boundaries_px": [447, 487, ...],       // akshara x-cuts; [0] and [-1] are the LINE'S OWN left/right edge, not the page edge
    "boundary_source": ["user", "proposed", ...],
    "akshara_labels": [null, ...],
    "top_seam_px": [...], "bottom_seam_px": [...]  // optional: one y-value per page column, present once a line has been split/traced into a curve (see "Split line"/"Extend line" in the UI); absent means "flat", use y0/y1
  }]
}
```
Key point, easy to get wrong: `boundaries_px[0]`/`boundaries_px[-1]` are **not** fixed to the page width. They used to be, but scanner-margin noise at page edges forced every model and the editor to treat a line's horizontal extent as its own editable property — trimmed inward by the Scissor tool, extended outward by the Extend-line tool, dragged directly like any other boundary. Every piece of code that crops a line (training datasets, inference, the frontend renderer) must read `boundaries_px[0]`/`[-1]`, never assume `0`/`page_width`.

Every `POST .../annotations` save also writes an immutable timestamped snapshot into `page_NNNN_versions/`; the UI's right-hand "Data versions" panel lists and can reload any of them.

A page only ever enters training if it has an actual saved annotation file — `annotation_store.list_annotated_pages()` is the single source of truth for "which pages are trustworthy," used both by training and by the page-list UI's checkmark.

## Backend services

- **`annotation_store`**: load/save/version annotation JSON, `list_pages()` (every rendered image), `list_annotated_pages()` (only ones with a save). No auto-CV-assist fallback — a page with no saved annotation 404s; the editor's explicit "Initialize CV proposals" button is the only way to bootstrap one.
- **`cv_assist`**: classical (non-ML) column/row-valley heuristics that produce the first-pass proposal for a brand-new page (Sauvola binarization + peak/valley finding).
- **`seam_extraction`**: SDT (signed distance transform) ink-energy computation, shared by multiple datasets, plus `find_ink_avoiding_path` — a classical DP seam-carving method used for the curved line/character seam overlay shown in "Show model output" (see below, this is *not* currently coming from a trained model).
- **`model_runtime`**: lazy-loads/caches the custom PyTorch checkpoints, runs `run_model_guide(page_id)` for the "Guide from model" button.
- **`yolo_runtime`**: lazy-loads/caches the two YOLO26n-seg checkpoints (pretrained vs. fine-tuned), runs inference for the "Check original/finetuned YOLO" preview buttons.
- **`training_runtime`**: launches training as a genuine OS subprocess (not a thread — an in-process thread version measured ~100x slower, apparently starved badly sharing the GIL with uvicorn's event loop even though CUDA calls themselves release it). Progress is polled via a per-model-type JSON status file; only one training job (any type) runs at a time. **stdout/stderr go to a log file, not a pipe** — a pipe nobody drains while the process is running fills its OS buffer once a sufficiently chatty script (YOLO training via ultralytics) writes enough output, and the subprocess deadlocks on `write()` forever. This was a real bug, found and fixed; don't revert it.

## Custom models (`model/`)

Two architectural families, four trained models, sharing a spine: a CNN backbone → a single-axis collapse-pool → a small SSM stack → a task head.

- `SlimResNet18` (boundary model's backbone): **no spatial downsampling anywhere** — boundary localization needs pixel-exact columns, so the 1D sequence fed to the SSM maps 1:1 to image columns.
- `LineSlimResNet18` (line-segmenter's backbone): downsamples width 16x (height untouched) — a whole-page input is affordable this way.
- `RowCollapsePool`/`ColumnCollapsePool`: single-head softmax-attention pool collapsing one spatial axis (height for boundaries, width for lines).
- `S4DStack`: a small, position-preserving state-space sequence model (`(B,L,128)->(B,L,128)`, length unchanged) — pre-norm blocks, matches the project's one deliberate "proper" sequence model investment.
- Four trained heads: `boundary` (BoundaryModel — straight+ink-avoiding-loss akshara x-cuts), `line_segmenter` (LineSegmentationModel — page-wide y-band detection), `character_separator` / `line_separator` (flow-matching curved-seam generators — trained, but **not currently used for inference**; `model_runtime.USE_CLASSICAL_SEAM_GENERATION = True` means the classical DP seam (`seam_extraction.find_ink_avoiding_path`) is used instead, because these two models have only ever been trained on flat/straight targets and don't generalize to real curvature yet).
- Training always **resumes from the currently-served checkpoint**, never restarts from a random init, for `boundary`/`line_segmenter` — a deliberate choice after the user explicitly rejected restart-from-scratch behavior.

## YOLO26n-seg integration (`model/train_yolo.py`, `yolo_dataset_export.py`)

A second, independent pipeline wrapping Ultralytics' `yolo26n-seg` (nano instance-segmentation), added to transfer-learn a general-purpose detector onto this task rather than only relying on the hand-built models above.

- **Dataset export** (`yolo_dataset_export.py`): rebuilt fresh on every training run from every saved-annotation page. Two classes: `line` (one polygon per text line, from its top/bottom seam + its own left/right edge) and `character` (one polygon per consecutive `boundaries_px` pair — i.e. literally the per-akshara segments a human already cut). A past bug here (normalizing a y-coordinate *after* clamping it to [0,1] instead of before, collapsing every polygon to near-zero height) was found and fixed by direct geometric verification (shoelace-formula area) — if label geometry ever looks wrong again, verify it the same way before suspecting the model.
- **Training**: always starts from the pretrained COCO checkpoint (`model/checkpoints/yolo26n-seg_pretrained.pt`, fetched once via `model/fetch_yolo_pretrained.py`) on the very first run, then continues from its own fine-tuned checkpoint on every run after — matching the custom models' "never restart from scratch" convention. `imgsz=640, batch=4, workers=0` (not YOLO's larger defaults — `imgsz=960` genuinely CUDA-OOM'd on this 4GB card; `workers=0` avoids Windows spawn-mode DataLoader-worker fragility on a 3-5 image dataset where it buys nothing anyway).
- **Known unresolved issue (as of this snapshot)**: training was run with `freeze=23` (freezes all but the final `Segment26` head, out of YOLO26n-seg's 24 top-level layers) at the user's explicit request to avoid retraining the whole pretrained model. A full 100-epoch run at this freeze depth produced **mAP ≈ 0 for the entire run** — flat loss, no real learning. Diagnosis (verified: dataset labels load cleanly, 0 corrupt; character-label quality is sound, 79/83 lines have real per-akshara boundaries) points to the freeze depth being too aggressive for the domain gap between COCO's natural photos and grayscale manuscript scans — the frozen backbone simply has no representation for this content for the head to build on. A partial-unfreeze comparison test (`freeze=10`, i.e. only freeze the earliest generic backbone conv/C3k2/SPPF layers, let the neck+head adapt) was queued but not yet completed before this snapshot was written — **this is the next thing to verify and, if it confirms real learning, make the new default** (`FREEZE_LAYERS` constant in `model/train_yolo.py`).

## Frontend (`frontend/app.js`, single file, ~1900 lines)

Canvas-based editor over one page at a time. No annotation shown until "Show annotation" is toggled on and no page auto-initializes a CV-assist proposal on load — both are explicit user actions now (`btn-init-cv`), per an earlier design change away from silent auto-population.

Editing tools, each a `*Mode` flag in `state` with its own start/cancel/confirm functions, mutually exclusive via the mousedown/click/keydown handlers:
- **Split line** / **New line** / **Extend line**: all three trace curves via the same point-array + Catmull-Rom interpolation (`interpolateSeam`) machinery. A shared **pencil tool** (click = one point as before; click-and-drag = dense freehand sampling every ~2px; hold Shift while dragging = collapse to one straight segment from the drag's start) backs all three. Extend line is New line's confirm/cancel code with one added branch: if `state.extendingLineIdx != null`, modify the selected line in place (union its edges outward, replace its top/bottom curve) instead of pushing a new line.
- **Scissor**: a single click-and-drag gesture classified by stroke width — a narrow stroke snips the nearest boundary, a wide stroke trims a line's edge inward to its nearest interior boundary (for cutting scanner-margin noise off without deleting the whole line).
- **Model/YOLO preview overlays**: "Show model output" (custom models, blue dashed) and "Check original/finetuned YOLO" (orange/purple polygons) are read-only previews that never touch the saved annotation; only "Adopt model output" copies the custom-model preview into the editable state (YOLO's polygon output has no "Adopt" — it doesn't map onto the `boundaries_px` schema).

Training buttons (`Train boundaries`, `Train lines`, `Train YOLO`, `Update model`) all POST to `/retrain` with `resume=true` and poll `/retrain/status`; only one can run at a time (enforced server-side), and all of them train across **every** saved-annotation page, not just the one currently open.

## Known constraints worth remembering

- **4GB GPU, shared with the uvicorn server process itself** (which holds its own small CUDA context from inference calls) — training jobs must stay modest in `batch`/`imgsz`, and two GPU jobs must never run concurrently (verified this directly: running a manual CLI test alongside a UI-triggered job starves both).
- **Windows pipe buffers deadlock undrained subprocess output** — any new training script added to `training_runtime.py` must go through the existing log-file redirection, not stdout/stderr pipes.
- Annotated dataset is tiny (5 pages as of this snapshot) — every dataset/training design here explicitly accepts "no meaningful held-out validation split," judging quality by visual/numeric inspection against the known pages instead.
