# Grantha Boundary Annotator

A browser-based annotation and training tool for segmenting scanned Grantha-script (South Indian) manuscript pages — down to the **line** level and the **akshara (character)** level — and turning those corrections directly into training data for several models that try to automate the same job.

You trace/correct line bands and character boundaries on a scanned page in a canvas editor; every save becomes a timestamped training example. A handful of custom PyTorch models and a fine-tuned YOLO26n-seg model train on exactly that saved data, and you can preview any of their output back on the page before trusting it.

## Features

- **Canvas editor**: zoom/pan, click to add an akshara boundary, drag to move one, a pencil tool (click = point, drag = freehand, Shift+drag = straight segment) for tracing curved line/character separators.
- **Split line / New line / Extend line**: fix merged or wavy lines, draw a brand-new one, or grow an existing one's edges outward — all with the same curve-tracing controls.
- **Scissor tool**: one gesture to either snip a stray boundary or trim a line's edge inward, for cutting scanner-margin noise without deleting the whole line.
- **CV-assist bootstrap**: a classical (non-ML) first pass proposes line bands + boundaries for a blank page, explicitly on request — nothing is auto-populated.
- **Four custom models** (`model/`): per-line akshara boundary detection, page-wide line-band detection, and two curved-seam generators — a shared CNN → attention-pool → small state-space-model spine, trained from scratch on your saved annotations.
- **YOLO26n-seg integration**: a second, independent pipeline fine-tuning Ultralytics' pretrained YOLO26n-seg on two classes (`line`, `character`) exported straight from your saved boundaries — "Check original/finetuned YOLO" previews either checkpoint's raw output.
- **Resume-always training**: every training button continues from the current checkpoint, never restarts from scratch, and trains across every page that has a saved annotation (not just the one open).
- **Version history**: every save is kept as an immutable snapshot, browsable and reloadable from the UI.

## Quick start

```bash
pip install -r requirements.txt

# torch needs the CUDA-specific index for a GPU build -- see requirements.txt's comment
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics==8.4.143

python -m uvicorn backend.main:app --host 127.0.0.1 --port 5050
```

Open `http://127.0.0.1:5050/`. Pick a page from the dropdown; a blank (never-annotated) page shows just the image until you click **Initialize CV proposals**.

To add more source pages, render them from the project's PDF:
```bash
python -m scripts.extract_page --page N --dpi 180          # one page
python -m scripts.extract_all_pages --dpi 180               # every remaining page
```

## Project layout

```
backend/    FastAPI app -- routes + services (annotation storage, CV-assist,
            model inference, YOLO inference, training orchestration)
frontend/   the canvas editor (index.html + app.js + style.css, no build step)
model/      custom PyTorch models, training scripts, dataset classes,
            and the YOLO fine-tuning pipeline
data/       rendered page images + saved annotations (+ version history)
scripts/    one-off CLIs (render pages, run CV-assist standalone, etc.)
tests/      a handful of pytest model-size/shape checks
docs/       architecture notes -- see docs/current-arch.md, docs/architectures-experimented.md, and docs/claude.md
```

See **`docs/current-arch.md`** for a full architecture writeup (data schema, model internals, the YOLO integration and its currently-open tuning question, known hardware constraints), **`docs/architectures-experimented.md`** for every modeling approach tried along the way (including ones trained but not currently used, and ones designed but never built), and **`docs/claude.md`** for working conventions if you're picking this project back up with Claude Code.

## Status

Actively evolving — this is a working research/annotation tool, not a packaged product. Only 5 pages are annotated so far; every model here is trained on a deliberately small dataset and is expected to improve as more pages get corrected.
