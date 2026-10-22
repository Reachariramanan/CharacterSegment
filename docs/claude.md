# Working notes for Claude Code in this repo

Read `docs/current-arch.md` first for what exists and why. This file is conventions and gotchas — things that aren't obvious from reading the code once, learned the hard way over this project's history.

## Running it

```
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu124        # see requirements.txt's comment -- CUDA build, not installable via the file itself
pip install torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics==8.4.143
python -m uvicorn backend.main:app --host 127.0.0.1 --port 5050
```
Open `http://127.0.0.1:5050/`. `ultralytics` pulls in the latest torch/torchvision as a dependency and **will silently replace the CUDA build with a CPU-only one** if you ever reinstall it or its deps — reinstall torch+torchvision from the CUDA index afterward and verify with `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` before assuming training is just being slow.

## Hard rules, each backed by a real incident

- **Never restart training from scratch once a checkpoint exists.** Every `Train *` button resumes from the currently-served checkpoint by default. This was an explicit, repeated user correction earlier in the project — don't reintroduce a "restart from random init" default anywhere.
- **Only pages with a saved annotation (`data/annotations/page_NNNN.json`) are trainable.** Never feed a page that's only ever had an in-memory/unsaved CV-assist proposal into a training set.
- **A line's horizontal extent is `boundaries_px[0]`/`[-1]`, never `0`/page-width.** This bit training/inference mismatch once already (boundaries always touching page edges, even past scanner-margin noise) — any new code that crops a line must read these, not assume full width.
- **Subprocess stdout/stderr must go to a log file, never a bare `subprocess.PIPE` left undrained.** A sufficiently verbose training script (ultralytics is one) will fill the pipe's OS buffer and deadlock on write() forever — this happened, looked exactly like "training is stuck," and cost real debugging time before the root cause was found. `training_runtime.py`'s log-file redirection is the fix; don't revert to `PIPE` for a new model type.
- **Never run two GPU training jobs at once**, including a manual CLI test alongside a UI-triggered job. This card has 4GB VRAM, shared with the server process's own CUDA context. `training_runtime.any_running()` already enforces this for UI-triggered jobs — if you start one by hand (e.g. to smoke-test a change), check that flag/the status JSON first.
- **Verify a geometry/label bug with actual numbers (shoelace area, bbox width/height), not by re-reading the code.** The YOLO dataset-export y-coordinate bug (clamp-before-normalize, collapsing every polygon to zero height) looked fine on a code read and only showed up when real widths/heights were computed and printed.
- Before declaring a training run "it's learning fine," check mAP/loss **from epoch 1**, not just the tail of the log — a flat near-zero mAP for the *entire* run (as opposed to a slow climb) means the model isn't learning at all, not that it needs more epochs. This is exactly what happened with `freeze=23` (see current-arch.md's open item).

## Conventions to match when adding a model type

Every `model/train_*.py` follows the same shape: `argparse` with `--pages`/`--epochs`/`--resume`/`--status-file`, a `progress_callback`/`write_status` call per epoch, and gets registered in `backend/services/training_runtime.py`'s `MODEL_TYPES`/`TRAIN_MODULES` plus `backend/api/routes_model.py`'s `DEFAULT_EPOCHS`/`DEFAULT_EPOCHS_RESUME`/`RESUME_CHECKPOINTS`/`VALID_MODEL_TYPES`. Frontend wiring mirrors this: add the model type to `MODEL_LABELS` and `FULL_TRAIN_EPOCHS` in `app.js`, a button in `index.html`.

## What NOT to add casually

- No new dedicated training button without checking first — the user explicitly pushed back once on "too many options" and two buttons were removed as a result. Prefer folding a new capability into an existing flow over adding UI surface.
- Don't re-enable `character_separator`/`line_separator` as the active inference path without first checking `model_runtime.USE_CLASSICAL_SEAM_GENERATION` — they're trained but known to only produce flat (non-curved) output; the classical DP seam is the deliberate current default.
