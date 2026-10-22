# Architectures experimented

Every modeling approach tried in this project, in roughly the order they were built on top of each other, with what each one is, why it was built that way, and its current status. `docs/current-arch.md` describes what's actually wired into the live app today; this file is the fuller history, including things that were trained but aren't used, and things that were designed but never built.

Status key: **ACTIVE** (what the app actually calls today) · **TRAINED, UNUSED** (checkpoint exists, code path exists, but inference doesn't currently call it) · **ABANDONED** (tried, evidence against it, dropped) · **CONSIDERED, NOT BUILT** (designed/discussed, never implemented).

---

## 0. Classical CV-assist (non-ML) — **ACTIVE**, bootstrap only

`backend/services/cv_assist.py`. The first-pass proposal for a brand-new, never-annotated page: Sauvola adaptive binarization (robust to uneven scanner illumination) → horizontal projection profile → text-line bands → per-line vertical projection profile → valley detection for akshara boundaries.

Explicitly **not** a clustering approach (e.g. DBSCAN on connected components): a kootu ezhuthu (conjunct/ligature glyph cluster) can have components spatially disjoint from their own base letter while touching a neighboring independent letter — pure proximity has no notion of "this gap is internal to one glyph" vs. "this is a real inter-akshara boundary." The line-relative, shape-aware projection-profile approach was chosen instead.

Used once per page, on explicit request ("Initialize CV proposals" / "Reset to proposals") — never auto-run, never used as a trained model's replacement.

## 1. BoundaryModel — per-line akshara x-boundary detection — **ACTIVE**

`model/boundary_model.py`, `model/backbone.py`, `model/ssm.py`, `model/diffusion_head.py`, `model/config.py::ModelConfig`.

Pipeline: `SlimResNet18` → `RowCollapsePool` → `S4DStack` → presence head, with a `DiffusionSelfCorrector` refinement pass.

- **`SlimResNet18`**: ResNet-18 topology at 1/4 channel width (16/32/64/128 instead of 64/128/256/512), **stride 1 everywhere — no spatial downsampling at all**. Deliberate: akshara segments can be ~12px wide, and any downsampling would blur boundary localization past usefulness. The 1D sequence fed to the SSM maps 1:1 to image columns.
- **`RowCollapsePool`**: single-head softmax-attention pool collapsing the height axis, keeping width — one feature vector per column.
- **`S4DStack`** (`model/ssm.py`): a hand-written, pure-PyTorch S4D (diagonal state-space model), explicitly **not** the official `mamba-ssm` package (its selective-scan/causal-conv1d kernels are custom CUDA builds with no Windows wheels). Implemented via complex tensors + FFT (O(L log L) convolutional form of the SSM's impulse response) so it runs on CPU or CUDA on Windows with no build step. 4 layers, `d_state=64`, pre-norm GLU-FFN blocks.
- **`DiffusionSelfCorrector`**: a DDPM-style head, but explicitly **not** literal generative diffusion — it's framed as iterative self-correction. Forward process only exists for training (noise a ground-truth Gaussian-bump signal via the standard DDPM marginal); at inference, instead of starting from pure noise, it starts from the SSM head's own one-shot boundary logits (already a decent estimate) and runs `T=4` deterministic DDIM-style reverse steps to refine them — x0-parameterization (predicts the clean signal directly, more stable than noise-prediction at such a small `T`). Operates on 256px sliding windows (50% overlap, stitched by averaging) so a full page never needs windowing in one shot.
- **Loss** (`model/losses.py`): focal BCE (presence) + soft-argmax localization + diffusion x0-MSE + **ink-avoidance** (`ink_avoidance_loss`) — penalizes predicted boundary probability sitting on ink (SDT energy < 0), contributes exactly zero at background columns so it never fights real boundary predictions. Added directly in response to "train boundaries should not pass through the blacks."
- Trains by always resuming from the current checkpoint (never a random-init restart), across every saved-annotation page.

## 2. LineSegmentationModel — page-wide line-band detection — **ACTIVE**

`model/line_segmenter.py`, `model/config.py::LineModelConfig`. Same spine shape as BoundaryModel, transposed and rescaled for a whole-page input:

- **`LineSlimResNet18`**: same 4-stage/16-32-64-128 topology, but downsamples **width** 16x (height untouched) — a line's y-position needs full resolution, but a page is ~900px wide and a resolution-preserving backbone over a full page was too large for the 4GB VRAM budget.
- **`ColumnCollapsePool`**: collapses width, keeps height — one feature vector per row.
- Same `S4DStack` + `DiffusionSelfCorrector` pattern as BoundaryModel, but `window_px=4096` (larger than any real page) — a full page height (~1300px) fits in one shot, no sliding window needed; line gaps are coarser/lower-frequency than akshara gaps.
- Needed noticeably more epochs than BoundaryModel to converge correctly: 150 epochs converged to a degenerate "almost everything is one giant line" solution; 300 resolved it properly. `DEFAULT_EPOCHS["line_segmenter"] = 300` reflects this directly, not a guess.

## 3/4. CharacterSeparatorModel / LineSeparatorModel — curved seam generation via flow matching — **TRAINED, UNUSED**

`model/character_separator_model.py`, `model/line_separator_model.py`, `model/flow_matching.py`.

The evolution story here is the most interesting one in the project:

1. **Why not just threshold the presence head's output for a curved cut?** BoundaryModel/LineSegmentationModel's presence heads produce a scalar likelihood per position, decoded via thresholding/peak-picking — that decode step is exactly what caps the output at straight, axis-aligned cuts. A real baseline is not always straight ("people write like mountain").
2. **Conditional Flow Matching** (Lipman et al. 2022; `PathFlowMatcher` in `flow_matching.py`) was built as the fix: instead of a per-position scalar decoded afterward, the model directly **generates** a length-L path (one y-value per column for a line separator, one x-value per row for a character separator) — curvature is representable by construction, not lost at decode time. Same architectural lineage as `DiffusionSelfCorrector` (a dilated-Conv1d stack conditioned on SSM hidden features), swapped from a discrete DDPM timestep to a continuous flow-matching time variable; same "start from a decent guess, not from scratch" self-correcting framing (the prior sample `x0` is the coarse detector's anchor, not noise).
3. **LineSeparatorModel**: `y(x)` over the full page width → needs per-column features (`RowCollapsePool`, same as BoundaryModel) and `LineSlimResNet18` (16x width downsample — running the full-resolution backbone over a full page width was measured directly: single-digit epochs took minutes, far too slow). Generates the path at 1/16 width resolution, upsamples only the final path via linear interpolation.
4. **CharacterSeparatorModel**: `x(y)` within one line's own height → needs per-row features (`ColumnCollapsePool`, same as LineSegmentationModel) and the full resolution-preserving `SlimResNet18` (both strip dimensions here — a narrow width window and one line's height, ~15-45px — are already small enough to be cheap).
5. **Loss**: `energy_avoidance_loss` (`model/seam_energy_loss.py`) — same SDT-energy-based ink-avoidance principle as BoundaryModel's `ink_avoidance_loss`, adapted from a dense per-column probability to a single sampled path position.

**Why unused despite being trained**: `CharacterSeamDataset`'s own docstring states plainly that there is currently no curved ground truth for character separators — the split-line editor only supported straight cuts at the time these were built, so every training target is a flat vertical/horizontal line at the boundary's anchor position. Near-zero training loss on these models does **not** mean genuine curvature was learned — it was diagnosed directly (flat output on visual inspection) and is an expected, acknowledged limitation, not a bug. `backend/services/model_runtime.py::USE_CLASSICAL_SEAM_GENERATION = True` is the flag that keeps inference on the classical method (#5) instead; the infrastructure is deliberately kept ready so flipping it to `False` is a one-line change once the annotation tool grows a curved per-character tracing mode (it has, for lines, via Split line/Extend line — a curved per-character mode doesn't exist yet).

## 5. Classical ink-avoiding seam DP — **ACTIVE**, replaces #3/#4 at inference time

`backend/services/seam_extraction.py::find_ink_avoiding_path`. A non-learned dynamic-program seam search over the SDT energy map (negative = inside ink, positive = background), negated from the usual medial-seam DP (`compute_seam_map`, which instead finds a line's own ink-following centerline) so it favors background instead of ink — i.e. it threads the gap between glyphs/lines rather than down their center.

A real bug was found and fixed here during development: the first version only applied the anchor-pull bias (toward the coarse detector's own guess) as a tie-break at the final endpoint, which let the seam wander almost anywhere across the search strip (measured directly: an 80px-wide strip produced a seam spanning 75 of those pixels). Fixed by applying a quadratic per-row bias toward the anchor at **every** column, not just the endpoint.

Chosen as the active method specifically because it needs no training data at all and produces genuinely curved, ink-avoiding output immediately — while #3/#4 mature.

## 6. YOLO26n-seg fine-tuning — **ACTIVE**, ongoing tuning

See `docs/current-arch.md`'s YOLO section for the data export/training mechanics. Architecturally this is a completely separate lineage from #1-#5 (a pretrained, general-purpose Ultralytics detector/segmenter, not a from-scratch custom model), added specifically to transfer-learn from large-scale pretraining rather than only ever training small models from zero.

**Freeze-depth experiment, in progress**: at the user's explicit request to avoid retraining the whole pretrained model, the first full run froze 23 of the model's 24 top-level layers (everything except the final `Segment26` head). Result: **mAP ≈ 0 for the entire 100-epoch run** — not slow learning, no learning. Diagnosed by elimination (dataset labels load cleanly with 0 corrupt entries; character-label quality is sound, 79/83 lines have real per-akshara boundaries) down to the freeze depth itself: the backbone was pretrained entirely on natural COCO photos, and freezing all of it leaves the head with no useful features to build "ink stroke" or "manuscript line" recognition out of — too large a domain gap for last-layer-only transfer. A partial-unfreeze comparison (`freeze=10`, keeping only the earliest generic backbone layers frozen) was queued to verify this diagnosis but not yet completed as of this writing — **next step, not yet resolved**.

## 7. TinyViT text-region semantic segmentation — **CONSIDERED, NOT BUILT** (Phase 0 of 4 substituted with a classical heuristic)

Proposed design (full plan written and approved, but only its first phase shipped): a small patch-based Vision Transformer that learns to tell real text apart from scanner-margin noise/artifacts, whose output would gate the existing BoundaryModel/LineSegmentationModel pipeline (`seq = seq * text_mask` applied post-pool, pre-SSM) so noisy page regions never get "proper attention" fed into the SSM — motivated directly by the observation that "Guide from model" boundaries always touched the raw page edges, even across scanner noise.

- **Phase 0 (shipped, no ML)**: the actual root cause turned out to be a plain train/inference crop mismatch, not something needing a transformer at all — training already cropped each line to its own saved `boundaries_px[0]:[-1]`, but inference fed the model the full page width. Fixed with a classical Sauvola ink-column heuristic (percentile-of-smoothed-profile + minimum-run-width, to avoid a thin scanner rule mark masquerading as text) deriving `x0,x1` before the inference crop. Verified: median boundary error dropped from ~90-450px to ~4-6px.
- **Phases 1-4 (TinyViT module, two-stage self-supervised + weak-supervised training, `encode()` integration, final wiring)**: designed in detail (patch embedding via strided Conv2d, fixed 2D sinusoidal position encoding, 4-layer Transformer encoder, Stage A ink-density self-supervised pretrain + Stage B weak-supervised fine-tune with an ignore-mask for unreviewed lines) but **never implemented** — Phase 0 alone resolved the immediate symptom well enough that the bigger investment hasn't been needed yet. The corrected injection point (a post-pool multiplicative gate, not an additive pre-softmax bias — the naive approach was proven a mathematical no-op via `torch.allclose` before any code was written) remains documented for whenever this gets picked back up.

## 8. Muon optimizer — **CONSIDERED, NOT ADOPTED**

Asked about directly during development ("rather than adam can we use muon loss fn?"). Clarified and not pursued: Muon is an optimizer (a replacement for AdamW, orthogonalizing gradient updates via Newton-Schulz iteration), not a loss function, and does not have a official Windows-friendly pure-PyTorch package the way this project's other components do. Given the dataset size here (a handful of annotated pages), the optimizer choice was judged not to be a bottleneck worth the integration cost; AdamW remains in use everywhere.
