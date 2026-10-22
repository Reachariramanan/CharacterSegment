"""Loads a trained BoundaryModel checkpoint (if one exists) and runs inference
to produce boundary proposals, as an alternative to cv_assist. Until a
checkpoint has been trained (requires annotated data, which doesn't exist yet
for a fresh project), callers should treat `is_available()` as the gate and
surface an explicit "not trained yet" response rather than silently falling
back to CV-assist.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "model" / "checkpoints" / "boundary_model.pt"
LINE_CHECKPOINT_PATH = PROJECT_ROOT / "model" / "checkpoints" / "line_segmenter_pageonly.pt"
LINE_SEPARATOR_CHECKPOINT_PATH = PROJECT_ROOT / "model" / "checkpoints" / "line_separator_pageonly.pt"
CHARACTER_SEPARATOR_CHECKPOINT_PATH = PROJECT_ROOT / "model" / "checkpoints" / "character_separator_pageonly.pt"

_model = None  # lazily loaded, module-level cache
_line_model = None
_line_separator_model = None
_line_separator_margin_px = None
_character_separator_model = None
_character_separator_margin_px = None

# The learned seam-generator checkpoints (LineSeparatorModel,
# CharacterSeparatorModel) have so far only ever been trained on FLAT
# targets -- no curved ground truth exists yet (see
# model/character_seam_dataset.py) -- so a near-zero training loss there
# does not mean they've learned genuine ink-avoiding curvature. Until they
# have (real curved annotations + a demonstrated non-flat result), seam
# generation uses a classical DP method instead (see
# seam_extraction.find_ink_avoiding_path) -- ink-avoiding by construction,
# needs no training data, and is what _generate_curved_edge /
# _generate_character_seam fall back to below. The DL models still load
# normally (their margin_px is reused either way); flip this back to False
# once they're ready.
USE_CLASSICAL_SEAM_GENERATION = True


def is_available() -> bool:
    return CHECKPOINT_PATH.exists()


def _device():
    """Runs inference on the RTX 500 Ada if available -- these models are
    tiny (a few MB) so this is mainly about avoiding the extra latency of
    CPU fallback on every /model-guide call, not a memory concern."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_model():
    global _model
    if _model is not None:
        return _model

    import torch

    from model.boundary_model import BoundaryModel

    model = BoundaryModel()
    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    model.to(_device())
    _model = model
    return _model


def _load_line_model():
    global _line_model
    if _line_model is not None:
        return _line_model
    if not LINE_CHECKPOINT_PATH.exists():
        return None

    import torch

    from model.config import LineModelConfig
    from model.line_segmenter import LineSegmentationModel

    model = LineSegmentationModel(LineModelConfig())
    model.load_state_dict(torch.load(LINE_CHECKPOINT_PATH, map_location="cpu"))
    model.eval()
    model.to(_device())
    _line_model = model
    return _line_model


def _load_line_separator_model():
    """Flow-matching model that generates a CURVED line-separator seam
    (y as a function of x), replacing the straight y0/y1 rectangle a
    detected line otherwise gets. Returns None if no checkpoint exists yet
    (callers should fall back to the straight line_bands result)."""
    global _line_separator_model, _line_separator_margin_px
    if _line_separator_model is not None:
        return _line_separator_model, _line_separator_margin_px
    if not LINE_SEPARATOR_CHECKPOINT_PATH.exists():
        return None, None

    import torch

    from model.line_separator_model import LineSeparatorModel

    checkpoint = torch.load(LINE_SEPARATOR_CHECKPOINT_PATH, map_location="cpu")
    model = LineSeparatorModel()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.to(_device())
    _line_separator_model = model
    _line_separator_margin_px = checkpoint["margin_px"]
    return _line_separator_model, _line_separator_margin_px


def _load_character_separator_model():
    """Flow-matching model that generates a curved character/akshara
    separator seam (x as a function of y within one line), nudged away from
    ink via the same SDT-energy avoidance loss as the line separator (see
    model/seam_energy_loss.py) -- this is what keeps a generated boundary
    from cutting through a character's stroke instead of just running
    straight through the middle of it. Returns None if no checkpoint exists
    yet (callers fall back to BoundaryModel's straight boundaries_px)."""
    global _character_separator_model, _character_separator_margin_px
    if _character_separator_model is not None:
        return _character_separator_model, _character_separator_margin_px
    if not CHARACTER_SEPARATOR_CHECKPOINT_PATH.exists():
        return None, None

    import torch

    from model.character_separator_model import CharacterSeparatorModel

    checkpoint = torch.load(CHARACTER_SEPARATOR_CHECKPOINT_PATH, map_location="cpu")
    model = CharacterSeparatorModel()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.to(_device())
    _character_separator_model = model
    _character_separator_margin_px = checkpoint["margin_px"]
    return _character_separator_model, _character_separator_margin_px


def _generate_character_seam(
    model, margin_px: int, gray: "np.ndarray", energy: "np.ndarray", y0: int, y1: int, anchor_x: float
):
    """Returns a generated character-separator seam within line rows
    [y0, y1) around `anchor_x` (one of BoundaryModel's straight
    boundaries_px), as a list of ints (page-global pixel x per row), or None
    if too close to the page edge for this margin to represent. Uses the
    classical DP path (see USE_CLASSICAL_SEAM_GENERATION) by default;
    `model` (CharacterSeparatorModel) is only actually run when that flag
    is False."""
    page_width = gray.shape[1]
    strip_x0 = round(anchor_x - margin_px)
    strip_x1 = round(anchor_x + margin_px)
    if strip_x0 < 0 or strip_x1 > page_width:
        return None

    if USE_CLASSICAL_SEAM_GENERATION:
        from backend.services.seam_extraction import find_ink_avoiding_path

        # Transpose: the character seam is x(row) -- one value per ROW --
        # but find_ink_avoiding_path always returns one value per COLUMN of
        # its input, so the strip's row/column roles need swapping here
        # (unlike the line-separator case below, which is already y(col)).
        energy_strip = energy[y0:y1, strip_x0:strip_x1].T
        anchor_index = round(anchor_x - strip_x0)
        seam_local = find_ink_avoiding_path(energy_strip, anchor_index=anchor_index)
        return [int(strip_x0 + v) for v in seam_local]

    import numpy as np
    import torch

    strip = np.stack([gray[y0:y1, strip_x0:strip_x1] / 255.0, energy[y0:y1, strip_x0:strip_x1]], axis=0)
    strip_tensor = torch.from_numpy(strip.astype("float32")).unsqueeze(0).to(_device())  # (1,2,H,W_strip)

    generated = model.generate_path(
        strip_tensor, strip_x0_global=strip_x0, strip_width=strip_x1 - strip_x0, anchor_x_norm=0.5
    )
    return [round(float(v)) for v in generated.cpu().tolist()]


def _generate_curved_edge(model, margin_px: int, gray: "np.ndarray", energy: "np.ndarray", anchor: float):
    """Returns a generated line-separator seam around `anchor` (a y0 or y1
    scalar), as a list of ints (page-global pixel y per column), or None if
    too close to the page edge for this margin to represent. Uses the
    classical DP path (see USE_CLASSICAL_SEAM_GENERATION) by default;
    `model` (LineSeparatorModel) is only actually run when that flag is
    False."""
    page_height = gray.shape[0]
    strip_y0 = round(anchor - margin_px)
    strip_y1 = round(anchor + margin_px)
    if strip_y0 < 0 or strip_y1 > page_height:
        return None

    if USE_CLASSICAL_SEAM_GENERATION:
        from backend.services.seam_extraction import find_ink_avoiding_path

        energy_strip = energy[strip_y0:strip_y1, :]
        anchor_index = round(anchor - strip_y0)
        seam_local = find_ink_avoiding_path(energy_strip, anchor_index=anchor_index)
        return [int(strip_y0 + v) for v in seam_local]

    import numpy as np
    import torch

    strip = np.stack([gray[strip_y0:strip_y1, :] / 255.0, energy[strip_y0:strip_y1, :]], axis=0)
    strip_tensor = torch.from_numpy(strip.astype("float32")).unsqueeze(0).to(_device())  # (1,2,H,W)

    generated = model.generate_path(
        strip_tensor, strip_y0_global=strip_y0, strip_height=strip_y1 - strip_y0, anchor_y_norm=0.5
    )
    return [round(float(v)) for v in generated.cpu().tolist()]


def _estimate_line_text_extent(
    ink: "np.ndarray",
    y0p: int,
    y1p: int,
    smooth_px: int = 7,
    percentile: float = 85.0,
    frac_of_typical: float = 0.2,
    min_run_width_px: int = 20,
) -> tuple[int, int]:
    """Classical Sauvola-ink heuristic for a line's real left/right text
    extent within row-span [y0p, y1p) -- Phase 0 of the text-region plan
    (see the "Text-region semantic segmentation" plan doc): trims the same
    scanner-margin noise the Scissor tool trims by hand, so /model-guide's
    crop matches what BoundaryWindowDataset already trains on (it crops
    each line to its own annotated boundaries_px[0]/[-1] extent -- see
    model/dataset.py). Without this, predict_boundaries was always fed the
    full untrimmed page width, a train/inference distribution mismatch.

    A naive "threshold relative to this row-span's single busiest column"
    approach was tried first and diagnosed as broken on real data: a thin
    (1-2px), full-height scanner rule/margin mark can have a HIGHER ink
    count than genuine (textured, varying-height) text, so it dominates the
    max and pulls the threshold up past real text's own lower density --
    the naive version measurably under-trimmed by ~90px on page_0002. Fix:
    smooth first (bridges inter-character gaps within real text), reference
    a PERCENTILE of the smoothed profile rather than its raw max (robust to
    one narrow spike), then require a minimum contiguous RUN WIDTH above
    that -- a thin rule/mark fails the width check even at high density,
    genuine text (wide, sustained ink) passes it even at lower density.

    Verified against page_0002's real (human-trimmed) extents: close
    (within a few px) for most lines. Known remaining limitation: a legit
    but spatially-isolated mark separated from the main text by a wide
    blank gap (e.g. a trailing verse-number mark) can still get dropped,
    since it's judged as a separate, too-narrow run -- this is exactly the
    kind of "is this noise or real content" call a shape/density heuristic
    fundamentally can't always get right, which is what Phases 1-4 of the
    plan (the ViT) exist to actually solve properly.

    Returns (x0, x1) in page-global pixel columns, or (0, page_width) for a
    blank row-span or one with no run wide enough to trust."""
    import numpy as np
    import torch

    from backend.services.cv_assist import _moving_average
    from model.peaks import contiguous_runs_above_threshold

    page_width = ink.shape[1]
    col_ink = ink[y0p:y1p, :].sum(axis=0).astype("float32")
    smoothed = _moving_average(col_ink, smooth_px)

    nonzero = smoothed[smoothed > 0]
    if len(nonzero) == 0:
        return 0, page_width
    reference = np.percentile(nonzero, percentile)
    if reference <= 0:
        return 0, page_width

    runs = contiguous_runs_above_threshold(
        torch.from_numpy(smoothed), reference * frac_of_typical, min_run_width_px
    )
    if not runs:
        return 0, page_width
    return int(runs[0][0]), int(runs[-1][1])


def run_model_guide(page_id: str) -> dict:
    """Run the trained model on a page's line crops to produce boundary
    proposals. Raises FileNotFoundError if no checkpoint exists -- callers
    (the API route) should translate that into an explicit 501 response."""
    if not is_available():
        raise FileNotFoundError(
            f"No trained checkpoint at {CHECKPOINT_PATH}. Train a model first "
            "(see model/train.py) before using model-guided proposals."
        )

    import numpy as np
    import torch
    from PIL import Image

    from backend.services.annotation_store import IMAGES_DIR, load_meta
    from backend.services.cv_assist import CvAssistConfig, _detect_text_lines, _binarize_sauvola
    from backend.services.seam_extraction import compute_normalized_energy_channel

    model = _load_model()

    meta = load_meta(page_id)
    png_path = IMAGES_DIR / f"{page_id}.png"
    gray = np.array(Image.open(png_path).convert("L"))

    # Prefer the trained LineSegmentationModel for y-ranges (line bands) if a
    # checkpoint exists; otherwise fall back to cv_assist's classical
    # projection-profile line detector. `cfg` is needed either way, below,
    # for the per-line vertical padding applied before x-boundary inference.
    # `ink` is needed unconditionally now too (not just in the fallback
    # branch) -- see _estimate_line_text_extent below.
    cfg = CvAssistConfig()
    ink = _binarize_sauvola(gray, cfg)
    line_model = _load_line_model()
    if line_model is not None:
        crop = gray.astype("float32") / 255.0
        x = torch.from_numpy(crop).unsqueeze(0).unsqueeze(0).to(_device())
        line_bands = line_model.predict_line_bands(x)
    else:
        line_bands = _detect_text_lines(ink, cfg)

    # Curved line separators (flow matching) -- optional, replaces each
    # line's straight y0/y1 rectangle with a generated top_seam_px/
    # bottom_seam_px pair when a trained checkpoint exists. Computed once
    # per page (not per line): the energy channel is a full-page array.
    separator_model, separator_margin_px = _load_line_separator_model()
    character_separator_model, character_separator_margin_px = _load_character_separator_model()
    energy = (
        compute_normalized_energy_channel(gray)
        if (separator_model is not None or character_separator_model is not None)
        else None
    )

    lines = []
    for idx, (y0, y1) in enumerate(line_bands):
        y0p = max(0, y0 - cfg.line_vpad_px)
        y1p = min(gray.shape[0], y1 + cfg.line_vpad_px)
        # Trim to this line's own real text extent (Phase 0 of the
        # text-region plan) -- matches BoundaryWindowDataset's training
        # crop convention (model/dataset.py), instead of always feeding the
        # model the full untrimmed page width the way this used to.
        x0, x1 = _estimate_line_text_extent(ink, y0p, y1p)
        crop = gray[y0p:y1p, x0:x1].astype("float32") / 255.0
        tensor = torch.from_numpy(crop).unsqueeze(0).unsqueeze(0).to(_device())  # (1,1,H,W)

        with torch.no_grad():
            boundaries_px = model.predict_boundaries(tensor)
        boundaries_px = [x0 + b for b in boundaries_px]

        line_out = {
            "line_id": f"L{idx + 1:02d}",
            "y0": y0,
            "y1": y1,
            "boundaries_px": boundaries_px,
            "boundary_source": ["proposed"] * len(boundaries_px),
            "akshara_labels": [None] * max(0, len(boundaries_px) - 1),
        }

        if separator_model is not None:
            top_seam = _generate_curved_edge(separator_model, separator_margin_px, gray, energy, y0)
            bottom_seam = _generate_curved_edge(separator_model, separator_margin_px, gray, energy, y1)
            if top_seam is not None:
                line_out["top_seam_px"] = top_seam
            if bottom_seam is not None:
                line_out["bottom_seam_px"] = bottom_seam

        if character_separator_model is not None and len(boundaries_px) > 2:
            # One entry per boundaries_px, parallel array -- None for the
            # two structural edges (never generated/moved) and wherever the
            # margin would run off the page; the frontend falls back to a
            # straight vertical line for any None entry.
            seams = [None] * len(boundaries_px)
            for i, bx in enumerate(boundaries_px[1:-1], start=1):
                seams[i] = _generate_character_seam(
                    character_separator_model, character_separator_margin_px, gray, energy, y0, y1, bx
                )
            if any(s is not None for s in seams):
                line_out["boundary_seams_px"] = seams

        lines.append(line_out)

    return {
        "page_id": page_id,
        "source_pdf": meta["source_pdf"],
        "pdf_page_index_1based": meta["pdf_page_index_1based"],
        "image_path": f"data/images/{page_id}.png",
        "dpi": meta["dpi"],
        "image_width_px": meta["width_px"],
        "image_height_px": meta["height_px"],
        "rotation_deg_applied": meta["rotation_deg"],
        "line_source": "model_v1",
        "lines": lines,
    }
