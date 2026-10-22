"""Quantitative comparison of the trained models' predictions against the
user's hand-corrected ground-truth annotation.

Replaces ad-hoc "print true vs predicted, eyeball it" checks with real
metrics:
  1. Line-level: greedy best-IoU matching between predicted and true line
     y-bands. Reports matched/missed/spurious line counts and an IoU
     distribution.
  2. Boundary-level (within matched line pairs only): nearest-position
     matching of INTERIOR boundaries (structural edges 0/width excluded)
     within a tolerance window. Reports mean/median absolute error, %
     within 5px/10px, and missed/spurious boundary counts.

Re-run any time after retraining or additional annotation:
    python -m scripts.check_model_quality --page page_0002
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"

IOU_MATCH_THRESHOLD = 0.5
BOUNDARY_MATCH_WINDOW_PX = 15


def interval_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    a0, a1 = a
    b0, b1 = b
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    if union <= 0:
        return 0.0
    return inter / union


def match_lines(true_lines: list[dict], pred_lines: list[dict]) -> list[tuple[int, int, float]]:
    """Greedy best-IoU matching. Returns list of (true_idx, pred_idx, iou)."""
    pairs = []
    for ti, t in enumerate(true_lines):
        for pi, p in enumerate(pred_lines):
            iou = interval_iou((t["y0"], t["y1"]), (p["y0"], p["y1"]))
            if iou > 0:
                pairs.append((iou, ti, pi))
    pairs.sort(reverse=True)

    used_true, used_pred = set(), set()
    matches = []
    for iou, ti, pi in pairs:
        if ti in used_true or pi in used_pred:
            continue
        used_true.add(ti)
        used_pred.add(pi)
        matches.append((ti, pi, iou))
    return matches


def match_boundaries(true_b: list[int], pred_b: list[int], window: int) -> dict:
    """Nearest-position greedy matching within `window` px. true_b/pred_b
    are INTERIOR boundaries only (edges already excluded)."""
    pairs = []
    for ti, tv in enumerate(true_b):
        for pi, pv in enumerate(pred_b):
            dist = abs(tv - pv)
            if dist <= window:
                pairs.append((dist, ti, pi))
    pairs.sort()

    used_true, used_pred = set(), set()
    errors = []
    for dist, ti, pi in pairs:
        if ti in used_true or pi in used_pred:
            continue
        used_true.add(ti)
        used_pred.add(pi)
        errors.append(dist)

    return {
        "errors": errors,
        "n_true": len(true_b),
        "n_pred": len(pred_b),
        "n_missed": len(true_b) - len(used_true),
        "n_spurious": len(pred_b) - len(used_pred),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page", type=str, default="page_0002")
    args = parser.parse_args()

    ann_path = ANNOTATIONS_DIR / f"{args.page}.json"
    truth = json.loads(ann_path.read_text(encoding="utf-8"))
    true_lines = truth["lines"]

    from backend.services.model_runtime import run_model_guide

    pred = run_model_guide(args.page)
    pred_lines = pred["lines"]

    print(f"=== Model quality check: {args.page} ===")
    print(f"line_source: {pred.get('line_source')}")
    print(f"True lines: {len(true_lines)}   Predicted lines: {len(pred_lines)}\n")

    # ---- Line-level ----
    matches = match_lines(true_lines, pred_lines)
    ious = [m[2] for m in matches]
    good_matches = [m for m in matches if m[2] >= IOU_MATCH_THRESHOLD]

    n_missed_lines = len(true_lines) - len(good_matches)
    n_spurious_lines = len(pred_lines) - len(good_matches)

    print("--- Line-level (y-band) ---")
    print(f"Matched (IoU >= {IOU_MATCH_THRESHOLD}): {len(good_matches)} / {len(true_lines)} true lines")
    print(f"Missed true lines (no adequate match): {n_missed_lines}")
    print(f"Spurious predicted lines (no adequate true match): {n_spurious_lines}")
    if ious:
        print(f"IoU over all found pairs -- mean: {statistics.mean(ious):.3f}  "
              f"median: {statistics.median(ious):.3f}  min: {min(ious):.3f}  max: {max(ious):.3f}")
    print()

    # ---- Boundary-level (only within good line matches) ----
    all_errors = []
    total_missed_b = 0
    total_spurious_b = 0
    total_true_b = 0

    print("--- Boundary-level (interior boundaries, matched lines only) ---")
    for ti, pi, iou in good_matches:
        t = true_lines[ti]
        p = pred_lines[pi]
        true_interior = t["boundaries_px"][1:-1]
        pred_interior = p["boundaries_px"][1:-1]
        result = match_boundaries(true_interior, pred_interior, BOUNDARY_MATCH_WINDOW_PX)
        all_errors.extend(result["errors"])
        total_missed_b += result["n_missed"]
        total_spurious_b += result["n_spurious"]
        total_true_b += result["n_true"]

    if all_errors:
        within5 = sum(1 for e in all_errors if e <= 5) / len(all_errors) * 100
        within10 = sum(1 for e in all_errors if e <= 10) / len(all_errors) * 100
        print(f"Matched boundaries: {len(all_errors)} / {total_true_b} true interior boundaries "
              f"(in matched lines)")
        print(f"Mean abs error: {statistics.mean(all_errors):.2f}px   "
              f"Median: {statistics.median(all_errors):.2f}px   Max: {max(all_errors)}px")
        print(f"Within 5px: {within5:.1f}%   Within 10px: {within10:.1f}%")
    else:
        print("No matched boundaries found.")
    print(f"Missed true boundaries: {total_missed_b}   Spurious predicted boundaries: {total_spurious_b}")
    print()

    # ---- Verdict ----
    line_pct = (len(good_matches) / len(true_lines) * 100) if true_lines else 0.0
    b_pct_5px = (sum(1 for e in all_errors if e <= 5) / total_true_b * 100) if total_true_b else 0.0
    avg_err = statistics.mean(all_errors) if all_errors else float("nan")
    print("=== Verdict ===")
    print(f"{line_pct:.0f}% of true lines correctly detected; of matched lines' boundaries, "
          f"{b_pct_5px:.0f}% land within 5px of ground truth (avg error {avg_err:.1f}px).")


if __name__ == "__main__":
    main()
