"""Training for BoundaryModel (per-line akshara x-boundaries).

Usage:
    python -m model.train --pages page_0002 --epochs 200

`train_boundary_model` is the reusable entrypoint (used by both this CLI and
the backend's /retrain endpoint, which runs it in a background thread so a
UI-triggered training run doesn't block the server); `main()` is a thin CLI
wrapper around it.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader

from model.boundary_model import BoundaryModel
from model.config import ModelConfig
from model.dataset import BoundaryWindowDataset
from model.losses import total_loss

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "model" / "checkpoints"
DEFAULT_OUT_PATH = CHECKPOINT_DIR / "boundary_model_pageonly.pt"
SERVED_CHECKPOINT_PATH = CHECKPOINT_DIR / "boundary_model.pt"


def collate(batch):
    """Pads variable-height (different lines have different heights) and
    variable-width (only the final, possibly-shorter tail window of a line
    differs from `window_px`) crops to the batch max on each dimension.
    energy_col is padded with +1.0 (background) rather than the default 0
    (which would read as "deep inside ink" and wrongly penalize the padded
    region -- see model/losses.py:ink_avoidance_loss)."""
    max_height = max(crop.shape[-2] for crop, _, _ in batch)
    max_width = max(crop.shape[-1] for crop, _, _ in batch)
    crops, targets, energy_cols, boundaries_batch = [], [], [], []
    for crop, target, energy_col in batch:
        pad_h = max_height - crop.shape[-2]
        pad_w = max_width - crop.shape[-1]
        if pad_h > 0 or pad_w > 0:
            crop = torch.nn.functional.pad(crop, (0, pad_w, 0, pad_h))
            target = torch.nn.functional.pad(target, (0, pad_w))
            energy_col = torch.nn.functional.pad(energy_col, (0, pad_w), value=1.0)
        crops.append(crop)
        targets.append(target)
        energy_cols.append(energy_col)
        nonzero = (target > 0.5).nonzero(as_tuple=True)[0].tolist()
        boundaries_batch.append(nonzero)
    return torch.stack(crops), torch.stack(targets), torch.stack(energy_cols), boundaries_batch


def train_boundary_model(
    pages: list[str],
    max_lines: int | None = None,
    epochs: int = 150,
    batch_size: int = 8,
    lr: float = 1e-3,
    resume: Path | None = None,
    out_path: Path = DEFAULT_OUT_PATH,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> Path:
    """Trains BoundaryModel and saves the checkpoint to `out_path`, returning
    that path. progress_callback(epoch, total_epochs, avg_loss) is called
    once per epoch if given -- the retrain API route uses this to update a
    shared status dict the UI polls."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = BoundaryWindowDataset(pages, max_lines=max_lines)
    if len(dataset) == 0:
        raise RuntimeError(
            f"No training windows found for pages {pages}. "
            "Annotate at least one page via the UI and save it first."
        )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)

    cfg = ModelConfig()
    model = BoundaryModel(cfg).to(device)
    if resume and resume.exists():
        model.load_state_dict(torch.load(resume, map_location=device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        epoch_losses = []
        for crops, targets, energy_cols, boundaries_batch in loader:
            crops, targets, energy_cols = crops.to(device), targets.to(device), energy_cols.to(device)

            logits, hidden = model.encode(crops)
            diffusion_mse = model.diffusion.training_loss(targets.unsqueeze(1), hidden)

            losses = total_loss(logits, targets, boundaries_batch, diffusion_mse, energy_col=energy_cols)

            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()
            epoch_losses.append(losses["total"].item())

        avg = sum(epoch_losses) / len(epoch_losses)
        if progress_callback:
            progress_callback(epoch, epochs, avg)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", nargs="+", default=["page_0002"])
    parser.add_argument("--max-lines", type=int, default=None, help="only trust the first N lines per page")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument(
        "--status-file", type=Path, default=None,
        help="JSON file to write progress to (used when launched as a subprocess by the backend)",
    )
    args = parser.parse_args()

    from model.training_status import now_iso, write_status

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    def log_progress(epoch, total_epochs, avg_loss):
        if epoch % 10 == 0 or epoch == total_epochs - 1:
            print(f"epoch {epoch:4d}  loss={avg_loss:.4f}")
        write_status(args.status_file, running=True, epoch=epoch + 1, total_epochs=total_epochs, loss=avg_loss)

    try:
        out_path = train_boundary_model(
            pages=args.pages,
            max_lines=args.max_lines,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            resume=args.resume,
            out_path=args.out,
            progress_callback=log_progress,
        )
    except Exception as e:
        write_status(args.status_file, running=False, done=False, error=str(e), finished_at=now_iso())
        raise

    print(f"Saved checkpoint: {out_path}")

    # Copy to the path model_runtime.py serves from, so a fresh checkpoint
    # is picked up on the next /model-guide call with no manual step.
    import shutil

    shutil.copy(out_path, SERVED_CHECKPOINT_PATH)
    print(f"Copied to served checkpoint: {SERVED_CHECKPOINT_PATH}")

    write_status(args.status_file, running=False, done=True, finished_at=now_iso())


if __name__ == "__main__":
    main()
