"""Training for LineSegmentationModel (page-level line y-band detection).
Mirrors model/train.py's structure; see model/line_dataset.py for why only
a trusted line-prefix (`--max-lines`) is used when the annotation isn't
fully reviewed end to end.

Usage:
    python -m model.train_line_segmenter --pages page_0002 --epochs 300

`train_line_segmentation_model` is the reusable entrypoint (used by both
this CLI and the backend's /retrain endpoint); `main()` is a thin CLI
wrapper around it.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.config import LineModelConfig
from model.line_dataset import LineSegmentationDataset
from model.line_segmenter import LineSegmentationModel
from model.losses import total_loss

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "model" / "checkpoints"
DEFAULT_OUT_PATH = CHECKPOINT_DIR / "line_segmenter_pageonly.pt"


def collate(batch):
    """Pads variable-height page crops to the batch max height."""
    max_h = max(crop.shape[-2] for crop, _, _ in batch)
    crops, targets, edges_batch = [], [], []
    for crop, target, edges in batch:
        pad = max_h - crop.shape[-2]
        if pad > 0:
            crop = F.pad(crop, (0, 0, 0, pad))
            target = F.pad(target, (0, pad))
        crops.append(crop)
        targets.append(target)
        edges_batch.append(edges)
    return torch.stack(crops), torch.stack(targets), edges_batch


def train_line_segmentation_model(
    pages: list[str],
    max_lines: int | None = None,
    epochs: int = 300,
    batch_size: int = 1,
    lr: float = 1e-3,
    resume: Path | None = None,
    out_path: Path = DEFAULT_OUT_PATH,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> Path:
    """Trains LineSegmentationModel and saves the checkpoint to `out_path`,
    returning that path. progress_callback(epoch, total_epochs, avg_loss) is
    called once per epoch if given."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = LineSegmentationDataset(pages, max_lines=max_lines)
    if len(dataset) == 0:
        raise RuntimeError(
            f"No training examples found for pages {pages}. "
            "Annotate and save at least one page via the UI first."
        )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)

    cfg = LineModelConfig()
    model = LineSegmentationModel(cfg).to(device)
    if resume and resume.exists():
        model.load_state_dict(torch.load(resume, map_location=device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        epoch_losses = []
        for crops, targets, edges_batch in loader:
            crops, targets = crops.to(device), targets.to(device)

            logits, hidden = model.encode(crops)
            diffusion_mse = model.diffusion.training_loss(targets.unsqueeze(1), hidden)

            losses = total_loss(logits, targets, edges_batch, diffusion_mse)

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
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1)
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
        if epoch % 20 == 0 or epoch == total_epochs - 1:
            print(f"epoch {epoch:4d}  loss={avg_loss:.4f}")
        write_status(args.status_file, running=True, epoch=epoch + 1, total_epochs=total_epochs, loss=avg_loss)

    try:
        out_path = train_line_segmentation_model(
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
    write_status(args.status_file, running=False, done=True, finished_at=now_iso())


if __name__ == "__main__":
    main()
