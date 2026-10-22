"""Training for LineSeparatorModel (flow-matching line-separator seam
generation). Mirrors model/train.py's structure/CLI conventions.

Usage:
    python -m model.train_line_separator --pages page_0002 --epochs 500
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader

from model.config import LineBackboneConfig, LineModelConfig
from model.line_separator_model import LineSeparatorModel
from model.seam_path_dataset import SeamPathDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "model" / "checkpoints"
DEFAULT_OUT_PATH = CHECKPOINT_DIR / "line_separator_pageonly.pt"


def collate(batch):
    """All strips share the same height (2*margin_px) but width can differ
    -- either because pages differ in width, or because individual lines on
    the same page have been trimmed to different extents via Scissor (see
    seam_path_dataset.py) -- pad to batch max width."""
    max_width = max(s.shape[-1] for s, _, _ in batch)
    strips, y1s, y0s = [], [], []
    for strip, y1, y0 in batch:
        pad = max_width - strip.shape[-1]
        if pad > 0:
            strip = torch.nn.functional.pad(strip, (0, pad))
            y1 = torch.nn.functional.pad(y1, (0, pad), value=0.5)
            y0 = torch.nn.functional.pad(y0, (0, pad), value=0.5)
        strips.append(strip)
        y1s.append(y1)
        y0s.append(y0)
    return torch.stack(strips), torch.stack(y1s), torch.stack(y0s)


def train_line_separator_model(
    pages: list[str],
    margin_px: int = 40,
    epochs: int = 500,
    batch_size: int = 16,
    lr: float = 1e-3,
    resume: Path | None = None,
    out_path: Path = DEFAULT_OUT_PATH,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> Path:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = SeamPathDataset(pages, margin_px=margin_px)
    if len(dataset) == 0:
        raise RuntimeError(f"No seam-edge examples found for pages {pages}.")

    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True, collate_fn=collate)

    cfg = LineModelConfig(line_backbone=LineBackboneConfig(in_channels=2))  # grayscale + SDT energy
    model = LineSeparatorModel(cfg).to(device)
    if resume and resume.exists():
        checkpoint = torch.load(resume, map_location=device)
        # Saved checkpoints are {"state_dict":..., "margin_px":...}, not a
        # bare state_dict -- but tolerate a bare state_dict too in case of
        # an older/different checkpoint format.
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        epoch_losses = []
        for strips, y1s, y0s in loader:
            strips, y1s, y0s = strips.to(device), y1s.to(device), y0s.to(device)
            loss = model.training_loss(strips, y1s, y0s)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        avg = sum(epoch_losses) / len(epoch_losses)
        if progress_callback:
            progress_callback(epoch, epochs, avg)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "margin_px": margin_px}, out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", nargs="+", default=["page_0002"])
    parser.add_argument("--margin-px", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--status-file", type=Path, default=None)
    args = parser.parse_args()

    from model.training_status import now_iso, write_status

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    def log_progress(epoch, total_epochs, avg_loss):
        if epoch % 50 == 0 or epoch == total_epochs - 1:
            print(f"epoch {epoch:4d}  loss={avg_loss:.5f}")
        write_status(args.status_file, running=True, epoch=epoch + 1, total_epochs=total_epochs, loss=avg_loss)

    try:
        out_path = train_line_separator_model(
            pages=args.pages,
            margin_px=args.margin_px,
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
