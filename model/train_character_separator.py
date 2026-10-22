"""Training for CharacterSeparatorModel (flow-matching character/akshara
separator seam generation). Mirrors model/train_line_separator.py.

Usage:
    python -m model.train_character_separator --pages page_0002 --epochs 400
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.character_seam_dataset import CharacterSeamDataset
from model.character_separator_model import CharacterSeparatorModel
from model.config import BackboneConfig, ModelConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "model" / "checkpoints"
DEFAULT_OUT_PATH = CHECKPOINT_DIR / "character_separator_pageonly.pt"


def collate(batch):
    """Each example's line has a different HEIGHT -- pad to the batch max
    (width is already fixed at 2*margin_px, same for every example)."""
    max_height = max(s.shape[-2] for s, _, _ in batch)
    strips, x1s, x0s = [], [], []
    for strip, x1, x0 in batch:
        pad = max_height - strip.shape[-2]
        if pad > 0:
            strip = F.pad(strip, (0, 0, 0, pad))
            x1 = F.pad(x1, (0, pad), value=0.5)
            x0 = F.pad(x0, (0, pad), value=0.5)
        strips.append(strip)
        x1s.append(x1)
        x0s.append(x0)
    return torch.stack(strips), torch.stack(x1s), torch.stack(x0s)


def train_character_separator_model(
    pages: list[str],
    margin_px: int = 15,
    epochs: int = 400,
    batch_size: int = 16,
    lr: float = 1e-3,
    resume: Path | None = None,
    out_path: Path = DEFAULT_OUT_PATH,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> Path:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = CharacterSeamDataset(pages, margin_px=margin_px)
    if len(dataset) == 0:
        raise RuntimeError(f"No character-boundary examples found for pages {pages}.")

    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True, collate_fn=collate)

    cfg = ModelConfig(backbone=BackboneConfig(in_channels=2))
    model = CharacterSeparatorModel(cfg).to(device)
    if resume and resume.exists():
        checkpoint = torch.load(resume, map_location=device)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        epoch_losses = []
        for strips, x1s, x0s in loader:
            strips, x1s, x0s = strips.to(device), x1s.to(device), x0s.to(device)
            loss = model.training_loss(strips, x1s, x0s)
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
    parser.add_argument("--margin-px", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=400)
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
        out_path = train_character_separator_model(
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
