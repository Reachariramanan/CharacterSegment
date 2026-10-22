"""CLI: render one PDF page to a PNG at a given DPI.

Usage:
    python -m scripts.extract_page --page 2 --dpi 180
"""

from __future__ import annotations

import argparse
from pathlib import Path

from backend.services.pdf_extract import render_page

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PDF = PROJECT_ROOT / "data" / "Yaajusha_Manthra_Rathnaakaram.PDF.pdf"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "images"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page", type=int, required=True, help="1-based page number")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    png_path, meta = render_page(args.pdf, args.page, args.dpi, args.out_dir)

    print(f"Wrote: {png_path}")
    print(f"Size:  {meta.width_px}x{meta.height_px} px")
    print(f"DPI:   {meta.dpi} (scale={meta.render_scale})")
    print(f"Rotation applied by source /Rotate: {meta.rotation_deg} deg")


if __name__ == "__main__":
    main()
