"""CLI: render every page of the source PDF to PNG (skips pages already
rendered). Same convention as extract_page.py (page_{n:04d}.png +
sidecar meta.json), just batched.

Usage:
    python -m scripts.extract_all_pages --dpi 180
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pypdfium2 as pdfium

from backend.services.pdf_extract import render_page

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PDF = PROJECT_ROOT / "data" / "Yaajusha_Manthra_Rathnaakaram.PDF.pdf"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "images"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=None, help="inclusive; default = last page")
    args = parser.parse_args()

    doc = pdfium.PdfDocument(str(args.pdf))
    n_pages = len(doc)
    doc.close()
    end = args.end if args.end is not None else n_pages

    n_done, n_skipped, n_failed = 0, 0, 0
    for page_number in range(args.start, end + 1):
        page_id = f"page_{page_number:04d}"
        png_path = args.out_dir / f"{page_id}.png"
        if png_path.exists():
            n_skipped += 1
            continue
        try:
            render_page(args.pdf, page_number, args.dpi, args.out_dir)
            n_done += 1
            print(f"[{page_number}/{end}] rendered {page_id}")
        except Exception as e:
            n_failed += 1
            print(f"[{page_number}/{end}] FAILED {page_id}: {e}")

    print(f"Done. rendered={n_done} skipped(existing)={n_skipped} failed={n_failed}")


if __name__ == "__main__":
    main()
