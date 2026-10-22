"""Render a single PDF page to a PNG at a given DPI, using pypdfium2.

pypdfium2 uses 0-based page indices; the CLI/API here take a 1-based
"human" page number and convert internally. The rendered PNG's actual
dimensions and rotation are always read back from the render result and
written to a sidecar JSON, rather than assumed from DPI math alone --
`/Rotate` handling is version-dependent in pypdfium2, so we verify rather
than hardcode.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import pypdfium2 as pdfium


@dataclass
class PageRenderMeta:
    source_pdf: str
    pdf_page_index_1based: int
    dpi: int
    render_scale: float
    width_px: int
    height_px: int
    rotation_deg: int


def render_page(
    pdf_path: Path,
    page_number_1based: int,
    dpi: int,
    out_dir: Path,
) -> tuple[Path, PageRenderMeta]:
    """Render `page_number_1based` of `pdf_path` to a PNG at `dpi` in `out_dir`.

    Returns (png_path, meta). Raises ValueError if the rendered image's
    orientation doesn't look sane (defensive check against silent
    mis-rotation), and if the requested page is out of range.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf = pdfium.PdfDocument(str(pdf_path))
    n_pages = len(pdf)
    if not (1 <= page_number_1based <= n_pages):
        raise ValueError(
            f"Page {page_number_1based} out of range: PDF has {n_pages} pages (1-based)."
        )

    page_index_0based = page_number_1based - 1
    page = pdf[page_index_0based]

    rotation_deg = page.get_rotation()  # pdfium normalizes to {0, 90, 180, 270}
    scale = dpi / 72.0

    # pypdfium2's PdfPage.render() applies the page's own /Rotate automatically;
    # we do NOT pass an extra rotation= here to avoid double-applying it. We
    # instead verify the *result* matches what /Rotate implies.
    bitmap = page.render(scale=scale)
    pil_image = bitmap.to_pil().convert("L")  # grayscale: manuscript scan, no color info needed

    width_px, height_px = pil_image.size

    # Sanity check: pypdfium2's page.get_size() already returns the rotated
    # (as-displayed) page size in points -- it is NOT the raw /MediaBox --
    # so it should match render() output directly with no extra swap for
    # rotation_deg. (Verified empirically: this version's get_size() tracks
    # /Rotate, unlike some other PDF libraries' raw MediaBox getters.)
    mb_w_pt, mb_h_pt = page.get_size()
    expected_w = round(mb_w_pt * scale)
    expected_h = round(mb_h_pt * scale)

    # Allow +/-1px slack for rounding.
    if abs(width_px - expected_w) > 1 or abs(height_px - expected_h) > 1:
        raise ValueError(
            f"Rendered size {width_px}x{height_px} does not match expected "
            f"{expected_w}x{expected_h} given rotation={rotation_deg}. "
            "pypdfium2's rotation handling may differ from what this code assumes -- "
            "inspect page.render() behavior before trusting downstream CV tuning."
        )

    page_id = f"page_{page_number_1based:04d}"
    png_path = out_dir / f"{page_id}.png"
    pil_image.save(png_path)

    meta = PageRenderMeta(
        source_pdf=str(pdf_path).replace("\\", "/"),
        pdf_page_index_1based=page_number_1based,
        dpi=dpi,
        render_scale=scale,
        width_px=width_px,
        height_px=height_px,
        rotation_deg=rotation_deg,
    )
    meta_path = out_dir / f"{page_id}.meta.json"
    meta_path.write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")

    pdf.close()
    return png_path, meta
