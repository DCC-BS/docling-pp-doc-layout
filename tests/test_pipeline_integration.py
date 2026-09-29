"""Integration test: the plugin inside docling's real standard PDF pipeline.

Needs the PP-DocLayout-V3 weights (downloaded from Hugging Face on first use) and
RapidOCR with onnxruntime, so it only runs on request:

    PP_DOC_LAYOUT_INTEGRATION=1 uv run --with onnxruntime pytest tests/test_pipeline_integration.py
"""

from __future__ import annotations

import importlib.util
import os
from typing import TYPE_CHECKING

import pytest
from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.skipif(
    os.environ.get("PP_DOC_LAYOUT_INTEGRATION") != "1" or importlib.util.find_spec("onnxruntime") is None,
    reason="set PP_DOC_LAYOUT_INTEGRATION=1 and install onnxruntime to run the pipeline integration test",
)

LETTER = [
    "Basel, 14 February 2025",
    "Subject: Application for an extended permit",
    "Dear Sir or Madam",
    "I hereby apply to extend the permit for the market stall",
    "on the square for another twelve months.",
    "Kind regards",
    "Peter Example",
]


def _scanned_pdf(path: Path, pages: int = 1) -> Path:
    """Write an image-only PDF (no text layer), like a scanner produces."""
    font = ImageFont.load_default(size=36)
    images = []
    for _ in range(pages):
        image = Image.new("RGB", (1654, 2339), "white")  # A4 at 200 dpi
        draw = ImageDraw.Draw(image)
        y = 260
        for line in LETTER:
            draw.text((200, y), line, fill="black", font=font)
            y += 90
        images.append(image)
    images[0].save(path, save_all=True, append_images=images[1:], resolution=200)
    return path


def _convert(pdf: Path) -> str:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    from docling_pp_doc_layout.options import PPDocLayoutV3Options

    options = PdfPipelineOptions(
        allow_external_plugins=True,
        do_ocr=True,
        ocr_options=RapidOcrOptions(),
        layout_options=PPDocLayoutV3Options(),
    )
    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
    return converter.convert(pdf).document.export_to_markdown()


@pytest.mark.parametrize("pages", [1, 2])
def test_scanned_pages_are_ocred(tmp_path: Path, pages: int) -> None:
    """Regression: with layout-driven OCR (docling >= 2.116) a scanned page came out empty,
    because the plugin post-processed its regions before OCR and dropped all of them."""
    markdown = _convert(_scanned_pdf(tmp_path / "scan.pdf", pages))

    for word in ("Basel", "permit", "market", "twelve", "Example"):
        assert markdown.count(word) >= pages, f"{word!r} missing from:\n{markdown}"
