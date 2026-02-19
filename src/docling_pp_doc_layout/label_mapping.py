"""Mapping from PP-DocLayout-V3 label names to docling DocItemLabel values.

Every label produced here must exist in
``docling.utils.layout_postprocessor.LayoutPostprocessor.CONFIDENCE_THRESHOLDS``
so that the postprocessor can apply confidence filtering without a ``KeyError``.
"""

from __future__ import annotations

from docling_core.types.doc import DocItemLabel

LABEL_MAP: dict[str, DocItemLabel] = {
    "abstract": DocItemLabel.TEXT,
    "algorithm": DocItemLabel.CODE,
    "aside_text": DocItemLabel.TEXT,
    "chart": DocItemLabel.PICTURE,
    "content": DocItemLabel.TEXT,
    "doc_title": DocItemLabel.TITLE,
    "figure_title": DocItemLabel.CAPTION,
    "footer": DocItemLabel.PAGE_FOOTER,
    "footnote": DocItemLabel.FOOTNOTE,
    "formula": DocItemLabel.FORMULA,
    "formula_number": DocItemLabel.TEXT,
    "header": DocItemLabel.PAGE_HEADER,
    "image": DocItemLabel.PICTURE,
    "number": DocItemLabel.TEXT,
    "paragraph_title": DocItemLabel.SECTION_HEADER,
    "reference": DocItemLabel.TEXT,
    "reference_content": DocItemLabel.TEXT,
    "seal": DocItemLabel.PICTURE,
    "table": DocItemLabel.TABLE,
    "text": DocItemLabel.TEXT,
    "vision_footnote": DocItemLabel.FOOTNOTE,
}
