"""Mapping from PP-DocLayout-V3 classes to docling DocItemLabel values.

Every label produced here must exist in
``docling.utils.layout_postprocessor.LayoutPostprocessor.CONFIDENCE_THRESHOLDS``
so that the postprocessor can apply confidence filtering without a ``KeyError``.

The HuggingFace ``config.json`` of ``PaddlePaddle/PP-DocLayoutV3_safetensors`` names only
21 of the model's 25 classes distinctly: it calls ``footer_image`` "footer",
``header_image`` "header", ``display_formula`` and ``inline_formula`` "formula" and
``vertical_text`` "text". :func:`class_names` restores the model's own class list
(``label_list`` in the repository's ``inference.yml``) so each class is mapped on its own.
"""

from __future__ import annotations

from docling_core.types.doc import DocItemLabel

# Class ids 0..24 of PP-DocLayout-V3, from the model's inference.yml
PP_DOC_LAYOUT_V3_CLASSES: tuple[str, ...] = (
    "abstract",
    "algorithm",
    "aside_text",
    "chart",
    "content",
    "display_formula",
    "doc_title",
    "figure_title",
    "footer",
    "footer_image",
    "footnote",
    "formula_number",
    "header",
    "header_image",
    "image",
    "inline_formula",
    "number",
    "paragraph_title",
    "reference",
    "reference_content",
    "seal",
    "table",
    "text",
    "vertical_text",
    "vision_footnote",
)

# Names the HuggingFace config uses instead of the ones above
_HF_MERGED_NAMES = {
    "footer_image": "footer",
    "header_image": "header",
    "display_formula": "formula",
    "inline_formula": "formula",
    "vertical_text": "text",
}

LABEL_MAP: dict[str, DocItemLabel] = {
    "abstract": DocItemLabel.TEXT,
    "algorithm": DocItemLabel.CODE,
    "aside_text": DocItemLabel.TEXT,
    "chart": DocItemLabel.PICTURE,
    "content": DocItemLabel.TEXT,
    "display_formula": DocItemLabel.FORMULA,
    "doc_title": DocItemLabel.TITLE,
    "figure_title": DocItemLabel.CAPTION,
    "footer": DocItemLabel.PAGE_FOOTER,
    # Logos in the page footer/header: furniture, like the text next to them
    "footer_image": DocItemLabel.PAGE_FOOTER,
    "footnote": DocItemLabel.FOOTNOTE,
    # The merged name of the HuggingFace config, for models without the class list above
    "formula": DocItemLabel.FORMULA,
    "formula_number": DocItemLabel.TEXT,
    "header": DocItemLabel.PAGE_HEADER,
    "header_image": DocItemLabel.PAGE_HEADER,
    "image": DocItemLabel.PICTURE,
    # A formula inside a text line belongs to that line, not to a formula block of its own
    "inline_formula": DocItemLabel.TEXT,
    "number": DocItemLabel.TEXT,
    "paragraph_title": DocItemLabel.SECTION_HEADER,
    # The block around a bibliography; dropped when it holds reference entries (model.py)
    "reference": DocItemLabel.TEXT,
    # One bibliography entry, a list item as with docling's own layout model
    "reference_content": DocItemLabel.LIST_ITEM,
    "seal": DocItemLabel.PICTURE,
    "table": DocItemLabel.TABLE,
    "text": DocItemLabel.TEXT,
    "vertical_text": DocItemLabel.TEXT,
    "vision_footnote": DocItemLabel.FOOTNOTE,
}


def class_names(id2label: dict[int, str]) -> list[str]:
    """Class name per class id.

    Returns the model's own 25 class names when ``id2label`` is PP-DocLayout-V3's merged
    HuggingFace naming, and the config's names otherwise (a different or fine-tuned model).
    """
    names = [id2label.get(i, "text") for i in range(max(id2label, default=-1) + 1)]
    if len(names) == len(PP_DOC_LAYOUT_V3_CLASSES) and all(
        got in (want, _HF_MERGED_NAMES.get(want)) for got, want in zip(names, PP_DOC_LAYOUT_V3_CLASSES, strict=True)
    ):
        return list(PP_DOC_LAYOUT_V3_CLASSES)
    return names
