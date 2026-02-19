"""Tests for the PP-DocLayout-V3 → docling label mapping.

The critical invariant is that every ``DocItemLabel`` value produced by the
mapping must be a key in
``LayoutPostprocessor.CONFIDENCE_THRESHOLDS``, otherwise the
postprocessor raises ``KeyError`` at runtime.
"""

from __future__ import annotations

from docling.utils.layout_postprocessor import LayoutPostprocessor
from docling_core.types.doc import DocItemLabel

from docling_pp_doc_layout.label_mapping import LABEL_MAP

SUPPORTED_LABELS: set[DocItemLabel] = set(LayoutPostprocessor.CONFIDENCE_THRESHOLDS.keys())

PP_DOCLAYOUT_V3_RAW_LABELS = [
    "abstract",
    "algorithm",
    "aside_text",
    "chart",
    "content",
    "doc_title",
    "figure_title",
    "footer",
    "footnote",
    "formula",
    "formula_number",
    "header",
    "image",
    "number",
    "paragraph_title",
    "reference",
    "reference_content",
    "seal",
    "table",
    "text",
    "vision_footnote",
]


class TestCoverage:
    """All 21 raw PP-DocLayout-V3 labels must be present in the mapping."""

    def test_all_raw_labels_covered(self):
        for raw in PP_DOCLAYOUT_V3_RAW_LABELS:
            assert raw in LABEL_MAP, f"Raw label '{raw}' is missing from LABEL_MAP"

    def test_no_extra_entries(self):
        extras = set(LABEL_MAP.keys()) - set(PP_DOCLAYOUT_V3_RAW_LABELS)
        assert not extras, f"Unexpected entries in LABEL_MAP: {extras}"


class TestValidity:
    """Every mapped value must be a valid ``DocItemLabel``."""

    def test_all_values_are_doc_item_labels(self):
        for raw, label in LABEL_MAP.items():
            assert isinstance(label, DocItemLabel), f"LABEL_MAP['{raw}'] = {label!r} is not a DocItemLabel"


class TestPostprocessorCompatibility:
    """Every mapped label must be in LayoutPostprocessor.CONFIDENCE_THRESHOLDS.

    This is the root cause of the ``KeyError: <DocItemLabel.CHART: 'chart'>``
    crash -- labels not in CONFIDENCE_THRESHOLDS blow up when the
    postprocessor filters by confidence.
    """

    def test_all_mapped_labels_in_confidence_thresholds(self):
        for raw, label in LABEL_MAP.items():
            assert label in SUPPORTED_LABELS, (
                f"LABEL_MAP['{raw}'] = {label!r} is NOT in "
                "LayoutPostprocessor.CONFIDENCE_THRESHOLDS -- this will cause "
                "a KeyError at runtime"
            )


class TestSpecificMappings:
    """Verify semantically important mappings."""

    def test_chart_maps_to_picture(self):
        assert LABEL_MAP["chart"] == DocItemLabel.PICTURE

    def test_table_maps_to_table(self):
        assert LABEL_MAP["table"] == DocItemLabel.TABLE

    def test_image_maps_to_picture(self):
        assert LABEL_MAP["image"] == DocItemLabel.PICTURE

    def test_doc_title_maps_to_title(self):
        assert LABEL_MAP["doc_title"] == DocItemLabel.TITLE

    def test_formula_maps_to_formula(self):
        assert LABEL_MAP["formula"] == DocItemLabel.FORMULA

    def test_text_maps_to_text(self):
        assert LABEL_MAP["text"] == DocItemLabel.TEXT

    def test_paragraph_title_maps_to_section_header(self):
        assert LABEL_MAP["paragraph_title"] == DocItemLabel.SECTION_HEADER

    def test_header_maps_to_page_header(self):
        assert LABEL_MAP["header"] == DocItemLabel.PAGE_HEADER

    def test_footer_maps_to_page_footer(self):
        assert LABEL_MAP["footer"] == DocItemLabel.PAGE_FOOTER

    def test_footnote_maps_to_footnote(self):
        assert LABEL_MAP["footnote"] == DocItemLabel.FOOTNOTE

    def test_algorithm_maps_to_code(self):
        assert LABEL_MAP["algorithm"] == DocItemLabel.CODE

    def test_reference_maps_to_text(self):
        assert LABEL_MAP["reference"] == DocItemLabel.TEXT

    def test_reference_content_maps_to_text(self):
        assert LABEL_MAP["reference_content"] == DocItemLabel.TEXT

    def test_seal_maps_to_picture(self):
        assert LABEL_MAP["seal"] == DocItemLabel.PICTURE

    def test_caption_mapping(self):
        assert LABEL_MAP["figure_title"] == DocItemLabel.CAPTION


class TestFallbackBehaviour:
    """The model code uses ``LABEL_MAP.get(raw, DocItemLabel.TEXT)`` -- verify
    that unknown labels do not silently produce postprocessor-incompatible
    values via the default.
    """

    def test_default_fallback_is_supported(self):
        assert DocItemLabel.TEXT in SUPPORTED_LABELS
