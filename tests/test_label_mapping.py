"""Tests for the PP-DocLayout-V3 → docling label mapping.

The critical invariant is that every ``DocItemLabel`` value produced by the
mapping must be a key in
``LayoutPostprocessor.CONFIDENCE_THRESHOLDS``, otherwise the
postprocessor raises ``KeyError`` at runtime.
"""

from __future__ import annotations

from docling.utils.layout_postprocessor import LayoutPostprocessor
from docling_core.types.doc import DocItemLabel

from docling_pp_doc_layout.label_mapping import LABEL_MAP, PP_DOC_LAYOUT_V3_CLASSES, class_names

SUPPORTED_LABELS: set[DocItemLabel] = set(LayoutPostprocessor.CONFIDENCE_THRESHOLDS.keys())

# The names the HuggingFace config.json of PP-DocLayoutV3_safetensors uses, by class id
HF_CONFIG_NAMES = [
    "abstract", "algorithm", "aside_text", "chart", "content", "formula", "doc_title", "figure_title", "footer",
    "footer", "footnote", "formula_number", "header", "header", "image", "formula", "number", "paragraph_title",
    "reference", "reference_content", "seal", "table", "text", "text", "vision_footnote",
]  # fmt: skip


class TestCoverage:
    """Every PP-DocLayout-V3 class, and every name the HuggingFace config uses, is mapped."""

    def test_all_model_classes_covered(self):
        for raw in PP_DOC_LAYOUT_V3_CLASSES:
            assert raw in LABEL_MAP, f"Class '{raw}' is missing from LABEL_MAP"

    def test_all_hf_config_names_covered(self):
        for raw in HF_CONFIG_NAMES:
            assert raw in LABEL_MAP, f"HF config name '{raw}' is missing from LABEL_MAP"

    def test_no_extra_entries(self):
        extras = set(LABEL_MAP) - set(PP_DOC_LAYOUT_V3_CLASSES) - set(HF_CONFIG_NAMES)
        assert not extras, f"Unexpected entries in LABEL_MAP: {extras}"


class TestClassNames:
    """The merged HuggingFace names are restored to the model's 25 classes."""

    def test_hf_config_is_restored(self):
        names = class_names(dict(enumerate(HF_CONFIG_NAMES)))
        assert names == list(PP_DOC_LAYOUT_V3_CLASSES)
        assert names[13] == "header_image"
        assert names[15] == "inline_formula"

    def test_other_models_keep_their_names(self):
        assert class_names({0: "text", 1: "table", 2: "image"}) == ["text", "table", "image"]

    def test_gaps_fall_back_to_text(self):
        assert class_names({0: "table", 2: "image"}) == ["table", "text", "image"]

    def test_empty(self):
        assert class_names({}) == []


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

    def test_reference_content_maps_to_list_item(self):
        assert LABEL_MAP["reference_content"] == DocItemLabel.LIST_ITEM

    def test_inline_formula_maps_to_text(self):
        assert LABEL_MAP["inline_formula"] == DocItemLabel.TEXT

    def test_display_formula_maps_to_formula(self):
        assert LABEL_MAP["display_formula"] == DocItemLabel.FORMULA

    def test_logos_are_page_furniture(self):
        assert LABEL_MAP["header_image"] == DocItemLabel.PAGE_HEADER
        assert LABEL_MAP["footer_image"] == DocItemLabel.PAGE_FOOTER

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
