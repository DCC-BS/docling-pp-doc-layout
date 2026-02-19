"""Tests for the docling plugin entry point."""

from __future__ import annotations

from docling.models.base_layout_model import BaseLayoutModel


def test_layout_engines_returns_dict():
    from docling_pp_doc_layout.plugin import layout_engines

    result = layout_engines()
    assert isinstance(result, dict)
    assert "layout_engines" in result


def test_layout_engines_contains_model_class():
    from docling_pp_doc_layout.plugin import layout_engines

    result = layout_engines()
    classes = result["layout_engines"]
    assert len(classes) == 1
    assert issubclass(classes[0], BaseLayoutModel)


def test_registered_model_name():
    from docling_pp_doc_layout.plugin import layout_engines

    cls = layout_engines()["layout_engines"][0]
    assert cls.__name__ == "PPDocLayoutV3Model"
