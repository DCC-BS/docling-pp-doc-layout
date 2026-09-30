"""Tests for the post-OCR list-detection hook (postprocess_hook.py)."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from docling.datamodel.base_models import BoundingBox, Cluster, Page, Size
from docling.datamodel.pipeline_options import LayoutPostprocessorOptions
from docling.utils.layout_postprocessor import LayoutPostprocessor
from docling_core.types.doc import CoordOrigin, DocItemLabel
from docling_core.types.doc.page import BoundingRectangle, TextCell

from docling_pp_doc_layout import postprocess_hook


@pytest.fixture
def hook(monkeypatch):
    """Install the hook on LayoutPostprocessor with fresh state; everything is restored afterwards."""
    monkeypatch.setattr(LayoutPostprocessor, "postprocess", LayoutPostprocessor.postprocess)
    monkeypatch.setattr(postprocess_hook, "_installed", False)
    monkeypatch.setattr(postprocess_hook, "_warned", False)
    monkeypatch.setattr(postprocess_hook, "_pages", {})
    monkeypatch.setattr(postprocess_hook, "_ink_source", lambda page: None)
    assert postprocess_hook.install()
    return postprocess_hook


def make_page(texts: list[str]) -> tuple[Page, list[Cluster]]:
    """A page whose text lines are PDF cells, one TEXT cluster per line."""
    cells, clusters = [], []
    for i, text in enumerate(texts):
        bb = BoundingBox(l=50, t=100 + i * 30, r=300, b=112 + i * 30, coord_origin=CoordOrigin.TOPLEFT)
        rect = BoundingRectangle.from_bounding_box(bb)
        cells.append(TextCell(index=i, text=text, orig=text, rect=rect, from_ocr=False))
        box = BoundingBox(l=49, t=99 + i * 30, r=301, b=113 + i * 30)
        clusters.append(Cluster(id=i, label=DocItemLabel.TEXT, bbox=box, confidence=0.9))
    page = Page(page_no=0)
    page.size = Size(width=600, height=800)
    page.parsed_page = SimpleNamespace(textline_cells=cells, has_lines=True)  # ty: ignore[invalid-assignment]
    return page, clusters


def postprocess(page: Page, clusters: list[Cluster]) -> list[Cluster]:
    return LayoutPostprocessor(page, clusters, LayoutPostprocessorOptions()).postprocess()


class TestHook:
    def test_registered_page_gets_lists(self, hook):
        page, clusters = make_page(["• erstens", "• zweitens"])
        hook.register(page, "doc")
        out = postprocess(page, clusters)
        assert [c.label for c in out] == [DocItemLabel.LIST_ITEM] * 2
        assert hook._pages == {}

    def test_unregistered_page_untouched(self, hook):
        page, clusters = make_page(["• erstens", "• zweitens"])
        assert [c.label for c in postprocess(page, clusters)] == [DocItemLabel.TEXT] * 2

    def test_page_is_processed_once(self, hook):
        page, clusters = make_page(["• erstens"])
        hook.register(page, "doc")
        postprocess(page, clusters)
        _, again = make_page(["• erstens"])
        assert postprocess(page, again)[0].label == DocItemLabel.TEXT

    def test_failure_keeps_docling_clusters(self, hook, monkeypatch, caplog):
        def boom(*args, **kwargs):
            message = "boom"
            raise RuntimeError(message)

        monkeypatch.setattr(postprocess_hook, "relabel", boom)
        page, clusters = make_page(["• erstens"])
        hook.register(page, "doc")
        with caplog.at_level(logging.WARNING):
            out = postprocess(page, clusters)
        assert out[0].label == DocItemLabel.TEXT
        assert "list detection failed" in caplog.text

    def test_install_is_idempotent(self, hook):
        wrapped = LayoutPostprocessor.postprocess
        assert hook.install()
        assert LayoutPostprocessor.postprocess is wrapped
        assert getattr(wrapped, "__dcc_pp_lists__", False)

    def test_registry_is_bounded(self, hook, monkeypatch):
        monkeypatch.setattr(postprocess_hook, "_MAX_PAGES", 3)
        for _ in range(5):
            hook.register(make_page(["x"])[0], "doc")
        assert len(hook._pages) == 3


def test_tuple_result_passes_through(monkeypatch):
    """docling < 2.116 returned (clusters, cells); the wrapper leaves that alone."""
    monkeypatch.setattr(postprocess_hook, "_detect_lists", lambda *a: pytest.fail("must not run"))
    result = ([], [])
    wrapper = postprocess_hook._wrap(lambda self: result)
    assert wrapper(SimpleNamespace(page=None)) is result


def test_install_without_postprocessor(monkeypatch, caplog):
    monkeypatch.setattr(postprocess_hook, "_installed", False)
    monkeypatch.setattr(postprocess_hook, "_warned", False)
    monkeypatch.delattr(LayoutPostprocessor, "postprocess")
    with caplog.at_level(logging.WARNING):
        assert not postprocess_hook.install()
    assert "list detection is off" in caplog.text
