"""A plain list instead of an R-tree inside docling's LayoutPostprocessor, for PP-DocLayout-V3 pages only.

``LayoutPostprocessor`` builds R-trees (``rtree`` / libspatialindex via ctypes) of the few dozen clusters
of a page, several times per page. Creating, querying and destroying them costs more than comparing
every box: a list scan with the same inclusive intersection test (libspatialindex counts touching
boxes as intersecting) returns the same ids.

Only pages that a model with ``fast_paths`` registered get the list scan, and only while docling
post-processes them: ``docling.utils.layout_postprocessor.BoundingBoxSpatialIndex`` becomes a factory
that returns the list index inside such a ``postprocess`` call and docling's own index everywhere else,
so other layout models and ``fast_paths=False`` keep the R-trees. docling before 2.122 has no
``BoundingBoxSpatialIndex`` there; :func:`install` then changes nothing.
"""

from __future__ import annotations

import contextvars
import functools
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from docling.datamodel.base_models import Page
    from docling_core.types.doc import BoundingBox

_Bounds = tuple[float, float, float, float]
_active: contextvars.ContextVar[bool] = contextvars.ContextVar("pp_doc_layout_list_index", default=False)
_lock = threading.Lock()
# id(page) -> page, for the pages to post-process with the list index; holding the page keeps its id
_pages: dict[int, Page] = {}
_MAX_PAGES = 10_000
_installed = False


def _ordered_bounds(bbox: BoundingBox) -> _Bounds:
    """As docling's ``ordered_bounds``: (left, top, right, bottom) whatever the box's corner order."""
    return min(bbox.l, bbox.r), min(bbox.t, bbox.b), max(bbox.l, bbox.r), max(bbox.t, bbox.b)


class ListSpatialIndex:
    """``BoundingBoxSpatialIndex`` as a list: same methods, same inclusive intersection test."""

    def __init__(self) -> None:
        self._items: list[tuple[int, _Bounds]] = []

    def insert(self, item_id: int, bbox: BoundingBox) -> None:
        """Add a box under ``item_id``."""
        self._items.append((item_id, _ordered_bounds(bbox)))

    def delete(self, item_id: int, bbox: BoundingBox) -> None:
        """Remove the entry of ``item_id`` with exactly these bounds."""
        bounds = _ordered_bounds(bbox)
        for i, item in enumerate(self._items):
            if item[0] == item_id and item[1] == bounds:
                del self._items[i]
                return

    def intersection(self, bbox: BoundingBox) -> Iterable[int]:
        """Ids of the boxes that intersect or touch ``bbox``."""
        left, top, right, bottom = _ordered_bounds(bbox)
        return [i for i, (il, it, ir, ib) in self._items if il <= right and ir >= left and it <= bottom and ib >= top]


def register(page: Page) -> None:
    """Post-process ``page`` with the list index."""
    with _lock:
        if len(_pages) >= _MAX_PAGES:  # pages never post-processed (failed conversions)
            _pages.pop(next(iter(_pages)))
        _pages[id(page)] = page


def _take(page: object) -> bool:
    with _lock:
        if _pages.get(id(page)) is not page:
            return False
        del _pages[id(page)]
        return True


def _factory(original: type) -> Callable[[], Any]:
    def spatial_index() -> Any:  # noqa: ANN401 - docling's index or ours
        return ListSpatialIndex() if _active.get() else original()

    spatial_index.__dcc_pp_list_index__ = True  # ty: ignore[unresolved-attribute]
    return spatial_index


def _wrap(original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    def postprocess(self: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401 - wraps docling's method as is
        if not _take(getattr(self, "page", None)):
            return original(self, *args, **kwargs)
        token = _active.set(True)
        try:
            return original(self, *args, **kwargs)
        finally:
            _active.reset(token)

    return postprocess


def install() -> bool:
    """Route registered pages to the list index; False (and nothing changed) if docling's internals do not fit."""
    global _installed  # noqa: PLW0603
    with _lock:
        if _installed:
            return True
        try:
            from docling.utils import layout_postprocessor  # noqa: PLC0415

            original_index = layout_postprocessor.BoundingBoxSpatialIndex
            original_postprocess = layout_postprocessor.LayoutPostprocessor.postprocess
        except (ImportError, AttributeError):
            return False
        layout_postprocessor.BoundingBoxSpatialIndex = _factory(original_index)  # ty: ignore[invalid-assignment]
        layout_postprocessor.LayoutPostprocessor.postprocess = _wrap(original_postprocess)  # ty: ignore[invalid-assignment]
        _installed = True
        return True
