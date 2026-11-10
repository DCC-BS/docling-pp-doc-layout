"""A plain list instead of an R-tree inside docling's LayoutPostprocessor.

``LayoutPostprocessor`` builds R-trees (``rtree`` / libspatialindex via ctypes) of the few dozen clusters
of a page, several times per page. Creating, querying and destroying them costs more than comparing
every box: a list scan with the same inclusive intersection test (libspatialindex counts touching
boxes as intersecting) returns the same ids. Only ``docling.utils.layout_postprocessor`` is patched;
OCR's use of the index is left alone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from docling.datamodel.spatial import ordered_bounds

if TYPE_CHECKING:
    from collections.abc import Iterable

    from docling_core.types.doc import BoundingBox


class ListSpatialIndex:
    def __init__(self) -> None:
        self._items: list[tuple[int, tuple[float, float, float, float]]] = []

    def insert(self, item_id: int, bbox: BoundingBox) -> None:
        self._items.append((item_id, ordered_bounds(bbox)))

    def delete(self, item_id: int, bbox: BoundingBox) -> None:
        bounds = ordered_bounds(bbox)
        for i, item in enumerate(self._items):
            if item[0] == item_id and item[1] == bounds:
                del self._items[i]
                return

    def intersection(self, bbox: BoundingBox) -> Iterable[int]:
        left, top, right, bottom = ordered_bounds(bbox)
        return [i for i, (il, it, ir, ib) in self._items if il <= right and ir >= left and it <= bottom and ib >= top]


def install() -> None:
    from docling.utils import layout_postprocessor  # noqa: PLC0415

    layout_postprocessor.BoundingBoxSpatialIndex = ListSpatialIndex  # ty: ignore[invalid-assignment]
