"""Run list detection after OCR, inside docling's layout post-processing.

docling has no plugin interface after OCR: the layout model runs before OCR, and docling's
``LayoutPostprocessor`` assigns the (PDF and OCR) text cells to the regions afterwards.
List detection needs that text, so :func:`install` wraps ``LayoutPostprocessor.postprocess``.

The wrapper only touches pages that :class:`PPDocLayoutV3Model` registered (other layout
models, other pipelines in the same process are left alone) and never breaks a conversion:
if docling's internals changed or list detection fails, it logs a warning and returns
docling's clusters unchanged.
"""

from __future__ import annotations

import functools
import logging
import threading
from typing import TYPE_CHECKING, Any

from docling_pp_doc_layout.lists import relabel, to_top_left

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable

    from docling.datamodel.base_models import Cluster, Page
    from docling.utils.layout_postprocessor import LayoutPostprocessor
    from docling_core.types.doc.page import TextCell
    from PIL import Image

logger = logging.getLogger(__name__)

_lock = threading.Lock()
# id(page) -> (page, document key, veto). Holding the page keeps its id from being reused;
# the hook removes the entry when it has processed the page.
_pages: dict[int, tuple[Page, Hashable, bool]] = {}
_MAX_PAGES = 10_000
_installed = False
_warned = False


def register(page: Page, doc_key: Hashable, *, veto: bool = False) -> None:
    """Mark a page laid out by PP-DocLayout-V3 for list detection."""
    with _lock:
        if len(_pages) >= _MAX_PAGES:  # pages never post-processed (failed conversions)
            _pages.pop(next(iter(_pages)))
        _pages[id(page)] = (page, doc_key, veto)


def _take(page: Page) -> tuple[Hashable, bool] | None:
    with _lock:
        entry = _pages.get(id(page))
        if entry is None or entry[0] is not page:
            return None
        del _pages[id(page)]
        return entry[1], entry[2]


def _warn_once(message: str, *args: object, exc_info: bool = False) -> None:
    global _warned  # noqa: PLW0603
    if not _warned:
        _warned = True
        logger.warning(message, *args, exc_info=exc_info)


def _ink_source(page: Page) -> tuple[Image.Image, float, list[TextCell]] | None:
    """The page image at twice the layout scale and the page's text cells, for ink marks."""
    image = page.get_image(scale=2.0)
    if image is None or page.size is None:
        return None
    return image, 2.0, to_top_left(list(page.cells or []), page.size.height)


def _detect_lists(processor: LayoutPostprocessor, clusters: list[Cluster]) -> list[Cluster]:
    page = processor.page
    entry = _take(page)
    if entry is None or page.size is None:
        return clusters
    doc_key, veto = entry
    height = page.size.height
    try:
        return relabel(
            clusters,
            cells_of=lambda cl: to_top_left(cl.cells, height),
            doc_key=doc_key,
            ink=lambda: _ink_source(page),
            veto=veto,
        )
    except Exception:  # noqa: BLE001 - list detection must never fail a conversion
        _warn_once(
            "PP-DocLayout-V3 list detection failed on page %s; lists stay plain text", page.page_no, exc_info=True
        )
        return clusters


def _wrap(original: Callable[..., Any]) -> Callable[..., Any]:
    """``LayoutPostprocessor.postprocess`` followed by list detection on registered pages."""

    @functools.wraps(original)
    def postprocess(self: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401 - wraps docling's method as is
        result = original(self, *args, **kwargs)
        if not isinstance(result, list) or not hasattr(self, "page"):
            return result  # docling < 2.116 returned (clusters, cells) from the layout model itself
        return _detect_lists(self, result)

    postprocess.__dcc_pp_lists__ = True  # ty: ignore[unresolved-attribute]
    return postprocess


def install() -> bool:
    """Wrap ``LayoutPostprocessor.postprocess`` once. Returns False if docling's internals do not fit."""
    global _installed  # noqa: PLW0603
    with _lock:
        if _installed:
            return True
        try:
            from docling.utils.layout_postprocessor import LayoutPostprocessor

            original = LayoutPostprocessor.postprocess
        except (ImportError, AttributeError):
            _warn_once("docling has no LayoutPostprocessor.postprocess; PP-DocLayout-V3 list detection is off")
            return False

        LayoutPostprocessor.postprocess = _wrap(original)  # ty: ignore[invalid-assignment]
        _installed = True
        return True
