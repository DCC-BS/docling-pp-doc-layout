"""List detection for PP-DocLayout-V3, which has no list-item class.

PP-DocLayout-V3 labels list items as plain text, so docling would never build lists from
its layout. After OCR, when the text of every region is known, :func:`relabel` turns text
regions into list items when their lines start with a list marker:

- a bullet glyph (``•``, a dash, ``■``, ``✓``, Symbol/Wingdings bullets from Word, ...)
- an enumerator (``1.``, ``2)``, ``a)``, ``(iv)``) that has a neighbour of the same style
  and value n-1 or n+1 on the page or on an earlier page, so ``12. März`` alone stays text
- a small isolated ink mark left of the line where OCR found no text: a bullet that OCR
  dropped (scans) or that the PDF draws as a shape

A region holding several items is split into one list item per marker line. A table whose
lines all start with a bullet becomes a list. Table-of-contents lines (dot leaders),
paragraph numbers such as ``(1)`` and EU amendment markers such as ``►M3`` are not markers.

With ``list_detection="heron"`` docling's own layout model (Heron) runs next to
PP-DocLayout-V3 and its list-item boxes are adopted inside PP text regions
(:func:`adopt_heron`); :func:`relabel` then vetoes Heron list items whose enumerator has
no neighbour.
"""

from __future__ import annotations

import logging
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np
from docling.datamodel.base_models import BoundingBox, Cluster
from docling_core.types.doc import CoordOrigin, DocItemLabel

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Sequence

    from docling_core.types.doc.page import TextCell
    from PIL import Image

    # Loads the page image, its scale and all text cells of the page (top-left coordinates)
    InkSource = Callable[[], tuple[Image.Image, float, Sequence[TextCell]] | None]

logger = logging.getLogger(__name__)

# A bullet glyph (not a triangle followed by an EU amendment code such as "►M3"), or a dash
# before a non-digit ("- 1'200" is an amount)
BULLET = re.compile(
    r"^\s*(?:[\u2022\u00b7\u25aa\u25ab\u25e6\u25cb\u25cf\u25a0\u25a1\u25ba\u25b8\u2023\u2043\u2713\u2714\u2192\u27a2"
    r"\uf020-\uf0ff](?![A-Z]\d)\s*|[-\u2013\u2014]\s+(?=\D))\S"
)
# An enumerator: "1." needs a space after it ("1.5", "z.B."), "1)" and "(a)" do not
ENUM = re.compile(r"^\s*(?P<open>\()?(?P<v>\d{1,2}|[a-zA-Z]|[ivxIVX]{2,5})(?:(?P<paren>\))\s*|\.\s+)(?=\S)")
# Table-of-contents line: dot leaders and a page number
TOC = re.compile(r"(?:\.{4,}|\u2026{2,}|(?:\. ){3,})\s*\d+\s*$")
_ROMAN = {"i": 1, "v": 5, "x": 10}

MIN_ITEMS = 2  # a list, a split region, a bullet table: two or more items
CONTINUATION_INDENT = 0.25  # share of the region width: a line starting further right continues an item
TABLE_COLUMN_PT = 20  # bullet lines of a one-column table start within this x bucket
SAME_X_PT = 4  # ink marks of one list line up within this distance
HERON_INSIDE = 0.5  # share of a Heron list item that must lie inside the PP region
HERON_COVER = 0.5  # share of the PP region a single Heron list item must cover
MIN_ROOM_PT = 2  # narrower search windows left of a line are skipped
MAX_MARK_INK = 0.5  # share of dark pixels in the search window, above which it is an image, not a mark

EnumKey = tuple[tuple[str, bool, str], int]


@dataclass
class Line:
    """A visual text line of a region, in top-left page coordinates."""

    text: str
    left: float
    top: float
    right: float
    bottom: float
    cells: list[TextCell] = field(default_factory=list)


def _box(lines: Sequence[Line]) -> BoundingBox:
    return BoundingBox(
        l=min(x.left for x in lines), t=min(x.top for x in lines), r=max(x.right for x in lines),
        b=max(x.bottom for x in lines),
    )  # fmt: skip


def _roman(s: str) -> int | None:
    s = s.lower()
    if not s or any(ch not in _ROMAN for ch in s):
        return None
    total = 0
    for a, b in zip(s, [*s[1:], ""], strict=True):
        total += -_ROMAN[a] if b and _ROMAN[a] < _ROMAN[b] else _ROMAN[a]
    return total


def enum_keys(text: str) -> list[EnumKey]:
    """All (style, value) readings of the enumerator a line starts with; ``i)`` is a letter and a roman numeral."""
    m = ENUM.match(text)
    if not m:
        return []
    v, opened = m["v"], bool(m["open"])
    closer = ")" if m["paren"] else "."
    if v.isdigit():
        return [] if opened else [(("digit", opened, closer), int(v))]  # "(1)" numbers paragraphs
    keys: list[EnumKey] = []
    case = "lower" if v.islower() else "upper"
    if len(v) == 1:
        keys.append(((case, opened, closer), ord(v.lower()) - 96))
    if (r := _roman(v)) is not None:
        keys.append((("roman-" + case, opened, closer), r))
    return keys


class EnumSequences:
    """Enumerators seen per document, so ``n.`` is accepted only next to ``n-1`` or ``n+1``."""

    def __init__(self, max_documents: int = 16) -> None:
        self._seen: OrderedDict[Hashable, set[EnumKey]] = OrderedDict()
        self._max = max_documents

    def accept(self, doc_key: Hashable, page_keys: set[EnumKey], keys: list[EnumKey]) -> bool:
        """Whether one reading of ``keys`` has a neighbour on this page or an earlier page of the document."""
        pool = self._seen.get(doc_key, set()) | page_keys
        return any((s, v - 1) in pool or (s, v + 1) in pool for s, v in keys)

    def add(self, doc_key: Hashable, keys: set[EnumKey]) -> None:
        """Remember the enumerators of a page; only the most recent documents are kept."""
        self._seen.setdefault(doc_key, set()).update(keys)
        self._seen.move_to_end(doc_key)
        while len(self._seen) > self._max:
            self._seen.popitem(last=False)


SEQUENCES = EnumSequences()


def to_top_left(cells: Sequence[TextCell], page_height: float) -> list[TextCell]:
    """Cells with top-left coordinates (docling keeps PDF cells bottom-left)."""
    return [
        c if c.rect.coord_origin == CoordOrigin.TOPLEFT
        else c.model_copy(update={"rect": c.rect.to_top_left_origin(page_height)})
        for c in cells
    ]  # fmt: skip


def text_lines(cells: Sequence[TextCell]) -> list[Line]:
    """Group cells (top-left coordinates) into visual lines, top to bottom."""
    boxes = sorted(((c.rect.to_bounding_box(), c) for c in cells), key=lambda x: (x[0].t, x[0].l))
    groups: list[list] = []  # [top, bottom, [(bbox, cell), ...]]
    for bb, c in boxes:
        mid = (bb.t + bb.b) / 2
        for g in groups:
            if g[0] <= mid <= g[1]:
                g[2].append((bb, c))
                g[0], g[1] = min(g[0], bb.t), max(g[1], bb.b)
                break
        else:
            groups.append([bb.t, bb.b, [(bb, c)]])
    lines = []
    for top, bottom, members in groups:
        members.sort(key=lambda x: x[0].l)
        text = " ".join(c.text.strip() for _, c in members if c.text.strip())
        left, right = min(bb.l for bb, _ in members), max(bb.r for bb, _ in members)
        lines.append(Line(text, left, top, right, bottom, [c for _, c in members]))
    lines.sort(key=lambda ln: ln.top)
    return lines


def _ink_mark(gray: np.ndarray, threshold: int, scale: float, line: Line, boxes: Sequence[BoundingBox]) -> float | None:
    h = max(line.bottom - line.top, 4.0)
    x0, x1 = max(0.0, line.left - 2.5 * h), line.left - 0.12 * h
    if x1 - x0 < MIN_ROOM_PT or any(bb.r > x0 and bb.l < x1 and bb.b > line.top and bb.t < line.bottom for bb in boxes):
        return None  # no room, or text there: not a bare mark
    window = gray[int(line.top * scale) : int(line.bottom * scale) + 1, int(x0 * scale) : int(x1 * scale) + 1]
    window = window < threshold
    if window.size == 0 or not window.any():
        return None
    ys, xs = np.nonzero(window)
    width, height = (xs.max() - xs.min() + 1) / scale, (ys.max() - ys.min() + 1) / scale
    cy = line.top + (ys.min() + ys.max()) / (2 * scale)
    isolated = ys.min() > 0 and ys.max() < window.shape[0] - 1 and xs.min() > 0  # not the edge of an image
    if (
        isolated
        and width <= 0.9 * h
        and height <= 0.8 * h
        and line.top + 0.15 * h <= cy <= line.bottom - 0.1 * h
        and window.mean() < MAX_MARK_INK
    ):
        return float(x0 + xs.min() / scale)
    return None


def ink_marks(
    image: Image.Image, scale: float, lines: Sequence[Line], page_cells: Sequence[TextCell]
) -> list[float | None]:
    """Per line: x of a small isolated ink mark just left of the text where no text cell is, else None."""
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    threshold = min(160, int(gray.mean()) - 40)
    boxes = [c.rect.to_bounding_box() for c in page_cells]
    return [_ink_mark(gray, threshold, scale, ln, boxes) for ln in lines]


def _aligned(marks: dict[int, list[float | None]]) -> dict[int, list[float | None]]:
    """Keep a mark only if another mark sits at the same x on the page: a list has two or more items."""
    xs = [x for m in marks.values() for x in m if x is not None]

    def keep(x: float | None) -> float | None:
        return x if x is not None and sum(abs(y - x) < SAME_X_PT for y in xs) >= MIN_ITEMS else None

    return {k: [keep(x) for x in m] for k, m in marks.items()}


@dataclass
class _PageState:
    doc_key: Hashable
    page_keys: set[EnumKey]
    marks: dict[int, list[float | None]]
    next_id: int

    def cluster(self, label: DocItemLabel, lines: Sequence[Line], confidence: float) -> Cluster:
        self.next_id += 1
        return Cluster(id=self.next_id, label=label, bbox=_box(lines), confidence=confidence,
                       cells=[c for ln in lines for c in ln.cells])  # fmt: skip

    def is_enumerator(self, text: str) -> bool:
        keys = enum_keys(text)
        return bool(keys) and SEQUENCES.accept(self.doc_key, self.page_keys, keys)


def _bullet_table(cl: Cluster, lines: list[Line], state: _PageState) -> list[Cluster] | None:
    """A one-column table whose lines all start with a bullet, as list items; None if it is a real table."""
    if (
        len(lines) >= MIN_ITEMS
        and all(BULLET.match(ln.text) for ln in lines)
        and len({round(ln.left / TABLE_COLUMN_PT) for ln in lines}) == 1
    ):
        return [state.cluster(DocItemLabel.LIST_ITEM, [ln], cl.confidence) for ln in lines]
    return None


def _item_starts(cl: Cluster, lines: list[Line], state: _PageState) -> list[int]:
    """Indices of the lines that start a list item."""
    marks = state.marks.get(cl.id)
    starts = []
    for i, ln in enumerate(lines):
        if ln.left - cl.bbox.l >= CONTINUATION_INDENT * max(cl.bbox.width, 1) or TOC.search(ln.text):
            continue  # an indented continuation line, or a table-of-contents entry
        if BULLET.match(ln.text) or (marks is not None and marks[i] is not None) or state.is_enumerator(ln.text):
            starts.append(i)
    return starts


def _text_region(cl: Cluster, lines: list[Line], state: _PageState) -> list[Cluster]:
    starts = _item_starts(cl, lines, state)
    if not starts:
        return [cl]
    if starts == [0]:
        cl.label = DocItemLabel.LIST_ITEM
        return [cl]
    bounds = ([0] if starts[0] else []) + starts + [len(lines)]
    return [
        state.cluster(DocItemLabel.LIST_ITEM if a in starts else DocItemLabel.TEXT, lines[a:b], cl.confidence)
        for a, b in pairwise(bounds)
    ]


def relabel(
    clusters: list[Cluster],
    *,
    cells_of: Callable[[Cluster], list[TextCell]],
    doc_key: Hashable,
    ink: InkSource | None = None,
    veto: bool = False,
) -> list[Cluster]:
    """Turn text regions (and bullet tables) whose lines start with list markers into list items.

    Args:
        clusters: Post-processed clusters of one page, cells assigned.
        cells_of: Cells of a cluster in top-left coordinates.
        doc_key: Identifies the document, for enumerator sequences across pages.
        ink: Loads the page image, its scale and the page's text cells, for ink marks; None disables them.
        veto: Turn list items whose enumerator has no neighbour back into text (Heron list items).
    """
    kinds = (DocItemLabel.TEXT, DocItemLabel.LIST_ITEM, DocItemLabel.TABLE)
    lines_of = {cl.id: text_lines(cells_of(cl)) for cl in clusters if cl.label in kinds}
    state = _PageState(doc_key, {k for lines in lines_of.values() for ln in lines for k in enum_keys(ln.text)}, {},
                       max((c.id for c in clusters), default=0))  # fmt: skip
    if ink is not None and (loaded := ink()) is not None:
        image, scale, page_cells = loaded
        text_ids = {cl.id for cl in clusters if cl.label == DocItemLabel.TEXT}
        state.marks = _aligned({cid: ink_marks(image, scale, lines, page_cells)
                                for cid, lines in lines_of.items() if cid in text_ids})  # fmt: skip

    out: list[Cluster] = []
    for cl in clusters:
        lines = lines_of.get(cl.id)
        if not lines:
            out.append(cl)
        elif cl.label == DocItemLabel.TABLE:
            out.extend(_bullet_table(cl, lines, state) or [cl])
        elif cl.label == DocItemLabel.LIST_ITEM:
            if veto and enum_keys(lines[0].text) and not state.is_enumerator(lines[0].text):
                cl.label = DocItemLabel.TEXT
            out.append(cl)
        else:
            out.extend(_text_region(cl, lines, state))
    SEQUENCES.add(doc_key, state.page_keys)
    return out


def adopt_heron(
    pp_clusters: list[Cluster], heron_clusters: Sequence[Cluster], min_confidence: float = 0.5
) -> list[Cluster]:
    """Take Heron's list-item boxes inside PP text regions: split a region holding several, relabel one holding one."""
    items = [h for h in heron_clusters if h.label == DocItemLabel.LIST_ITEM and h.confidence >= min_confidence]
    if not items:
        return pp_clusters
    next_id = max((c.id for c in pp_clusters), default=0) + 1
    out: list[Cluster] = []
    for cl in pp_clusters:
        if cl.label != DocItemLabel.TEXT:
            out.append(cl)
            continue
        inside = [h for h in items if h.bbox.intersection_area_with(cl.bbox) / max(h.bbox.area(), 1e-6) >= HERON_INSIDE]
        if len(inside) >= MIN_ITEMS:
            # Text of the region outside Heron's boxes ends up in docling's orphan clusters
            for h in sorted(inside, key=lambda h: (h.bbox.t, h.bbox.l)):
                out.append(Cluster(id=next_id, label=DocItemLabel.LIST_ITEM, bbox=h.bbox, confidence=cl.confidence))
                next_id += 1
            continue
        if inside and inside[0].bbox.intersection_area_with(cl.bbox) / max(cl.bbox.area(), 1e-6) >= HERON_COVER:
            cl.label = DocItemLabel.LIST_ITEM
        out.append(cl)
    return out
