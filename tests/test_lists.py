"""Tests for list detection (lists.py)."""

from __future__ import annotations

import numpy as np
import pytest
from docling.datamodel.base_models import BoundingBox, Cluster
from docling_core.types.doc import CoordOrigin, DocItemLabel
from docling_core.types.doc.page import BoundingRectangle, TextCell
from PIL import Image, ImageDraw

from docling_pp_doc_layout import lists
from docling_pp_doc_layout.lists import (
    BULLET,
    EnumSequences,
    adopt_heron,
    enum_keys,
    ink_marks,
    relabel,
    text_lines,
    to_top_left,
)

LINE_H = 12.0


def cell(
    text: str, left: float, t: float, r: float | None = None, *, bottom_left_height: float | None = None
) -> TextCell:
    r = left + 6 * len(text) if r is None else r
    bb = BoundingBox(l=left, t=t, r=r, b=t + LINE_H, coord_origin=CoordOrigin.TOPLEFT)
    if bottom_left_height is not None:
        bb = bb.to_bottom_left_origin(bottom_left_height)
    return TextCell(text=text, orig=text, rect=BoundingRectangle.from_bounding_box(bb), from_ocr=False)


def text_cluster(cid: int, cells: list[TextCell], label: DocItemLabel = DocItemLabel.TEXT) -> Cluster:
    boxes = [c.rect.to_bounding_box() for c in cells]
    bbox = BoundingBox(l=min(b.l for b in boxes), t=min(b.t for b in boxes), r=max(b.r for b in boxes),
                       b=max(b.b for b in boxes))  # fmt: skip
    return Cluster(id=cid, label=label, bbox=bbox, confidence=0.9, cells=cells)


def para(cid: int, texts: list[str], top: float = 100, left: float = 50) -> Cluster:
    return text_cluster(cid, [cell(t, left, top + i * (LINE_H + 2)) for i, t in enumerate(texts)])


@pytest.fixture(autouse=True)
def fresh_sequences(monkeypatch):
    monkeypatch.setattr(lists, "SEQUENCES", EnumSequences())


def run(clusters, doc="doc", **kw):
    return relabel(clusters, cells_of=lambda c: c.cells, doc_key=doc, **kw)


def labels(clusters):
    return [(c.label, " ".join(x.text for x in c.cells)) for c in clusters]


class TestMarkers:
    @pytest.mark.parametrize(
        "text",
        [
            "• Punkt",
            "·Punkt",
            "\u2013 Angaben",
            "- Kinder",
            "■Sitzung",
            "✓ Budget",
            "\uf0b7 Word-Aufzählung",
            "► Punkt",
        ],
    )
    def test_bullets(self, text):
        assert BULLET.match(text)

    @pytest.mark.parametrize("text", ["- 1'200", "►M3 Gemische ◄", "▼M5 xyz", "Text", "2025 war ein Jahr"])
    def test_not_bullets(self, text):
        assert not BULLET.match(text)

    @pytest.mark.parametrize(
        ("text", "key"),
        [
            ("1. Bestandesaufnahme", (("digit", False, "."), 1)),
            ("12) Antrag", (("digit", False, ")"), 12)),
            ("a)Aggregatzustand", (("lower", False, ")"), 1)),
            ("(ii) die Händler", (("roman-lower", True, ")"), 2)),
            ("B. Zweitens", (("upper", False, "."), 2)),
        ],
    )
    def test_enumerators(self, text, key):
        assert key in enum_keys(text)

    def test_i_is_letter_and_roman(self):
        assert {k[0][0] for k in enum_keys("i) erstens")} == {"lower", "roman-lower"}

    @pytest.mark.parametrize("text", ["(1) Absatz", "1.5 Mio", "z.B. etwas", "3 Varianten", "Artikel 5"])
    def test_not_enumerators(self, text):
        assert enum_keys(text) == []


class TestSequences:
    def test_needs_a_neighbour(self):
        seq = EnumSequences()
        key = enum_keys("12. März 2025")
        assert not seq.accept("d", set(key), key)
        assert seq.accept("d", set(key) | set(enum_keys("13. Mai")), key)

    def test_neighbour_on_earlier_page(self):
        seq = EnumSequences()
        seq.add("d", set(enum_keys("4. Umsetzung")))
        assert seq.accept("d", set(), enum_keys("5. Abschluss"))
        assert not seq.accept("other", set(), enum_keys("5. Abschluss"))

    def test_forgets_old_documents(self):
        seq = EnumSequences(max_documents=2)
        for doc in ("a", "b", "c"):
            seq.add(doc, set(enum_keys("1. x")))
        assert not seq.accept("a", set(), enum_keys("2. y"))
        assert seq.accept("c", set(), enum_keys("2. y"))


class TestLines:
    def test_groups_cells_on_a_line(self):
        lines = text_lines([cell("Punkt", 62, 100), cell("•", 50, 101, 55), cell("zwei", 50, 116)])
        assert [ln.text for ln in lines] == ["• Punkt", "zwei"]

    def test_to_top_left(self):
        c = cell("x", 10, 20, bottom_left_height=800)
        assert c.rect.coord_origin == CoordOrigin.BOTTOMLEFT
        bb = to_top_left([c], 800)[0].rect.to_bounding_box()
        assert (bb.t, bb.b) == pytest.approx((20, 20 + LINE_H))


class TestRelabel:
    def test_bullet_items_split_from_one_region(self):
        out = run([para(1, ["Die Punkte:", "• erstens", "• zweitens lang", "  und weiter", "• drittens"])])
        assert [c.label for c in out] == [DocItemLabel.TEXT] + [DocItemLabel.LIST_ITEM] * 3
        assert [len(c.cells) for c in out] == [1, 1, 2, 1]

    def test_single_item_region_is_relabelled(self):
        out = run([para(1, ["• nur ein Punkt"])])
        assert out[0].label == DocItemLabel.LIST_ITEM
        assert out[0].id == 1

    def test_numbered_items_need_a_sequence(self):
        out = run([para(1, ["1. Bestandesaufnahme"]), para(2, ["2. Befragung"], top=130),
                   para(3, ["12. März 2025: Der Grosse Rat"], top=160)])  # fmt: skip
        assert [c.label for c in out] == [DocItemLabel.LIST_ITEM, DocItemLabel.LIST_ITEM, DocItemLabel.TEXT]

    def test_sequence_continues_on_next_page(self):
        run([para(1, ["1. eins"]), para(2, ["2. zwei"], top=130)], doc="d")
        out = run([para(1, ["3. drei"])], doc="d")
        assert out[0].label == DocItemLabel.LIST_ITEM

    def test_traps_stay_text(self):
        out = run([para(1, ["2025 war ein Jahr"]), para(2, ["(1) Absatz eins"], top=130),
                   para(3, ["(2) Absatz zwei"], top=160), para(4, ["1. Begehren ........ 3"], top=190),
                   para(5, ["2. Begründung ....... 4"], top=220)])  # fmt: skip
        assert {c.label for c in out} == {DocItemLabel.TEXT}

    def test_indented_marker_line_does_not_split(self):
        cl = text_cluster(1, [cell("Ein langer Satz, der hier weitergeht", 50, 100),
                              cell("• kein Punkt", 200, 114)])  # fmt: skip
        assert [c.label for c in run([cl])] == [DocItemLabel.TEXT]

    def test_other_labels_untouched(self):
        header = para(1, ["• Kopfzeile"])
        header.label = DocItemLabel.PAGE_HEADER
        assert run([header])[0].label == DocItemLabel.PAGE_HEADER

    def test_bullet_table_becomes_list(self):
        table = text_cluster(1, [cell(f"• Name {i}", 50, 100 + i * 14) for i in range(4)], DocItemLabel.TABLE)
        out = run([table])
        assert [c.label for c in out] == [DocItemLabel.LIST_ITEM] * 4
        assert all(c.id != 1 for c in out)

    def test_real_table_stays(self):
        table = text_cluster(1, [cell("Position", 50, 100), cell("2024", 200, 100), cell("• x", 50, 114)],
                             DocItemLabel.TABLE)  # fmt: skip
        assert run([table])[0].label == DocItemLabel.TABLE

    def test_veto_unsequenced_list_item(self):
        item = para(1, ["12. März 2025: Der Grosse Rat"])
        item.label = DocItemLabel.LIST_ITEM
        assert run([item], veto=True)[0].label == DocItemLabel.TEXT
        item.label = DocItemLabel.LIST_ITEM
        assert run([item], veto=False)[0].label == DocItemLabel.LIST_ITEM

    def test_veto_keeps_items_without_enumerator(self):
        item = para(1, ["Finanzkommission"])
        item.label = DocItemLabel.LIST_ITEM
        assert run([item], veto=True)[0].label == DocItemLabel.LIST_ITEM


def page_with_marks(marks_at: list[tuple[float, float]], size=(600, 400), scale=2.0) -> Image.Image:
    img = Image.new("L", (int(size[0] * scale), int(size[1] * scale)), 255)
    draw = ImageDraw.Draw(img)
    for x, y in marks_at:
        draw.ellipse([x * scale, y * scale, (x + 3) * scale, (y + 3) * scale], fill=0)
    return img


class TestInk:
    def test_bullet_dropped_by_ocr_is_found(self):
        # Two lines whose bullet OCR did not read: a dot 10pt left of the text, mid-line
        lines = text_lines([cell("Budgetberatung", 60, 100), cell("Schuldenberatung", 60, 120)])
        img = page_with_marks([(47, 104.5), (47, 124.5)])
        marks = ink_marks(img, 2.0, lines, [])
        assert all(m is not None for m in marks)

    def test_no_mark_on_blank_margin(self):
        lines = text_lines([cell("Text", 60, 100)])
        assert ink_marks(page_with_marks([]), 2.0, lines, []) == [None]

    def test_image_edge_is_not_a_mark(self):
        img = Image.new("L", (1200, 800), 255)
        ImageDraw.Draw(img).rectangle([0, 150, 110, 300], fill=40)  # a photo reaching into the line band
        lines = text_lines([cell("Text neben Foto", 60, 100)])
        assert ink_marks(img, 2.0, lines, []) == [None]

    def test_text_left_of_line_is_not_a_mark(self):
        lines = text_lines([cell("Wert", 60, 100)])
        img = page_with_marks([(47, 104.5)])
        assert ink_marks(img, 2.0, lines, [cell("x", 44, 100, 52)]) == [None]

    def test_relabel_needs_two_aligned_marks(self):
        clusters = [para(1, ["Budgetberatung"], left=60), para(2, ["Schuldenberatung"], top=120, left=60)]
        img = page_with_marks([(47, 104.5), (47, 124.5)])
        out = run(clusters, ink=lambda: (img, 2.0, []))
        assert [c.label for c in out] == [DocItemLabel.LIST_ITEM] * 2

        lone = run([para(1, ["Einzeln"], left=60)], ink=lambda: (page_with_marks([(47, 104.5)]), 2.0, []))
        assert lone[0].label == DocItemLabel.TEXT

    def test_no_image(self):
        out = run([para(1, ["Budgetberatung"], left=60)], ink=lambda: None)
        assert out[0].label == DocItemLabel.TEXT


class TestAdoptHeron:
    @staticmethod
    def box(cid, label, ltrb, conf=0.9):
        left, t, r, b = ltrb
        return Cluster(id=cid, label=label, bbox=BoundingBox(l=left, t=t, r=r, b=b), confidence=conf)

    def test_region_with_several_items_is_split(self):
        pp = [self.box(1, DocItemLabel.TEXT, (50, 100, 400, 160))]
        heron = [
            self.box(1, DocItemLabel.LIST_ITEM, (50, 100, 400, 118)),
            self.box(2, DocItemLabel.LIST_ITEM, (50, 120, 400, 138)),
        ]
        out = adopt_heron(pp, heron)
        assert [c.label for c in out] == [DocItemLabel.LIST_ITEM] * 2
        assert out[0].bbox.t == 100

    def test_region_with_one_item_is_relabelled(self):
        out = adopt_heron([self.box(1, DocItemLabel.TEXT, (50, 100, 400, 118))],
                          [self.box(1, DocItemLabel.LIST_ITEM, (52, 101, 398, 117))])  # fmt: skip
        assert out[0].label == DocItemLabel.LIST_ITEM

    def test_low_confidence_and_non_text_ignored(self):
        pp = [self.box(1, DocItemLabel.TEXT, (50, 100, 400, 118)), self.box(2, DocItemLabel.TABLE, (50, 200, 400, 300))]
        heron = [self.box(1, DocItemLabel.LIST_ITEM, (50, 100, 400, 118), conf=0.3),
                 self.box(2, DocItemLabel.LIST_ITEM, (50, 200, 400, 300))]  # fmt: skip
        assert [c.label for c in adopt_heron(pp, heron)] == [DocItemLabel.TEXT, DocItemLabel.TABLE]

    def test_no_heron_items(self):
        pp = [self.box(1, DocItemLabel.TEXT, (50, 100, 400, 118))]
        assert adopt_heron(pp, []) is pp


def test_ink_marks_numpy_types():
    """ink_marks returns plain floats (or None), usable in pydantic boxes."""
    lines = text_lines([cell("A", 60, 100), cell("B", 60, 120)])
    marks = ink_marks(page_with_marks([(47, 104.5), (47, 124.5)]), 2.0, lines, [])
    assert all(isinstance(float(m), float) for m in marks if m is not None)
    assert np.isfinite([m for m in marks if m is not None]).all()
