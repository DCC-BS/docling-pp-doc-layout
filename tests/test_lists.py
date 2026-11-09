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
    InkPage,
    adopt_heron,
    enum_keys,
    ink_marks,
    page_ink_marks,
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


def reference_ink_marks(image, scale, lines, page_cells):
    """ink_marks as it was before the windows were cropped: the whole page in gray, the threshold always."""
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    threshold = min(160, int(gray.mean()) - 40)
    boxes = [c.rect.to_bounding_box() for c in page_cells]
    out = []
    for line in lines:
        h = max(line.bottom - line.top, 4.0)
        x0, x1 = max(0.0, line.left - 2.5 * h), line.left - 0.12 * h
        if x1 - x0 < lists.MIN_ROOM_PT or any(
            bb.r > x0 and bb.l < x1 and bb.b > line.top and bb.t < line.bottom for bb in boxes
        ):
            out.append(None)
            continue
        window = gray[int(line.top * scale) : int(line.bottom * scale) + 1, int(x0 * scale) : int(x1 * scale) + 1]
        out.append(lists._ink_mark(window, threshold, scale, line, x0))
    return out


class TestLazyInk:
    def test_same_marks_as_the_whole_page_version(self):
        rng = np.random.default_rng(7)
        found = 0
        for _ in range(60):
            background = int(rng.choice([255, 230, 120, 60]))  # dark pages move the threshold
            img = Image.new("RGB", (600, 400), (background,) * 3)
            draw = ImageDraw.Draw(img)
            cells = [cell(f"Zeile {i}", float(rng.uniform(5, 120)), 10 + 16 * i) for i in range(10)]
            for c in cells:  # bullets of varied size, shade and position beside most lines, some touching the edge
                if rng.random() < 0.7:
                    left = c.rect.to_bounding_box().l - float(rng.uniform(1, 25))
                    top = c.rect.to_bounding_box().t + float(rng.uniform(-2, 9))
                    size, shade = float(rng.uniform(1, 9)), int(rng.integers(0, 220))
                    draw.ellipse([left * 2, top * 2, (left + size) * 2, (top + size) * 2], fill=(shade,) * 3)
            for _ in range(int(rng.integers(0, 4))):  # photos and noise
                x, y, w = (float(v) for v in rng.uniform([0, 0, 2], [150, 200, 60]))
                draw.rectangle([x * 2, y * 2, (x + w) * 2, (y + w) * 2], fill=(int(rng.integers(0, 255)),) * 3)
            lines = text_lines(cells)
            page_cells = (
                [cell("x", float(rng.uniform(0, 100)), float(rng.uniform(0, 200)))] if rng.random() < 0.3 else []
            )
            expected = reference_ink_marks(img, 2.0, lines, page_cells)
            assert ink_marks(img, 2.0, lines, page_cells) == expected
            # A preview at half the resolution (each pixel the mean of four) skips no mark either
            lines_of = {i: [ln] for i, ln in enumerate(lines)}
            full = page_ink_marks(InkPage(cells=page_cells, image=lambda img=img: img), lines_of)
            previewed = page_ink_marks(
                InkPage(cells=page_cells, image=lambda img=img: img, preview=lambda img=img: (img.reduce(2), 1.0)),
                lines_of,
            )
            assert previewed == full
            found += sum(m is not None for m in expected)
        assert found > 50  # the comparison covers lines with marks, not only empty windows

    def test_window_beyond_the_image(self):
        img = Image.new("L", (200, 150), 255)  # smaller than the page coordinates: crops are clamped
        ImageDraw.Draw(img).ellipse([94, 145, 100, 150], fill=0)
        lines = text_lines([cell("Unten", 60, 70), cell("Rand", 60, 100)])
        assert ink_marks(img, 2.0, lines, []) == reference_ink_marks(img, 2.0, lines, [])

    @staticmethod
    def two_lines():
        return {1: text_lines([cell("Budgetberatung", 60, 100)]), 2: text_lines([cell("Schuldenberatung", 60, 120)])}

    def test_lone_window_loads_no_image(self):
        def image():
            pytest.fail("no line can have an aligned mark: the image must not load")

        page = InkPage(cells=[], image=image, preview=image)
        assert page_ink_marks(page, {1: text_lines([cell("Einzeln", 60, 100)])}) == {1: [None]}

    def test_windows_too_far_apart_load_no_image(self):
        def image():
            pytest.fail("windows 100pt apart cannot hold aligned marks")

        lines_of = {1: text_lines([cell("Links", 60, 100)]), 2: text_lines([cell("Rechts", 300, 120)])}
        assert page_ink_marks(InkPage(cells=[], image=image), lines_of) == {1: [None], 2: [None]}

    def test_blank_preview_skips_the_image(self):
        def image():
            pytest.fail("the preview shows no ink: the image must not load")

        page = InkPage(cells=[], image=image, preview=lambda: (page_with_marks([], scale=1.0), 1.0))
        assert page_ink_marks(page, self.two_lines()) == {1: [None], 2: [None]}

    def test_preview_with_ink_loads_the_image(self):
        marks = [(47, 104.5), (47, 124.5)]
        loaded = []

        def image():
            loaded.append(True)
            return page_with_marks(marks)

        page = InkPage(cells=[], image=image, preview=lambda: (page_with_marks(marks, scale=1.0), 1.0))
        out = page_ink_marks(page, self.two_lines())
        assert loaded == [True]
        assert out[1][0] is not None
        assert out[2][0] is not None

    def test_threshold_only_when_a_window_has_dark_pixels(self):
        lines = text_lines([cell("A", 60, 100), cell("B", 60, 120)])
        threshold = lists._Threshold(page_with_marks([]))
        assert lists._line_marks(page_with_marks([]), 2.0, lines, [(30.0, 58.0), (30.0, 58.0)], threshold) == [
            None,
            None,
        ]
        assert threshold._value is None

    def test_relabel_with_ink_page(self):
        img = page_with_marks([(47, 104.5), (47, 124.5)])
        clusters = [para(1, ["Budgetberatung"], left=60), para(2, ["Schuldenberatung"], top=120, left=60)]
        out = run(clusters, ink=lambda: InkPage(cells=[], image=lambda: img))
        assert [c.label for c in out] == [DocItemLabel.LIST_ITEM] * 2

    def test_page_without_image(self):
        clusters = [para(1, ["Budgetberatung"], left=60), para(2, ["Schuldenberatung"], top=120, left=60)]
        out = run(clusters, ink=lambda: InkPage(cells=[], image=lambda: None))
        assert [c.label for c in out] == [DocItemLabel.TEXT] * 2
