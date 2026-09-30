"""Model-level tests for the label fix and list detection wiring."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
from docling.datamodel.base_models import BoundingBox, Cluster
from docling_core.types.doc import DocItemLabel
from PIL import Image

from docling_pp_doc_layout import model as model_module
from docling_pp_doc_layout.model import PPDocLayoutV3Model, _drop_reference_blocks
from docling_pp_doc_layout.options import PPDocLayoutV3Options

HF_CONFIG_NAMES = [
    "abstract", "algorithm", "aside_text", "chart", "content", "formula", "doc_title", "figure_title", "footer",
    "footer", "footnote", "formula_number", "header", "header", "image", "formula", "number", "paragraph_title",
    "reference", "reference_content", "seal", "table", "text", "text", "vision_footnote",
]  # fmt: skip


def make_model(list_detection: str = "rules") -> PPDocLayoutV3Model:
    with patch.object(PPDocLayoutV3Model, "__init__", lambda self, *a, **kw: None):
        instance = PPDocLayoutV3Model.__new__(PPDocLayoutV3Model)
    instance.options = PPDocLayoutV3Options(list_detection=list_detection)
    instance._device = "cpu"
    instance._image_processor = MagicMock()
    instance._model = MagicMock()
    instance._id2label = dict(enumerate(HF_CONFIG_NAMES))
    instance._heron_model = None
    instance.artifacts_path = None
    instance.accelerator_options = MagicMock()
    return instance


def detect(instance: PPDocLayoutV3Model, labels: list[int], boxes: list[list[float]]) -> list[dict]:
    instance._image_processor.return_value = {"pixel_values": torch.zeros(1, 3, 10, 10)}
    instance._image_processor.post_process_object_detection.return_value = [
        {"scores": torch.full((len(labels),), 0.9), "labels": torch.tensor(labels), "boxes": torch.tensor(boxes)}
    ]
    return instance._run_inference([Image.new("RGB", (600, 800))])[0]


class TestClassIds:
    @pytest.mark.parametrize(
        ("class_id", "label"),
        [
            (13, DocItemLabel.PAGE_HEADER),  # header_image, "header" in the HF config
            (9, DocItemLabel.PAGE_FOOTER),  # footer_image
            (15, DocItemLabel.TEXT),  # inline_formula, "formula" in the HF config
            (5, DocItemLabel.FORMULA),  # display_formula
            (23, DocItemLabel.TEXT),  # vertical_text
            (19, DocItemLabel.LIST_ITEM),  # reference_content
        ],
    )
    def test_mapped_by_class_id(self, class_id, label):
        dets = detect(make_model(), [class_id], [[10.0, 10.0, 100.0, 30.0]])
        assert dets[0]["label"] == label
        assert "raw" not in dets[0]


class TestReferenceBlocks:
    def test_block_with_entries_is_dropped(self):
        dets = detect(make_model(), [18, 19, 19], [[0, 0, 300, 300], [5, 5, 290, 40], [5, 50, 290, 90]])
        assert [d["label"] for d in dets] == [DocItemLabel.LIST_ITEM] * 2

    def test_block_with_one_entry_stays_text(self):
        dets = detect(make_model(), [18, 19], [[0, 0, 300, 300], [5, 5, 290, 40]])
        assert [d["label"] for d in dets] == [DocItemLabel.TEXT, DocItemLabel.LIST_ITEM]

    def test_entries_outside_do_not_count(self):
        dets = _drop_reference_blocks([
            {"raw": "reference", "l": 0, "t": 0, "r": 100, "b": 100},
            {"raw": "reference_content", "l": 200, "t": 0, "r": 300, "b": 50},
            {"raw": "reference_content", "l": 200, "t": 60, "r": 300, "b": 90},
        ])  # fmt: skip
        assert len(dets) == 3


def page_mock(page_no: int = 0) -> MagicMock:
    page = MagicMock()
    page.page_no = page_no
    page._backend.is_valid.return_value = True
    page.size = MagicMock(width=600, height=800)
    page.get_image.return_value = Image.new("RGB", (600, 800))
    return page


def run_predict(instance: PPDocLayoutV3Model, conv_res: MagicMock | None = None):
    instance._run_inference = MagicMock(return_value=[[
        {"label": DocItemLabel.TEXT, "confidence": 0.9, "l": 50, "t": 100, "r": 400, "b": 160},
    ]])  # fmt: skip
    with patch("docling_pp_doc_layout.model.TimeRecorder"):
        return list(instance.predict_layout(conv_res or MagicMock(), [page_mock()]))


class TestRegistration:
    @pytest.mark.parametrize(("mode", "veto"), [("rules", False), ("heron", True)])
    def test_pages_registered_for_the_hook(self, monkeypatch, mode, veto):
        calls = []
        monkeypatch.setattr(model_module.postprocess_hook, "register", lambda p, key, veto: calls.append(veto))
        instance = make_model(mode)
        monkeypatch.setattr(instance, "_heron_clusters", lambda conv_res, page: [])
        run_predict(instance)
        assert calls == [veto]

    def test_off_registers_nothing(self, monkeypatch):
        monkeypatch.setattr(model_module.postprocess_hook, "register", lambda *a, **k: pytest.fail("registered"))
        run_predict(make_model("off"))


class TestHeronAssist:
    def test_heron_list_items_are_adopted(self, monkeypatch):
        monkeypatch.setattr(model_module.postprocess_hook, "register", lambda *a, **k: None)
        instance = make_model("heron")
        heron = [
            Cluster(id=0, label=DocItemLabel.LIST_ITEM, bbox=BoundingBox(l=50, t=100, r=400, b=128), confidence=0.9),
            Cluster(id=1, label=DocItemLabel.LIST_ITEM, bbox=BoundingBox(l=50, t=130, r=400, b=160), confidence=0.9),
        ]
        monkeypatch.setattr(instance, "_heron_clusters", lambda conv_res, page: heron)
        (prediction,) = run_predict(instance)
        assert [c.label for c in prediction.clusters] == [DocItemLabel.LIST_ITEM] * 2

    def test_heron_model_is_created_once(self, monkeypatch):
        instance = make_model("heron")
        created = []

        class FakeHeron:
            def __init__(self, **kwargs):
                created.append(kwargs)

            def predict_layout(self, conv_res, pages):
                return [MagicMock(clusters=[])]

        monkeypatch.setattr(
            "docling.models.stages.layout.layout_object_detection_model.LayoutObjectDetectionModel", FakeHeron
        )
        page = page_mock()
        assert instance._heron_clusters(MagicMock(), page) == []
        assert instance._heron_clusters(MagicMock(), page) == []
        assert len(created) == 1


class TestInitInstallsHook:
    @pytest.mark.parametrize(("mode", "installed"), [("rules", True), ("heron", True), ("off", False)])
    def test_install(self, monkeypatch, mode, installed):
        calls = []
        monkeypatch.setattr(model_module.postprocess_hook, "install", lambda: calls.append(1) or True)
        hf_model = MagicMock()
        hf_model.to.return_value = hf_model
        with (
            patch("docling_pp_doc_layout.model.AutoImageProcessor"),
            patch("docling_pp_doc_layout.model.AutoModelForObjectDetection") as aod,
            patch("docling_pp_doc_layout.model.decide_device", return_value="cpu"),
        ):
            aod.from_pretrained.return_value = hf_model
            PPDocLayoutV3Model(artifacts_path=None, accelerator_options=MagicMock(device="cpu"),
                               options=PPDocLayoutV3Options(list_detection=mode))  # fmt: skip
        assert bool(calls) == installed


class TestOption:
    def test_default_is_rules(self, monkeypatch):
        monkeypatch.delenv("PP_DOC_LAYOUT_LIST_DETECTION", raising=False)
        assert PPDocLayoutV3Options().list_detection == "rules"

    def test_env(self, monkeypatch):
        monkeypatch.setenv("PP_DOC_LAYOUT_LIST_DETECTION", "Heron")
        assert PPDocLayoutV3Options().list_detection == "heron"

    def test_invalid(self):
        with pytest.raises(ValueError, match="list_detection"):
            PPDocLayoutV3Options(list_detection="maybe")
