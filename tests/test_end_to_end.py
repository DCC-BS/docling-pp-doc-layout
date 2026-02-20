"""End-to-end integration tests.

These tests exercise the full chain:
  plugin.layout_engines() → PPDocLayoutV3Model instantiation
  → predict_layout() → LayoutPrediction output

All heavy I/O (HuggingFace model loading, LayoutPostprocessor, TimeRecorder)
is mocked so the tests run without GPU or real model weights.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
from docling.datamodel.base_models import LayoutPrediction
from docling.models.base_layout_model import BaseLayoutModel
from docling.utils.layout_postprocessor import LayoutPostprocessor
from docling_core.types.doc import DocItemLabel
from PIL import Image

from docling_pp_doc_layout.options import PPDocLayoutV3Options

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_page(
    page_no: int = 0,
    backend_valid: bool = True,
    has_size: bool = True,
    image: object = "default",
) -> MagicMock:
    page = MagicMock()
    page.page_no = page_no
    if backend_valid:
        page._backend = MagicMock()
        page._backend.is_valid.return_value = True
    else:
        page._backend = None
    page.size = MagicMock() if has_size else None
    if image == "default":
        page.get_image.return_value = Image.new("RGB", (800, 1000)) if has_size else None
    else:
        page.get_image.return_value = image
    page.predictions.layout = None
    return page


def _make_conv_res() -> MagicMock:
    return MagicMock()


def _make_inference_result(
    scores: list[float],
    label_ids: list[int],
    boxes: list[list[float]],
) -> list[dict]:
    return [
        {
            "scores": torch.tensor(scores),
            "labels": torch.tensor(label_ids),
            "boxes": torch.tensor(boxes) if boxes else torch.zeros(0, 4),
        }
    ]


# ---------------------------------------------------------------------------
# Fixture: instantiated model via the plugin entry point
# ---------------------------------------------------------------------------


@pytest.fixture
def model_from_plugin():
    """Instantiate PPDocLayoutV3Model via layout_engines() with mocked HF backend."""
    from docling_pp_doc_layout.plugin import layout_engines

    model_cls = layout_engines()["layout_engines"][0]

    mock_hf_model = MagicMock()
    mock_hf_model.config.id2label = {
        0: "text",
        1: "table",
        2: "image",
        3: "chart",
        4: "doc_title",
        5: "formula",
        6: "paragraph_title",
        7: "footer",
        8: "header",
        9: "footnote",
        10: "algorithm",
        11: "seal",
    }
    mock_hf_model.to.return_value = mock_hf_model

    accel_opts = MagicMock()
    accel_opts.device = "cpu"

    with (
        patch("docling_pp_doc_layout.model.AutoImageProcessor") as mock_aip,
        patch("docling_pp_doc_layout.model.AutoModelForObjectDetection") as mock_aod,
        patch("docling_pp_doc_layout.model.decide_device", return_value="cpu"),
    ):
        mock_aip.from_pretrained.return_value = MagicMock()
        mock_aod.from_pretrained.return_value = mock_hf_model
        return model_cls(
            artifacts_path=None,
            accelerator_options=accel_opts,
            options=PPDocLayoutV3Options(),
        )


# ---------------------------------------------------------------------------
# TestPluginRegistration
# ---------------------------------------------------------------------------


class TestPluginRegistration:
    """Verify the plugin entry point wires up the model correctly."""

    def test_layout_engines_returns_dict_with_list(self):
        from docling_pp_doc_layout.plugin import layout_engines

        result = layout_engines()
        assert isinstance(result, dict)
        assert "layout_engines" in result
        assert isinstance(result["layout_engines"], list)

    def test_registered_class_is_base_layout_model_subclass(self):
        from docling_pp_doc_layout.plugin import layout_engines

        cls = layout_engines()["layout_engines"][0]
        assert issubclass(cls, BaseLayoutModel)

    def test_registered_class_options_type_is_ppdoclayout(self):
        from docling_pp_doc_layout.plugin import layout_engines

        cls = layout_engines()["layout_engines"][0]
        assert cls.get_options_type() is PPDocLayoutV3Options

    def test_model_instantiated_from_plugin_is_base_layout_model(self, model_from_plugin):
        assert isinstance(model_from_plugin, BaseLayoutModel)

    def test_model_options_type_matches_options_used(self, model_from_plugin):
        assert isinstance(model_from_plugin.options, PPDocLayoutV3Options)


# ---------------------------------------------------------------------------
# TestEndToEndSinglePage
# ---------------------------------------------------------------------------


class TestEndToEndSinglePage:
    """Full pipeline with one valid page."""

    def _setup_inference(self, model, scores, label_ids, boxes):
        model._image_processor.return_value = {"pixel_values": torch.zeros(1, 3, 100, 100)}
        model._image_processor.post_process_object_detection.return_value = _make_inference_result(
            scores, label_ids, boxes
        )

    def test_returns_one_layout_prediction_per_page(self, model_from_plugin):
        page = _make_page(page_no=0)
        conv_res = _make_conv_res()
        self._setup_inference(model_from_plugin, [0.9], [0], [[0.0, 0.0, 1.0, 1.0]])

        with (
            patch("docling_pp_doc_layout.model.LayoutPostprocessor") as mock_pp,
            patch("docling_pp_doc_layout.model.TimeRecorder"),
        ):
            mock_pp.return_value.postprocess.return_value = ([], [])
            results = model_from_plugin.predict_layout(conv_res, [page])

        assert len(results) == 1
        assert isinstance(results[0], LayoutPrediction)

    def test_clusters_present_after_pipeline(self, model_from_plugin):
        page = _make_page(page_no=0)
        conv_res = _make_conv_res()
        self._setup_inference(
            model_from_plugin,
            [0.9, 0.8],
            [0, 1],
            [[10.0, 20.0, 50.0, 60.0], [100.0, 200.0, 300.0, 400.0]],
        )

        c1 = MagicMock()
        c1.confidence = 0.9
        c2 = MagicMock()
        c2.confidence = 0.8

        with (
            patch("docling_pp_doc_layout.model.LayoutPostprocessor") as mock_pp,
            patch("docling_pp_doc_layout.model.TimeRecorder"),
        ):
            mock_pp.return_value.postprocess.return_value = ([c1, c2], [])
            results = model_from_plugin.predict_layout(conv_res, [page])

        assert len(results[0].clusters) == 2

    def test_chart_label_mapped_to_picture_not_missing_key(self, model_from_plugin):
        """chart (id=3) → PICTURE must not KeyError in LayoutPostprocessor."""
        page = _make_page(page_no=0)
        conv_res = _make_conv_res()
        self._setup_inference(model_from_plugin, [0.9], [3], [[0.0, 0.0, 1.0, 1.0]])

        captured_clusters = []

        def capture(p, clusters, opts):
            captured_clusters.extend(clusters)
            m = MagicMock()
            m.postprocess.return_value = (clusters, [])
            return m

        with (
            patch("docling_pp_doc_layout.model.LayoutPostprocessor", side_effect=capture),
            patch("docling_pp_doc_layout.model.TimeRecorder"),
        ):
            model_from_plugin.predict_layout(conv_res, [page])

        assert len(captured_clusters) == 1
        assert captured_clusters[0].label == DocItemLabel.PICTURE
        assert captured_clusters[0].label in LayoutPostprocessor.CONFIDENCE_THRESHOLDS

    def test_all_id2label_entries_produce_postprocessor_safe_labels(self, model_from_plugin):
        """Every label in the model's id2label must map to a label that
        LayoutPostprocessor.CONFIDENCE_THRESHOLDS knows about."""
        supported = set(LayoutPostprocessor.CONFIDENCE_THRESHOLDS.keys())
        page = _make_page(page_no=0)
        conv_res = _make_conv_res()

        for label_id, raw_name in model_from_plugin._id2label.items():
            model_from_plugin._image_processor.return_value = {"pixel_values": torch.zeros(1, 3, 100, 100)}
            model_from_plugin._image_processor.post_process_object_detection.return_value = _make_inference_result(
                [0.9], [label_id], [[0.0, 0.0, 1.0, 1.0]]
            )

            captured_clusters = []

            def capture(p, clusters, opts, _cc=captured_clusters):
                _cc.extend(clusters)
                m = MagicMock()
                m.postprocess.return_value = (clusters, [])
                return m

            with (
                patch("docling_pp_doc_layout.model.LayoutPostprocessor", side_effect=capture),
                patch("docling_pp_doc_layout.model.TimeRecorder"),
            ):
                model_from_plugin.predict_layout(conv_res, [page])

            assert len(captured_clusters) == 1, f"Expected 1 cluster for label_id={label_id}"
            assert captured_clusters[0].label in supported, (
                f"id2label[{label_id}]='{raw_name}' → {captured_clusters[0].label!r} not in CONFIDENCE_THRESHOLDS"
            )

    def test_no_detections_yields_empty_clusters(self, model_from_plugin):
        page = _make_page(page_no=0)
        conv_res = _make_conv_res()
        self._setup_inference(model_from_plugin, [], [], [])

        with (
            patch("docling_pp_doc_layout.model.LayoutPostprocessor") as mock_pp,
            patch("docling_pp_doc_layout.model.TimeRecorder"),
        ):
            mock_pp.return_value.postprocess.return_value = ([], [])
            results = model_from_plugin.predict_layout(conv_res, [page])

        assert results[0].clusters == []


# ---------------------------------------------------------------------------
# TestEndToEndMultiPage
# ---------------------------------------------------------------------------


class TestEndToEndMultiPage:
    """Pipeline with multiple pages in a single batch."""

    def test_three_valid_pages_returns_three_predictions(self, model_from_plugin):
        pages = [_make_page(page_no=i) for i in range(3)]
        conv_res = _make_conv_res()

        model_from_plugin._image_processor.return_value = {"pixel_values": torch.zeros(3, 3, 100, 100)}
        model_from_plugin._image_processor.post_process_object_detection.return_value = [
            {"scores": torch.tensor([0.9]), "labels": torch.tensor([0]), "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]])},
            {"scores": torch.tensor([0.8]), "labels": torch.tensor([1]), "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]])},
            {"scores": torch.tensor([0.7]), "labels": torch.tensor([2]), "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]])},
        ]

        c = MagicMock()
        c.confidence = 0.8

        with (
            patch("docling_pp_doc_layout.model.LayoutPostprocessor") as mock_pp,
            patch("docling_pp_doc_layout.model.TimeRecorder"),
        ):
            mock_pp.return_value.postprocess.return_value = ([c], [])
            results = model_from_plugin.predict_layout(conv_res, pages)

        assert len(results) == 3

    def test_inference_called_once_for_whole_batch(self, model_from_plugin):
        """_image_processor must be called once with all images, not per-page."""
        pages = [_make_page(page_no=i) for i in range(2)]
        conv_res = _make_conv_res()

        model_from_plugin._image_processor.return_value = {"pixel_values": torch.zeros(2, 3, 100, 100)}
        model_from_plugin._image_processor.post_process_object_detection.return_value = [
            {"scores": torch.tensor([]), "labels": torch.tensor([], dtype=torch.long), "boxes": torch.zeros(0, 4)},
            {"scores": torch.tensor([]), "labels": torch.tensor([], dtype=torch.long), "boxes": torch.zeros(0, 4)},
        ]

        with (
            patch("docling_pp_doc_layout.model.LayoutPostprocessor") as mock_pp,
            patch("docling_pp_doc_layout.model.TimeRecorder"),
        ):
            mock_pp.return_value.postprocess.return_value = ([], [])
            model_from_plugin.predict_layout(conv_res, pages)

        # _image_processor (as callable) should have been called exactly once
        assert model_from_plugin._image_processor.call_count == 1

    def test_mixed_valid_and_invalid_pages(self, model_from_plugin):
        """Two valid + one invalid → three results, inference called for two images."""
        page0 = _make_page(page_no=0, backend_valid=True)
        page1 = _make_page(page_no=1, backend_valid=False)
        page2 = _make_page(page_no=2, backend_valid=True)
        conv_res = _make_conv_res()

        model_from_plugin._image_processor.return_value = {"pixel_values": torch.zeros(2, 3, 100, 100)}
        model_from_plugin._image_processor.post_process_object_detection.return_value = [
            {"scores": torch.tensor([0.9]), "labels": torch.tensor([0]), "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]])},
            {"scores": torch.tensor([0.8]), "labels": torch.tensor([1]), "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]])},
        ]

        c = MagicMock()
        c.confidence = 0.85

        with (
            patch("docling_pp_doc_layout.model.LayoutPostprocessor") as mock_pp,
            patch("docling_pp_doc_layout.model.TimeRecorder"),
        ):
            mock_pp.return_value.postprocess.return_value = ([c], [])
            results = model_from_plugin.predict_layout(conv_res, [page0, page1, page2])

        assert len(results) == 3
        assert isinstance(results[0], LayoutPrediction)
        assert isinstance(results[1], LayoutPrediction)
        assert isinstance(results[2], LayoutPrediction)


# ---------------------------------------------------------------------------
# TestEndToEndConfidenceThreshold
# ---------------------------------------------------------------------------


class TestEndToEndConfidenceThreshold:
    """Verify that confidence_threshold from options reaches post_process_object_detection."""

    def test_custom_threshold_passed_to_post_process(self, model_from_plugin):
        model_from_plugin.options = PPDocLayoutV3Options(confidence_threshold=0.73)
        page = _make_page(page_no=0)
        conv_res = _make_conv_res()

        model_from_plugin._image_processor.return_value = {"pixel_values": torch.zeros(1, 3, 100, 100)}
        model_from_plugin._image_processor.post_process_object_detection.return_value = [
            {"scores": torch.tensor([]), "labels": torch.tensor([], dtype=torch.long), "boxes": torch.zeros(0, 4)}
        ]

        with (
            patch("docling_pp_doc_layout.model.LayoutPostprocessor") as mock_pp,
            patch("docling_pp_doc_layout.model.TimeRecorder"),
        ):
            mock_pp.return_value.postprocess.return_value = ([], [])
            model_from_plugin.predict_layout(conv_res, [page])

        call_kwargs = model_from_plugin._image_processor.post_process_object_detection.call_args.kwargs
        assert call_kwargs["threshold"] == pytest.approx(0.73, abs=1e-6)

    def test_default_threshold_is_0_5(self, model_from_plugin):
        assert model_from_plugin.options.confidence_threshold == pytest.approx(0.5)

        page = _make_page(page_no=0)
        conv_res = _make_conv_res()

        model_from_plugin._image_processor.return_value = {"pixel_values": torch.zeros(1, 3, 100, 100)}
        model_from_plugin._image_processor.post_process_object_detection.return_value = [
            {"scores": torch.tensor([]), "labels": torch.tensor([], dtype=torch.long), "boxes": torch.zeros(0, 4)}
        ]

        with (
            patch("docling_pp_doc_layout.model.LayoutPostprocessor") as mock_pp,
            patch("docling_pp_doc_layout.model.TimeRecorder"),
        ):
            mock_pp.return_value.postprocess.return_value = ([], [])
            model_from_plugin.predict_layout(conv_res, [page])

        call_kwargs = model_from_plugin._image_processor.post_process_object_detection.call_args.kwargs
        assert call_kwargs["threshold"] == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# TestEndToEndTargetSizes
# ---------------------------------------------------------------------------


class TestEndToEndTargetSizes:
    """Verify that image dimensions are correctly passed to post_process_object_detection."""

    def test_target_sizes_are_height_width_tuples(self, model_from_plugin):
        """PIL Image.size is (width, height); reversed gives (height, width) for HF API."""
        img = Image.new("RGB", (640, 480))  # width=640, height=480
        page = _make_page(page_no=0, image=img)
        conv_res = _make_conv_res()

        model_from_plugin._image_processor.return_value = {"pixel_values": torch.zeros(1, 3, 480, 640)}
        model_from_plugin._image_processor.post_process_object_detection.return_value = [
            {"scores": torch.tensor([]), "labels": torch.tensor([], dtype=torch.long), "boxes": torch.zeros(0, 4)}
        ]

        with (
            patch("docling_pp_doc_layout.model.LayoutPostprocessor") as mock_pp,
            patch("docling_pp_doc_layout.model.TimeRecorder"),
        ):
            mock_pp.return_value.postprocess.return_value = ([], [])
            model_from_plugin.predict_layout(conv_res, [page])

        call_kwargs = model_from_plugin._image_processor.post_process_object_detection.call_args.kwargs
        # (height=480, width=640)
        assert call_kwargs["target_sizes"] == [(480, 640)]
