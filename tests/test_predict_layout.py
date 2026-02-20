"""Tests for PPDocLayoutV3Model.__init__ and predict_layout.

These tests mock all heavy dependencies (transformers, LayoutPostprocessor,
TimeRecorder) so they run without GPU or real model weights.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest
from docling.datamodel.base_models import BoundingBox, Cluster, LayoutPrediction
from docling_core.types.doc import DocItemLabel
from PIL import Image

from docling_pp_doc_layout.options import PPDocLayoutV3Options

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_page(
    page_no: int = 0, backend_valid: bool = True, has_size: bool = True, image: object = "default"
) -> MagicMock:
    """Return a mock Page with configurable validity properties."""
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


def _make_cluster(confidence: float = 0.9, ix: int = 0) -> Cluster:
    """Return a real Cluster so it passes LayoutPrediction Pydantic validation."""
    return Cluster(
        id=ix,
        label=DocItemLabel.TEXT,
        confidence=confidence,
        bbox=BoundingBox(l=0.0, t=0.0, r=1.0, b=1.0),
        cells=[],
    )


@pytest.fixture
def model_instance() -> MagicMock:
    """PPDocLayoutV3Model with __init__ bypassed and internals mocked."""
    from docling_pp_doc_layout.model import PPDocLayoutV3Model

    with patch.object(PPDocLayoutV3Model, "__init__", lambda self, *a, **kw: None):
        instance = PPDocLayoutV3Model.__new__(PPDocLayoutV3Model)
        instance.options = PPDocLayoutV3Options()
        instance._device = "cpu"
        instance._image_processor = MagicMock()
        instance._model = MagicMock()
        instance._id2label = {0: "text", 1: "table", 2: "image"}
        return instance


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------


class TestInit:
    """Verify that __init__ wires up all attributes correctly."""

    @pytest.fixture
    def hf_mocks(self):
        """Patch AutoImageProcessor, AutoModelForObjectDetection and decide_device."""
        mock_hf_model = MagicMock()
        mock_hf_model.config.id2label = {0: "text", 1: "table"}
        mock_hf_model.to.return_value = mock_hf_model
        mock_processor = MagicMock()

        with (
            patch("docling_pp_doc_layout.model.AutoImageProcessor") as mock_aip,
            patch("docling_pp_doc_layout.model.AutoModelForObjectDetection") as mock_aod,
            patch("docling_pp_doc_layout.model.decide_device", return_value="cpu"),
        ):
            mock_aip.from_pretrained.return_value = mock_processor
            mock_aod.from_pretrained.return_value = mock_hf_model
            yield {
                "mock_aip": mock_aip,
                "mock_aod": mock_aod,
                "mock_processor": mock_processor,
                "mock_hf_model": mock_hf_model,
            }

    def _build_model(self, hf_mocks, opts=None, device="cpu"):
        from docling_pp_doc_layout.model import PPDocLayoutV3Model

        accel_opts = MagicMock()
        accel_opts.device = device
        return PPDocLayoutV3Model(
            artifacts_path=None,
            accelerator_options=accel_opts,
            options=opts or PPDocLayoutV3Options(),
        )

    def test_stores_options(self, hf_mocks):
        opts = PPDocLayoutV3Options(confidence_threshold=0.3)
        model = self._build_model(hf_mocks, opts=opts)
        assert model.options is opts

    def test_stores_artifacts_path(self, hf_mocks):
        model = self._build_model(hf_mocks)
        assert model.artifacts_path is None

    def test_stores_accelerator_options(self, hf_mocks):
        from docling_pp_doc_layout.model import PPDocLayoutV3Model

        accel_opts = MagicMock()
        accel_opts.device = "cpu"
        with (
            patch("docling_pp_doc_layout.model.AutoModelForObjectDetection") as mock_aod,
            patch("docling_pp_doc_layout.model.decide_device", return_value="cpu"),
        ):
            mock_hf_model = MagicMock()
            mock_hf_model.config.id2label = {}
            mock_hf_model.to.return_value = mock_hf_model
            mock_aod.from_pretrained.return_value = mock_hf_model
            model = PPDocLayoutV3Model(
                artifacts_path=None,
                accelerator_options=accel_opts,
                options=PPDocLayoutV3Options(),
            )
        assert model.accelerator_options is accel_opts

    def test_model_moved_to_device(self, hf_mocks):
        with patch("docling_pp_doc_layout.model.decide_device", return_value="cuda"):
            self._build_model(hf_mocks, device="cuda")
        hf_mocks["mock_hf_model"].to.assert_called_with("cuda")

    def test_model_set_to_eval(self, hf_mocks):
        self._build_model(hf_mocks)
        hf_mocks["mock_hf_model"].eval.assert_called_once()

    def test_id2label_populated_from_config(self, hf_mocks):
        expected = {0: "text", 1: "table"}
        hf_mocks["mock_hf_model"].config.id2label = expected
        model = self._build_model(hf_mocks)
        assert model._id2label is expected

    def test_image_processor_stored(self, hf_mocks):
        model = self._build_model(hf_mocks)
        assert model._image_processor is hf_mocks["mock_processor"]

    def test_from_pretrained_called_with_model_name(self, hf_mocks):
        custom_name = "my-org/custom-layout-model"
        self._build_model(hf_mocks, opts=PPDocLayoutV3Options(model_name=custom_name))
        hf_mocks["mock_aip"].from_pretrained.assert_called_once_with(custom_name)
        hf_mocks["mock_aod"].from_pretrained.assert_called_once_with(custom_name)

    def test_enable_remote_services_is_accepted_and_ignored(self, hf_mocks):
        """enable_remote_services is a keyword-only arg that must not crash."""
        from docling_pp_doc_layout.model import PPDocLayoutV3Model

        accel_opts = MagicMock()
        accel_opts.device = "cpu"
        # Should not raise
        PPDocLayoutV3Model(
            artifacts_path=None,
            accelerator_options=accel_opts,
            options=PPDocLayoutV3Options(),
            enable_remote_services=True,
        )


# ---------------------------------------------------------------------------
# TestPredictLayoutHappyPath
# ---------------------------------------------------------------------------


class TestPredictLayoutHappyPath:
    """Happy-path scenarios for predict_layout."""

    def test_single_valid_page_returns_one_prediction(self, model_instance):
        page = _make_page(page_no=0)
        conv_res = _make_conv_res()
        model_instance._run_inference = MagicMock(return_value=[[]])

        with (
            patch("docling_pp_doc_layout.model.LayoutPostprocessor") as mock_pp,
            patch("docling_pp_doc_layout.model.TimeRecorder"),
        ):
            mock_pp.return_value.postprocess.return_value = ([], [])
            results = model_instance.predict_layout(conv_res, [page])

        assert len(results) == 1
        assert isinstance(results[0], LayoutPrediction)

    def test_multiple_valid_pages_returns_correct_count(self, model_instance):
        pages = [_make_page(page_no=i) for i in range(3)]
        conv_res = _make_conv_res()
        model_instance._run_inference = MagicMock(return_value=[[], [], []])

        with (
            patch("docling_pp_doc_layout.model.LayoutPostprocessor") as mock_pp,
            patch("docling_pp_doc_layout.model.TimeRecorder"),
        ):
            mock_pp.return_value.postprocess.return_value = ([], [])
            results = model_instance.predict_layout(conv_res, pages)

        assert len(results) == 3

    def test_empty_page_list_returns_empty(self, model_instance):
        conv_res = _make_conv_res()
        model_instance._run_inference = MagicMock()

        results = model_instance.predict_layout(conv_res, [])

        assert results == []
        model_instance._run_inference.assert_not_called()

    def test_run_inference_called_with_valid_images(self, model_instance):
        img = Image.new("RGB", (200, 300))
        page = _make_page(page_no=0, image=img)
        conv_res = _make_conv_res()
        model_instance._run_inference = MagicMock(return_value=[[]])

        with (
            patch("docling_pp_doc_layout.model.LayoutPostprocessor") as mock_pp,
            patch("docling_pp_doc_layout.model.TimeRecorder"),
        ):
            mock_pp.return_value.postprocess.return_value = ([], [])
            model_instance.predict_layout(conv_res, [page])

        model_instance._run_inference.assert_called_once_with([img])

    def test_images_fetched_at_scale_one(self, model_instance):
        page = _make_page(page_no=0)
        conv_res = _make_conv_res()
        model_instance._run_inference = MagicMock(return_value=[[]])

        with (
            patch("docling_pp_doc_layout.model.LayoutPostprocessor") as mock_pp,
            patch("docling_pp_doc_layout.model.TimeRecorder"),
        ):
            mock_pp.return_value.postprocess.return_value = ([], [])
            model_instance.predict_layout(conv_res, [page])

        page.get_image.assert_called_once_with(scale=1.0)

    def test_detections_become_clusters_passed_to_postprocessor(self, model_instance):
        page = _make_page(page_no=0)
        conv_res = _make_conv_res()
        detections = [
            {"label": DocItemLabel.TEXT, "confidence": 0.9, "l": 10.0, "t": 20.0, "r": 50.0, "b": 60.0},
            {"label": DocItemLabel.TABLE, "confidence": 0.8, "l": 5.0, "t": 5.0, "r": 100.0, "b": 200.0},
        ]
        model_instance._run_inference = MagicMock(return_value=[detections])

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
            model_instance.predict_layout(conv_res, [page])

        assert len(captured_clusters) == 2
        assert captured_clusters[0].label == DocItemLabel.TEXT
        assert captured_clusters[1].label == DocItemLabel.TABLE

    def test_cluster_ids_are_sequential_from_zero(self, model_instance):
        page = _make_page(page_no=0)
        conv_res = _make_conv_res()
        detections = [
            {"label": DocItemLabel.TEXT, "confidence": 0.9, "l": 0.0, "t": 0.0, "r": 1.0, "b": 1.0},
            {"label": DocItemLabel.TABLE, "confidence": 0.8, "l": 0.0, "t": 0.0, "r": 1.0, "b": 1.0},
            {"label": DocItemLabel.PICTURE, "confidence": 0.7, "l": 0.0, "t": 0.0, "r": 1.0, "b": 1.0},
        ]
        model_instance._run_inference = MagicMock(return_value=[detections])

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
            model_instance.predict_layout(conv_res, [page])

        assert [c.id for c in captured_clusters] == [0, 1, 2]

    def test_postprocessed_clusters_end_up_in_prediction(self, model_instance):
        page = _make_page(page_no=0)
        conv_res = _make_conv_res()
        model_instance._run_inference = MagicMock(return_value=[[]])

        final_cluster = _make_cluster(confidence=0.99)

        with (
            patch("docling_pp_doc_layout.model.LayoutPostprocessor") as mock_pp,
            patch("docling_pp_doc_layout.model.TimeRecorder"),
        ):
            mock_pp.return_value.postprocess.return_value = ([final_cluster], [])
            results = model_instance.predict_layout(conv_res, [page])

        assert results[0].clusters == [final_cluster]

    def test_cluster_bbox_coordinates_are_correct(self, model_instance):
        page = _make_page(page_no=0)
        conv_res = _make_conv_res()
        detections = [
            {"label": DocItemLabel.TEXT, "confidence": 0.9, "l": 11.1, "t": 22.2, "r": 33.3, "b": 44.4},
        ]
        model_instance._run_inference = MagicMock(return_value=[detections])

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
            model_instance.predict_layout(conv_res, [page])

        bbox = captured_clusters[0].bbox
        assert bbox.l == pytest.approx(11.1, abs=0.001)
        assert bbox.t == pytest.approx(22.2, abs=0.001)
        assert bbox.r == pytest.approx(33.3, abs=0.001)
        assert bbox.b == pytest.approx(44.4, abs=0.001)

    def test_postprocessor_called_with_correct_page_and_options(self, model_instance):
        page = _make_page(page_no=0)
        conv_res = _make_conv_res()
        detections = [{"label": DocItemLabel.TEXT, "confidence": 0.9, "l": 0.0, "t": 0.0, "r": 1.0, "b": 1.0}]
        model_instance._run_inference = MagicMock(return_value=[detections])

        with (
            patch("docling_pp_doc_layout.model.LayoutPostprocessor") as mock_pp,
            patch("docling_pp_doc_layout.model.TimeRecorder"),
        ):
            mock_pp.return_value.postprocess.return_value = ([], [])
            model_instance.predict_layout(conv_res, [page])

        call_args = mock_pp.call_args
        assert call_args[0][0] is page
        assert call_args[0][2] is model_instance.options

    def test_cluster_confidence_stored(self, model_instance):
        page = _make_page(page_no=0)
        conv_res = _make_conv_res()
        detections = [
            {"label": DocItemLabel.TEXT, "confidence": 0.77, "l": 0.0, "t": 0.0, "r": 1.0, "b": 1.0},
        ]
        model_instance._run_inference = MagicMock(return_value=[detections])

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
            model_instance.predict_layout(conv_res, [page])

        assert captured_clusters[0].confidence == pytest.approx(0.77, abs=0.001)


# ---------------------------------------------------------------------------
# TestPredictLayoutPageFiltering
# ---------------------------------------------------------------------------


class TestPredictLayoutPageFiltering:
    """Tests for how invalid/semi-invalid pages are handled — including bug exposure."""

    def test_page_with_none_backend_uses_existing_prediction(self, model_instance):
        existing = LayoutPrediction()
        page = _make_page(page_no=0, backend_valid=False)
        page.predictions.layout = existing
        conv_res = _make_conv_res()

        results = model_instance.predict_layout(conv_res, [page])

        assert results[0] is existing

    def test_page_with_none_backend_and_none_layout_produces_empty_prediction(self, model_instance):
        page = _make_page(page_no=0, backend_valid=False)
        page.predictions.layout = None
        conv_res = _make_conv_res()

        results = model_instance.predict_layout(conv_res, [page])

        assert isinstance(results[0], LayoutPrediction)

    def test_page_with_invalid_backend_is_treated_like_missing_backend(self, model_instance):
        page = _make_page(page_no=0, backend_valid=True)
        page._backend.is_valid.return_value = False
        page.predictions.layout = None
        conv_res = _make_conv_res()

        results = model_instance.predict_layout(conv_res, [page])

        assert len(results) == 1
        assert isinstance(results[0], LayoutPrediction)

    def test_no_valid_pages_skips_inference(self, model_instance):
        page = _make_page(page_no=0, backend_valid=False)
        conv_res = _make_conv_res()
        model_instance._run_inference = MagicMock()

        model_instance.predict_layout(conv_res, [page])

        model_instance._run_inference.assert_not_called()

    def test_page_with_valid_backend_but_none_size_triggers_indexerror(self, model_instance):
        """Bug: second loop doesn't guard against pages skipped due to None size.

        When a page has a valid backend but None size, it is excluded from
        batch_detections in the first loop. However the second loop only checks
        the backend — so it tries to consume a batch_detections entry that was
        never created, causing IndexError.
        """
        page = _make_page(page_no=0, backend_valid=True, has_size=False)
        conv_res = _make_conv_res()
        # _run_inference is never called because valid_images is empty,
        # so batch_detections == []. The second loop then tries batch_detections[0].
        model_instance._run_inference = MagicMock(return_value=[])

        with pytest.raises(IndexError):
            model_instance.predict_layout(conv_res, [page])

    def test_page_with_valid_backend_but_none_image_triggers_indexerror(self, model_instance):
        """Bug: same as above but for get_image() returning None."""
        page = _make_page(page_no=0, backend_valid=True, has_size=True, image=None)
        conv_res = _make_conv_res()
        model_instance._run_inference = MagicMock(return_value=[])

        with pytest.raises(IndexError):
            model_instance.predict_layout(conv_res, [page])

    def test_semi_invalid_page_before_valid_page_causes_detection_misassignment_and_indexerror(self, model_instance):
        """Bug: the semi-invalid page (valid backend, no size) steals the valid
        page's detections at valid_idx=0; then the valid page tries valid_idx=1
        which is out of range, raising IndexError.
        """
        page_no_size = _make_page(page_no=0, backend_valid=True, has_size=False)
        page_valid = _make_page(page_no=1, backend_valid=True, has_size=True)
        conv_res = _make_conv_res()

        # Only page_valid ends up in valid_images, so _run_inference returns one entry.
        model_instance._run_inference = MagicMock(return_value=[[]])

        with (
            patch("docling_pp_doc_layout.model.LayoutPostprocessor") as mock_pp,
            patch("docling_pp_doc_layout.model.TimeRecorder"),
        ):
            mock_pp.return_value.postprocess.return_value = ([], [])
            with pytest.raises(IndexError):
                model_instance.predict_layout(conv_res, [page_no_size, page_valid])

    def test_valid_and_invalid_backend_pages_separated_correctly(self, model_instance):
        """Only pages with valid backends contribute to inference; others get existing predictions."""
        page_valid = _make_page(page_no=0, backend_valid=True)
        page_invalid = _make_page(page_no=1, backend_valid=False)
        conv_res = _make_conv_res()
        model_instance._run_inference = MagicMock(return_value=[[]])

        with (
            patch("docling_pp_doc_layout.model.LayoutPostprocessor") as mock_pp,
            patch("docling_pp_doc_layout.model.TimeRecorder"),
        ):
            mock_pp.return_value.postprocess.return_value = ([], [])
            results = model_instance.predict_layout(conv_res, [page_valid, page_invalid])

        assert len(results) == 2
        model_instance._run_inference.assert_called_once_with([page_valid.get_image.return_value])

    def test_all_valid_pages_all_get_layout_predictions(self, model_instance):
        pages = [_make_page(page_no=i) for i in range(4)]
        conv_res = _make_conv_res()
        model_instance._run_inference = MagicMock(return_value=[[], [], [], []])

        with (
            patch("docling_pp_doc_layout.model.LayoutPostprocessor") as mock_pp,
            patch("docling_pp_doc_layout.model.TimeRecorder"),
        ):
            mock_pp.return_value.postprocess.return_value = ([], [])
            results = model_instance.predict_layout(conv_res, pages)

        assert len(results) == 4
        assert all(isinstance(r, LayoutPrediction) for r in results)


# ---------------------------------------------------------------------------
# TestPredictLayoutConfidenceScores
# ---------------------------------------------------------------------------


class TestPredictLayoutConfidenceScores:
    """Tests for layout_score and ocr_score written to ConversionResult."""

    def test_layout_score_is_mean_of_processed_cluster_confidences(self, model_instance):
        page = _make_page(page_no=7)
        conv_res = _make_conv_res()
        model_instance._run_inference = MagicMock(return_value=[[]])

        c1 = _make_cluster(confidence=0.6, ix=0)
        c2 = _make_cluster(confidence=0.8, ix=1)

        with (
            patch("docling_pp_doc_layout.model.LayoutPostprocessor") as mock_pp,
            patch("docling_pp_doc_layout.model.TimeRecorder"),
        ):
            mock_pp.return_value.postprocess.return_value = ([c1, c2], [])
            model_instance.predict_layout(conv_res, [page])

        assert conv_res.confidence.pages[7].layout_score == pytest.approx(0.7, abs=0.001)

    def test_ocr_score_uses_only_cells_where_from_ocr_is_true(self, model_instance):
        page = _make_page(page_no=3)
        conv_res = _make_conv_res()
        model_instance._run_inference = MagicMock(return_value=[[]])

        ocr_cell = MagicMock()
        ocr_cell.from_ocr = True
        ocr_cell.confidence = 0.75

        non_ocr_cell = MagicMock()
        non_ocr_cell.from_ocr = False
        non_ocr_cell.confidence = 0.1

        with (
            patch("docling_pp_doc_layout.model.LayoutPostprocessor") as mock_pp,
            patch("docling_pp_doc_layout.model.TimeRecorder"),
        ):
            mock_pp.return_value.postprocess.return_value = ([], [ocr_cell, non_ocr_cell])
            model_instance.predict_layout(conv_res, [page])

        assert conv_res.confidence.pages[3].ocr_score == pytest.approx(0.75, abs=0.001)

    def test_layout_score_is_nan_when_no_clusters(self, model_instance):
        page = _make_page(page_no=0)
        conv_res = _make_conv_res()
        model_instance._run_inference = MagicMock(return_value=[[]])

        with (
            patch("docling_pp_doc_layout.model.LayoutPostprocessor") as mock_pp,
            patch("docling_pp_doc_layout.model.TimeRecorder"),
        ):
            mock_pp.return_value.postprocess.return_value = ([], [])
            model_instance.predict_layout(conv_res, [page])

        assert math.isnan(conv_res.confidence.pages[0].layout_score)

    def test_ocr_score_is_nan_when_no_ocr_cells(self, model_instance):
        page = _make_page(page_no=0)
        conv_res = _make_conv_res()
        model_instance._run_inference = MagicMock(return_value=[[]])

        with (
            patch("docling_pp_doc_layout.model.LayoutPostprocessor") as mock_pp,
            patch("docling_pp_doc_layout.model.TimeRecorder"),
        ):
            mock_pp.return_value.postprocess.return_value = ([], [])
            model_instance.predict_layout(conv_res, [page])

        assert math.isnan(conv_res.confidence.pages[0].ocr_score)

    def test_scores_written_to_correct_page_no(self, model_instance):
        page = _make_page(page_no=42)
        conv_res = _make_conv_res()
        model_instance._run_inference = MagicMock(return_value=[[]])

        c = _make_cluster(confidence=0.9)

        with (
            patch("docling_pp_doc_layout.model.LayoutPostprocessor") as mock_pp,
            patch("docling_pp_doc_layout.model.TimeRecorder"),
        ):
            mock_pp.return_value.postprocess.return_value = ([c], [])
            model_instance.predict_layout(conv_res, [page])

        assert conv_res.confidence.pages[42].layout_score == pytest.approx(0.9, abs=0.001)

    def test_scores_not_written_for_invalid_backend_pages(self, model_instance):
        page = _make_page(page_no=0, backend_valid=False)
        conv_res = _make_conv_res()

        model_instance.predict_layout(conv_res, [page])

        conv_res.confidence.pages.__getitem__.assert_not_called()

    def test_ocr_score_multiple_ocr_cells_averaged(self, model_instance):
        page = _make_page(page_no=0)
        conv_res = _make_conv_res()
        model_instance._run_inference = MagicMock(return_value=[[]])

        cells = []
        for conf in (0.5, 0.7, 0.9):
            cell = MagicMock()
            cell.from_ocr = True
            cell.confidence = conf
            cells.append(cell)

        with (
            patch("docling_pp_doc_layout.model.LayoutPostprocessor") as mock_pp,
            patch("docling_pp_doc_layout.model.TimeRecorder"),
        ):
            mock_pp.return_value.postprocess.return_value = ([], cells)
            model_instance.predict_layout(conv_res, [page])

        assert conv_res.confidence.pages[0].ocr_score == pytest.approx(0.7, abs=0.001)
