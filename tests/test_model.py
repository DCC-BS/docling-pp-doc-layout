"""Tests for PPDocLayoutV3Model.

These tests mock ``transformers`` and the heavy model objects so they can
run without a GPU or the actual model weights.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from docling.datamodel.pipeline_options import BaseLayoutOptions
from docling_core.types.doc import DocItemLabel

from docling_pp_doc_layout.options import PPDocLayoutV3Options


class TestGetOptionsType:
    def test_returns_options_class(self):
        from docling_pp_doc_layout.model import PPDocLayoutV3Model

        assert PPDocLayoutV3Model.get_options_type() is PPDocLayoutV3Options

    def test_is_subclass_of_base(self):
        from docling_pp_doc_layout.model import PPDocLayoutV3Model

        assert issubclass(PPDocLayoutV3Model.get_options_type(), BaseLayoutOptions)


class TestRunInference:
    """Test ``_run_inference`` with a fully mocked model."""

    @pytest.fixture
    def model_instance(self):
        """Create a PPDocLayoutV3Model with mocked transformers internals."""
        from docling_pp_doc_layout.model import PPDocLayoutV3Model

        with (
            patch.object(PPDocLayoutV3Model, "__init__", lambda self, *a, **kw: None),
        ):
            instance = PPDocLayoutV3Model.__new__(PPDocLayoutV3Model)
            instance.options = PPDocLayoutV3Options()
            instance._device = "cpu"
            instance._image_processor = MagicMock()
            instance._model = MagicMock()
            instance._id2label = {
                0: "text",
                1: "table",
                2: "image",
                3: "chart",
                4: "doc_title",
                5: "unknown_label",
            }
            return instance

    def test_returns_list_per_image(self, model_instance):
        import torch
        from PIL import Image

        img = Image.new("RGB", (100, 100))

        model_instance._image_processor.return_value = {"pixel_values": torch.zeros(1, 3, 100, 100)}
        model_instance._image_processor.post_process_object_detection.return_value = [
            {
                "scores": torch.tensor([0.9, 0.8]),
                "labels": torch.tensor([0, 1]),
                "boxes": torch.tensor([[10.0, 20.0, 50.0, 60.0], [100.0, 200.0, 300.0, 400.0]]),
            }
        ]

        result = model_instance._run_inference([img])

        assert len(result) == 1
        assert len(result[0]) == 2

    def test_detection_dict_keys(self, model_instance):
        import torch
        from PIL import Image

        img = Image.new("RGB", (100, 100))

        model_instance._image_processor.return_value = {"pixel_values": torch.zeros(1, 3, 100, 100)}
        model_instance._image_processor.post_process_object_detection.return_value = [
            {
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
                "boxes": torch.tensor([[10.0, 20.0, 50.0, 60.0]]),
            }
        ]

        result = model_instance._run_inference([img])
        det = result[0][0]

        assert set(det.keys()) == {"label", "confidence", "l", "t", "r", "b"}

    def test_label_mapping_applied(self, model_instance):
        import torch
        from PIL import Image

        img = Image.new("RGB", (100, 100))

        model_instance._image_processor.return_value = {"pixel_values": torch.zeros(1, 3, 100, 100)}
        model_instance._image_processor.post_process_object_detection.return_value = [
            {
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([1]),
                "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            }
        ]

        result = model_instance._run_inference([img])
        assert result[0][0]["label"] == DocItemLabel.TABLE

    def test_unknown_label_defaults_to_text(self, model_instance):
        import torch
        from PIL import Image

        img = Image.new("RGB", (100, 100))

        model_instance._image_processor.return_value = {"pixel_values": torch.zeros(1, 3, 100, 100)}
        model_instance._image_processor.post_process_object_detection.return_value = [
            {
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([5]),
                "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            }
        ]

        result = model_instance._run_inference([img])
        assert result[0][0]["label"] == DocItemLabel.TEXT

    def test_chart_label_maps_to_picture(self, model_instance):
        """Regression: 'chart' used to map to DocItemLabel.CHART which is
        not in LayoutPostprocessor.CONFIDENCE_THRESHOLDS."""
        import torch
        from PIL import Image

        img = Image.new("RGB", (100, 100))

        model_instance._image_processor.return_value = {"pixel_values": torch.zeros(1, 3, 100, 100)}
        model_instance._image_processor.post_process_object_detection.return_value = [
            {
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([3]),
                "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            }
        ]

        result = model_instance._run_inference([img])
        assert result[0][0]["label"] == DocItemLabel.PICTURE

    def test_bounding_box_coordinates(self, model_instance):
        import torch
        from PIL import Image

        img = Image.new("RGB", (100, 100))

        model_instance._image_processor.return_value = {"pixel_values": torch.zeros(1, 3, 100, 100)}
        model_instance._image_processor.post_process_object_detection.return_value = [
            {
                "scores": torch.tensor([0.95]),
                "labels": torch.tensor([0]),
                "boxes": torch.tensor([[10.5, 20.3, 50.7, 60.1]]),
            }
        ]

        result = model_instance._run_inference([img])
        det = result[0][0]
        assert det["l"] == pytest.approx(10.5, abs=0.01)
        assert det["t"] == pytest.approx(20.3, abs=0.01)
        assert det["r"] == pytest.approx(50.7, abs=0.01)
        assert det["b"] == pytest.approx(60.1, abs=0.01)

    def test_confidence_score_preserved(self, model_instance):
        import torch
        from PIL import Image

        img = Image.new("RGB", (100, 100))

        model_instance._image_processor.return_value = {"pixel_values": torch.zeros(1, 3, 100, 100)}
        model_instance._image_processor.post_process_object_detection.return_value = [
            {
                "scores": torch.tensor([0.87]),
                "labels": torch.tensor([0]),
                "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            }
        ]

        result = model_instance._run_inference([img])
        assert result[0][0]["confidence"] == pytest.approx(0.87, abs=0.01)

    def test_multiple_images_batch(self, model_instance):
        import torch
        from PIL import Image

        images = [Image.new("RGB", (100, 100)), Image.new("RGB", (200, 200))]

        model_instance._image_processor.return_value = {"pixel_values": torch.zeros(2, 3, 100, 100)}
        model_instance._image_processor.post_process_object_detection.return_value = [
            {
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
                "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            },
            {
                "scores": torch.tensor([0.8, 0.7]),
                "labels": torch.tensor([1, 2]),
                "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]]),
            },
        ]

        result = model_instance._run_inference(images)
        assert len(result) == 2
        assert len(result[0]) == 1
        assert len(result[1]) == 2

    def test_empty_detections(self, model_instance):
        import torch
        from PIL import Image

        img = Image.new("RGB", (100, 100))

        model_instance._image_processor.return_value = {"pixel_values": torch.zeros(1, 3, 100, 100)}
        model_instance._image_processor.post_process_object_detection.return_value = [
            {
                "scores": torch.tensor([]),
                "labels": torch.tensor([], dtype=torch.long),
                "boxes": torch.zeros(0, 4),
            }
        ]

        result = model_instance._run_inference([img])
        assert result == [[]]


class TestAllDetectedLabelsArePostprocessorSafe:
    """Integration-style: run _run_inference for every known id2label value
    and ensure nothing would crash LayoutPostprocessor."""

    @pytest.fixture
    def model_instance(self):
        from docling_pp_doc_layout.model import PPDocLayoutV3Model

        with patch.object(PPDocLayoutV3Model, "__init__", lambda self, *a, **kw: None):
            instance = PPDocLayoutV3Model.__new__(PPDocLayoutV3Model)
            instance.options = PPDocLayoutV3Options()
            instance._device = "cpu"
            instance._image_processor = MagicMock()
            instance._model = MagicMock()
            instance._id2label = {
                0: "abstract",
                1: "algorithm",
                2: "aside_text",
                3: "chart",
                4: "content",
                5: "doc_title",
                6: "figure_title",
                7: "footer",
                8: "footnote",
                9: "formula",
                10: "formula_number",
                11: "header",
                12: "image",
                13: "number",
                14: "paragraph_title",
                15: "reference",
                16: "reference_content",
                17: "seal",
                18: "table",
                19: "text",
                20: "vision_footnote",
            }
            return instance

    def test_all_known_labels_produce_supported_doc_item_labels(self, model_instance):
        import torch
        from docling.utils.layout_postprocessor import LayoutPostprocessor
        from PIL import Image

        supported = set(LayoutPostprocessor.CONFIDENCE_THRESHOLDS.keys())
        img = Image.new("RGB", (100, 100))

        for label_id, raw_name in model_instance._id2label.items():
            model_instance._image_processor.return_value = {"pixel_values": torch.zeros(1, 3, 100, 100)}
            model_instance._image_processor.post_process_object_detection.return_value = [
                {
                    "scores": torch.tensor([0.9]),
                    "labels": torch.tensor([label_id]),
                    "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
                }
            ]

            result = model_instance._run_inference([img])
            detected_label = result[0][0]["label"]
            assert detected_label in supported, (
                f"id2label[{label_id}]='{raw_name}' → {detected_label!r} is NOT in CONFIDENCE_THRESHOLDS"
            )
