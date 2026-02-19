"""PP-DocLayout-V3 layout model for the docling standard pipeline.

Runs PaddlePaddle PP-DocLayout-V3 locally via HuggingFace ``transformers``
to detect document layout elements and returns ``LayoutPrediction`` objects
that docling merges with its standard-pipeline output.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np
import torch
from docling.datamodel.base_models import BoundingBox, Cluster, LayoutPrediction, Page
from docling.models.base_layout_model import BaseLayoutModel
from docling.utils.accelerator_utils import decide_device
from docling.utils.layout_postprocessor import LayoutPostprocessor
from docling.utils.profiling import TimeRecorder
from docling_core.types.doc import DocItemLabel
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from docling_pp_doc_layout.label_mapping import LABEL_MAP
from docling_pp_doc_layout.options import PPDocLayoutV3Options

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.document import ConversionResult
    from docling.datamodel.pipeline_options import BaseLayoutOptions
    from PIL import Image

logger = logging.getLogger(__name__)


class PPDocLayoutV3Model(BaseLayoutModel):
    """Layout engine using PP-DocLayout-V3 via HuggingFace transformers."""

    def __init__(
        self,
        artifacts_path: Path | None,
        accelerator_options: AcceleratorOptions,
        options: PPDocLayoutV3Options,
        *,
        enable_remote_services: bool = False,  # noqa: ARG002
    ) -> None:
        self.options = options
        self.artifacts_path = artifacts_path
        self.accelerator_options = accelerator_options

        self._device = decide_device(accelerator_options.device)
        logger.info(
            "Loading PP-DocLayout-V3 model %s on device=%s",
            options.model_name,
            self._device,
        )

        self._image_processor = AutoImageProcessor.from_pretrained(
            options.model_name,
        )
        self._model = AutoModelForObjectDetection.from_pretrained(
            options.model_name,
        ).to(self._device)
        self._model.eval()

        self._id2label: dict[int, str] = self._model.config.id2label
        logger.info("PP-DocLayout-V3 model loaded successfully")

    @classmethod
    def get_options_type(cls) -> type[BaseLayoutOptions]:
        """Return the options class for this layout model."""
        return PPDocLayoutV3Options

    def _run_inference(
        self,
        images: list[Image.Image],
    ) -> list[list[dict]]:
        """Run PP-DocLayout-V3 on a batch of PIL images.

        Returns a list (per image) of lists of detection dicts with keys
        ``label``, ``confidence``, ``l``, ``t``, ``r``, ``b``.
        """
        inputs = self._image_processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        target_sizes = [img.size[::-1] for img in images]  # (height, width)
        results = self._image_processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self.options.confidence_threshold,
        )

        batch_detections: list[list[dict]] = []
        for result in results:
            detections: list[dict] = []
            for score, label_id, box in zip(
                result["scores"],
                result["labels"],
                result["boxes"],
                strict=True,
            ):
                raw_label = self._id2label.get(label_id.item(), "text")
                doc_label = LABEL_MAP.get(raw_label, DocItemLabel.TEXT)
                x_min, y_min, x_max, y_max = box.tolist()
                detections.append({
                    "label": doc_label,
                    "confidence": score.item(),
                    "l": x_min,
                    "t": y_min,
                    "r": x_max,
                    "b": y_max,
                })
            batch_detections.append(detections)

        return batch_detections

    def predict_layout(
        self,
        conv_res: ConversionResult,
        pages: Sequence[Page],
    ) -> Sequence[LayoutPrediction]:
        """Detect layout regions for a batch of document pages."""
        pages = list(pages)

        valid_pages: list[Page] = []
        valid_images: list[Image.Image] = []

        for page in pages:
            if page._backend is None or not page._backend.is_valid():  # noqa: SLF001
                continue
            if page.size is None:
                continue
            page_image = page.get_image(scale=1.0)
            if page_image is None:
                continue
            valid_pages.append(page)
            valid_images.append(page_image)

        batch_detections: list[list[dict]] = []
        if valid_images:
            with TimeRecorder(conv_res, "layout"):
                batch_detections = self._run_inference(valid_images)

        layout_predictions: list[LayoutPrediction] = []
        valid_idx = 0

        for page in pages:
            if page._backend is None or not page._backend.is_valid():  # noqa: SLF001
                existing = page.predictions.layout or LayoutPrediction()
                layout_predictions.append(existing)
                continue

            detections = batch_detections[valid_idx]
            valid_idx += 1

            clusters: list[Cluster] = []
            for ix, det in enumerate(detections):
                cluster = Cluster(
                    id=ix,
                    label=det["label"],
                    confidence=det["confidence"],
                    bbox=BoundingBox(
                        l=det["l"],
                        t=det["t"],
                        r=det["r"],
                        b=det["b"],
                    ),
                    cells=[],
                )
                clusters.append(cluster)

            processed_clusters, processed_cells = LayoutPostprocessor(page, clusters, self.options).postprocess()

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    "Mean of empty slice|invalid value encountered in scalar divide",
                    RuntimeWarning,
                    "numpy",
                )
                conv_res.confidence.pages[page.page_no].layout_score = float(
                    np.mean([c.confidence for c in processed_clusters])
                )
                conv_res.confidence.pages[page.page_no].ocr_score = float(
                    np.mean([c.confidence for c in processed_cells if c.from_ocr])
                )

            prediction = LayoutPrediction(clusters=processed_clusters)
            layout_predictions.append(prediction)

        return layout_predictions
