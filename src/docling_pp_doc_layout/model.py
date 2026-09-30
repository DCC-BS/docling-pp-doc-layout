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

from docling_pp_doc_layout import postprocess_hook
from docling_pp_doc_layout.label_mapping import LABEL_MAP, class_names
from docling_pp_doc_layout.lists import adopt_heron
from docling_pp_doc_layout.options import PPDocLayoutV3Options

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.document import ConversionResult
    from docling.datamodel.pipeline_options import BaseLayoutOptions
    from PIL import Image

logger = logging.getLogger(__name__)

# docling >= 2.116 runs the LayoutPostprocessor (cell assignment, empty-cluster removal,
# layout score) as its own pipeline stage *after* OCR, and only OCRs inside layout
# regions. Layout models must therefore return the raw detections: post-processing them
# here, before OCR, drops every region without PDF text, i.e. whole scanned pages.
# Older docling expects the layout model to post-process itself.
DOCLING_POSTPROCESSES_LAYOUT = hasattr(BaseLayoutModel, "requires_layout_postprocessing")


REFERENCE_ENTRY_INSIDE = 0.8  # share of an entry inside a bibliography block for it to belong there
MIN_REFERENCE_ENTRIES = 2


def _drop_reference_blocks(detections: list[dict]) -> list[dict]:
    """Drop a bibliography block that holds two or more reference entries.

    PP-DocLayout-V3 returns both the block (``reference``) and its entries
    (``reference_content``); docling would keep the larger block and lose the entries.
    """
    entries = [d for d in detections if d["raw"] == "reference_content"]

    def holds(block: dict, entry: dict) -> bool:
        w = max(0.0, min(block["r"], entry["r"]) - max(block["l"], entry["l"]))
        h = max(0.0, min(block["b"], entry["b"]) - max(block["t"], entry["t"]))
        area = max((entry["r"] - entry["l"]) * (entry["b"] - entry["t"]), 1e-6)
        return w * h / area >= REFERENCE_ENTRY_INSIDE

    return [
        d for d in detections if d["raw"] != "reference" or sum(holds(d, e) for e in entries) < MIN_REFERENCE_ENTRIES
    ]


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
        self._heron_model: BaseLayoutModel | None = None
        if options.list_detection != "off" and DOCLING_POSTPROCESSES_LAYOUT:
            postprocess_hook.install()
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

        names = class_names({int(k): v for k, v in self._id2label.items()})
        batch_detections: list[list[dict]] = []
        for result in results:
            detections: list[dict] = []

            polys = result.get("polygons") or result.get("polygon_points")
            if polys is None:
                polys = [None] * len(result["scores"])

            for score, label_id, box, poly in zip(
                result["scores"],
                result["labels"],
                result["boxes"],
                polys,
                strict=True,
            ):
                label_ix = int(label_id.item())
                raw_label = names[label_ix] if 0 <= label_ix < len(names) else "text"
                doc_label = LABEL_MAP.get(raw_label, DocItemLabel.TEXT)

                if poly is not None and len(poly) > 0:
                    # Flatten or handle nested points to extract min/max
                    if isinstance(poly[0], int | float):
                        xs = poly[0::2]
                        ys = poly[1::2]
                    else:
                        xs = [pt[0] for pt in poly]
                        ys = [pt[1] for pt in poly]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                else:
                    x_min, y_min, x_max, y_max = box.tolist()

                detections.append({
                    "label": doc_label,
                    "confidence": score.item(),
                    "l": x_min,
                    "t": y_min,
                    "r": x_max,
                    "b": y_max,
                    "raw": raw_label,
                })
            batch_detections.append([
                {k: v for k, v in d.items() if k != "raw"} for d in _drop_reference_blocks(detections)
            ])

        return batch_detections

    def _heron_clusters(self, conv_res: ConversionResult, page: Page) -> list[Cluster]:
        """Raw clusters of docling's own layout model (Heron) for one page, for its list items."""
        if self._heron_model is None:
            try:
                from docling.datamodel.pipeline_options import LayoutObjectDetectionOptions
                from docling.models.stages.layout.layout_object_detection_model import (
                    LayoutObjectDetectionModel,
                )

                self._heron_model = LayoutObjectDetectionModel(
                    artifacts_path=self.artifacts_path,
                    accelerator_options=self.accelerator_options,
                    options=LayoutObjectDetectionOptions(),
                )
            except ImportError:  # docling before LayoutObjectDetectionModel
                from docling.datamodel.pipeline_options import LayoutOptions
                from docling.models.stages.layout.layout_model import LayoutModel

                self._heron_model = LayoutModel(
                    artifacts_path=self.artifacts_path,
                    accelerator_options=self.accelerator_options,
                    options=LayoutOptions(),
                )
        predictions = list(self._heron_model.predict_layout(conv_res, [page]))
        return list(predictions[0].clusters) if predictions else []

    @staticmethod
    def _extract_valid_pages(
        pages: Sequence[Page],
    ) -> tuple[list[Image.Image], list[bool]]:
        """Extract valid images and page validity flags from a sequence of pages."""
        valid_images: list[Image.Image] = []
        is_page_valid: list[bool] = []

        for page in pages:
            if page._backend is None or not page._backend.is_valid():  # noqa: SLF001
                is_page_valid.append(False)
                continue
            if page.size is None:
                is_page_valid.append(False)
                continue
            page_image = page.get_image(scale=1.0)
            if page_image is None:
                is_page_valid.append(False)
                continue

            valid_images.append(page_image)
            is_page_valid.append(True)

        return valid_images, is_page_valid

    def predict_layout(
        self,
        conv_res: ConversionResult,
        pages: Sequence[Page],
    ) -> Sequence[LayoutPrediction]:
        """Detect layout regions for a batch of document pages."""
        pages = list(pages)
        valid_images, is_page_valid = self._extract_valid_pages(pages)

        batch_detections: list[list[dict]] = []
        if valid_images:
            with TimeRecorder(conv_res, "layout"):
                bs = self.options.batch_size
                for i in range(0, len(valid_images), bs):
                    batch = valid_images[i : i + bs]
                    batch_detections.extend(self._run_inference(batch))

        layout_predictions: list[LayoutPrediction] = []
        valid_idx = 0

        for idx, page in enumerate(pages):
            if not is_page_valid[idx]:
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

            if DOCLING_POSTPROCESSES_LAYOUT:
                if self.options.list_detection == "heron":
                    clusters = adopt_heron(clusters, self._heron_clusters(conv_res, page))
                if self.options.list_detection != "off":
                    postprocess_hook.register(page, id(conv_res), veto=self.options.list_detection == "heron")
                layout_predictions.append(LayoutPrediction(clusters=clusters))
                continue

            postprocess_result = LayoutPostprocessor(page, clusters, self.options).postprocess()
            if isinstance(postprocess_result, tuple):
                processed_clusters, processed_cells = postprocess_result
            else:
                processed_clusters = postprocess_result
                processed_cells = [c for c in page.cells if getattr(c, "from_ocr", False)]

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
                    np.mean([c.confidence for c in processed_cells if getattr(c, "from_ocr", False)])
                )

            prediction = LayoutPrediction(clusters=processed_clusters)
            layout_predictions.append(prediction)

        return layout_predictions
