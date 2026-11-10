"""PP-DocLayout-V3 layout model for the docling standard pipeline.

Runs PaddlePaddle PP-DocLayout-V3 locally via HuggingFace ``transformers``
to detect document layout elements and returns ``LayoutPrediction`` objects
that docling merges with its standard-pipeline output.
"""

from __future__ import annotations

import functools
import logging
import threading
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from docling.datamodel.base_models import BoundingBox, Cluster, LayoutPrediction, Page
from docling.models.base_layout_model import BaseLayoutModel
from docling.utils.accelerator_utils import decide_device
from docling.utils.layout_postprocessor import LayoutPostprocessor
from docling.utils.profiling import TimeRecorder
from docling_core.types.doc import DocItemLabel
from torchvision.transforms.v2 import functional as tvF  # noqa: N812
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from transformers.image_utils import PILImageResampling

from docling_pp_doc_layout import _index, _resize, postprocess_hook
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


def _host(values: Any) -> list:  # noqa: ANN401 - a tensor or a list of tensors
    """Python values of a tensor (one host copy) or of a list of tensors."""
    if hasattr(values, "tolist"):
        return values.tolist()
    return [v.tolist() if hasattr(v, "tolist") else v for v in values]


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


def _doc_key(conv_res: ConversionResult) -> tuple[str, int]:
    """Key of a conversion for list detection's enumerator sequences across its pages.

    ``id(conv_res)`` alone is reused once a conversion is freed, so the next document could
    inherit the previous one's sequences; the document hash tells them apart.
    """
    return str(getattr(conv_res.input, "document_hash", "")), id(conv_res)


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
        self._heron_lock = threading.Lock()
        if options.list_detection != "off" and DOCLING_POSTPROCESSES_LAYOUT:
            postprocess_hook.install()
        self._list_index = options.fast_paths and DOCLING_POSTPROCESSES_LAYOUT and _index.install()
        logger.info("PP-DocLayout-V3 model loaded successfully")

    def _register_for_postprocessing(self, page: Page, conv_res: ConversionResult) -> None:
        """Mark the page for list detection and the list index, which run when docling post-processes it."""
        if self.options.list_detection != "off":
            postprocess_hook.register(page, _doc_key(conv_res), veto=self.options.list_detection == "heron")
        if getattr(self, "_list_index", False):
            _index.register(page)

    @functools.cached_property
    def _std255(self) -> torch.Tensor:
        """HF's normalisation divisor (std 1/rescale_factor = 255.0) as a 0-dim float32 tensor on the model's device."""
        return torch.tensor(1.0 / self._image_processor.rescale_factor, dtype=torch.float32).to(self._device)

    @functools.cached_property
    def _names(self) -> list[str]:
        """The model's class names by class id (see ``label_mapping.class_names``)."""
        return class_names({int(k): v for k, v in self._id2label.items()})

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
        fast = self._fast_paths_apply()
        pixel_values = self._fast_pixel_values(images) if fast else None
        if pixel_values is not None:
            inputs = {"pixel_values": pixel_values}
        else:
            inputs = self._image_processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        target_sizes = [img.size[::-1] for img in images]  # (height, width)
        if fast and isinstance(getattr(outputs, "pred_boxes", None), torch.Tensor):
            results = self._post_process_fast(outputs, target_sizes, self.options.confidence_threshold)
            return [self._detections_fast(result) for result in results]
        results = self._image_processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self.options.confidence_threshold,
        )

        return [self._detections_fast(result) for result in results]

    def _fast_paths_apply(self) -> bool:
        """Whether ``fast_paths`` is on and the image processor has the PP-DocLayout-V3 helpers the fast paths reuse."""
        proc_type = type(self._image_processor)
        return self.options.fast_paths and all(
            callable(getattr(proc_type, name, None)) for name in ("_get_order_seqs", "_extract_polygon_points_by_masks")
        )

    def _fast_pixel_values(self, images: list[Image.Image]) -> torch.Tensor | None:
        """The HF processor's pixel values without its per-call overhead (``fast_paths``).

        Same math as ``PPDocLayoutV3ImageProcessor``: torchvision's CPU uint8 bicubic resize
        (antialias off) to 800x800, then float32 / 255 -- the division runs on the GPU, on the
        uint8 tensor (4x less to copy than float32). On CUDA the resize itself runs on the GPU when
        :func:`_resize.exact_on_gpu` confirms, once per page size, that it gives the CPU kernel's pixels.
        None for images the fast path does not cover.
        """
        proc = self._image_processor
        size = proc.size
        if (
            not self._device.startswith(("cuda", "cpu"))  # other accelerators: float division may round differently
            or not getattr(proc, "do_resize", False)
            or int(getattr(proc, "resample", -1)) != int(PILImageResampling.BICUBIC)
            or not getattr(proc, "do_rescale", False)
            or any(getattr(proc, flag, None) for flag in ("do_pad", "do_center_crop"))
            or any(img.mode != "RGB" for img in images)
            or (
                getattr(proc, "do_normalize", False)
                and (list(proc.image_mean) != [0, 0, 0] or list(proc.image_std) != [1, 1, 1])
            )
        ):
            return None
        out_hw = [size["height"], size["width"]]
        if self._device.startswith("cuda"):
            hwcs = [torch.from_numpy(np.array(img)) for img in images]  # writable copy (no torch warning)
            if all(_resize.exact_on_gpu(c.permute(2, 0, 1), out_hw[0], out_hw[1], self._device) for c in hwcs):
                # upload the uint8 page as is (HWC), resize on the GPU
                batch = torch.stack([
                    _resize.resize(c.pin_memory().to(self._device, non_blocking=True).permute(2, 0, 1), *out_hw)
                    for c in hwcs
                ])  # fmt: skip
                return batch.to(torch.float32).div_(self._std255)
        resized = []
        for img in images:
            chw = torch.from_numpy(np.array(img)).permute(2, 0, 1)  # channels-last view of a writable copy
            resized.append(tvF.resize(chw.unsqueeze(0), out_hw, interpolation=tvF.InterpolationMode.BICUBIC,
                                      antialias=False)[0])  # fmt: skip
        batch = torch.stack(resized).contiguous()
        if self._device.startswith("cuda"):
            batch = batch.pin_memory().to(self._device, non_blocking=True)
        else:
            batch = batch.to(self._device)
        # HF: rescale fused into normalize with mean 0 and std 1/rescale_factor = 255.0 (float32)
        return batch.to(torch.float32).div_(self._std255)

    def _post_process_fast(self, outputs: Any, target_sizes: list[tuple[int, int]], threshold: float) -> list[dict]:  # noqa: ANN401
        """``post_process_object_detection`` with the same math, kept detections only.

        HF gathers, sigmoids and thresholds the masks of all 300 queries per image (300x200x200
        floats each) and then keeps the few above the threshold; here the kept queries are chosen
        first and only their masks are processed, and each image needs one host copy.
        """
        proc = self._image_processor
        boxes, logits, masks = outputs.pred_boxes, outputs.logits, outputs.out_masks
        order_seqs = proc._get_order_seqs(outputs.order_logits)  # noqa: SLF001
        box_centers, box_dims = torch.split(boxes, 2, dim=-1)
        boxes = torch.cat([box_centers - 0.5 * box_dims, box_centers + 0.5 * box_dims], dim=-1)
        img_height, img_width = torch.as_tensor(target_sizes).unbind(1)
        scale_factor = torch.stack([img_width, img_height, img_width, img_height], dim=1).to(boxes.device)
        boxes = boxes * scale_factor[:, None, :]
        num_top_queries, num_classes = logits.shape[1], logits.shape[2]
        scores, index = torch.topk(torch.nn.functional.sigmoid(logits).flatten(1), num_top_queries, dim=-1)
        labels = index % num_classes
        index = index // num_classes
        boxes = boxes.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, boxes.shape[-1]))
        order_seqs = order_seqs.gather(dim=1, index=index)
        results = []
        for b, target_size in enumerate(target_sizes):
            keep = scores[b] >= threshold
            order_seq, indices = torch.sort(order_seqs[b][keep])
            queries = index[b][keep][indices]
            mask = (masks[b, queries].sigmoid() > threshold).int()
            box = boxes[b][keep][indices]
            box_np, mask_np = box.cpu().numpy(), mask.cpu().numpy()
            results.append({
                "scores": scores[b][keep][indices],
                "labels": labels[b][keep][indices],
                "boxes": box,
                "polygon_points": proc._extract_polygon_points_by_masks(  # noqa: SLF001
                    box_np, mask_np, [proc.size["width"] / target_size[1], proc.size["height"] / target_size[0]]
                ),
                "order_seq": order_seq,
            })
        return results

    def _detections_fast(self, result: dict) -> list[dict]:
        """The detections of one post-processed image, read with one host copy per field."""
        names = self._names
        scores, labels, boxes = (_host(result[key]) for key in ("scores", "labels", "boxes"))
        polys = result.get("polygons") or result.get("polygon_points") or [None] * len(scores)
        detections: list[dict] = []
        for score, label_ix, box, poly in zip(scores, labels, boxes, polys, strict=True):
            raw_label = names[label_ix] if 0 <= label_ix < len(names) else "text"
            if poly is not None and len(poly) > 0:
                if isinstance(poly[0], int | float):
                    xs, ys = poly[0::2], poly[1::2]
                else:
                    xs = [pt[0] for pt in poly]
                    ys = [pt[1] for pt in poly]
                x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)
            else:
                x_min, y_min, x_max, y_max = box
            detections.append({"label": LABEL_MAP.get(raw_label, DocItemLabel.TEXT), "confidence": score,
                               "l": x_min, "t": y_min, "r": x_max, "b": y_max, "raw": raw_label})  # fmt: skip
        return [{k: v for k, v in d.items() if k != "raw"} for d in _drop_reference_blocks(detections)]

    def _heron_clusters(self, conv_res: ConversionResult, page: Page) -> list[Cluster]:
        """Raw clusters of docling's own layout model (Heron) for one page, for its list items."""
        with self._heron_lock:  # pages of concurrent conversions must not each build a model
            if self._heron_model is None:
                self._heron_model = self._load_heron()
        predictions = list(self._heron_model.predict_layout(conv_res, [page]))
        return list(predictions[0].clusters) if predictions else []

    def _load_heron(self) -> BaseLayoutModel:
        """Docling's own layout model (Heron), created on first use."""
        # Lazy imports on purpose: which layout model exists depends on the installed docling version.
        try:
            from docling.datamodel.pipeline_options import LayoutObjectDetectionOptions  # noqa: PLC0415
            from docling.models.stages.layout.layout_object_detection_model import (  # noqa: PLC0415
                LayoutObjectDetectionModel,
            )

            return LayoutObjectDetectionModel(
                artifacts_path=self.artifacts_path,
                accelerator_options=self.accelerator_options,
                options=LayoutObjectDetectionOptions(),
            )
        except ImportError:  # docling before LayoutObjectDetectionModel
            from docling.datamodel.pipeline_options import LayoutOptions  # noqa: PLC0415
            from docling.models.stages.layout.layout_model import LayoutModel  # noqa: PLC0415

            return LayoutModel(
                artifacts_path=self.artifacts_path,
                accelerator_options=self.accelerator_options,
                options=LayoutOptions(),
            )

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
                self._register_for_postprocessing(page, conv_res)
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
