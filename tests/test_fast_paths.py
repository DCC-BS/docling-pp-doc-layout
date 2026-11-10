"""The fast paths (``fast_paths`` option) give exactly what the Hugging Face processor and docling's R-trees give."""

from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from docling.utils.layout_postprocessor import BoundingBoxSpatialIndex
from docling_core.types.doc import BoundingBox
from PIL import Image
from torchvision.transforms.v2 import functional as tvF  # noqa: N812
from transformers import PPDocLayoutV3ImageProcessor

from docling_pp_doc_layout import _index, _resize
from docling_pp_doc_layout.model import PPDocLayoutV3Model
from docling_pp_doc_layout.options import PPDocLayoutV3Options

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _tv_resize(chw: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return tvF.resize(chw.unsqueeze(0), [h, w], interpolation=tvF.InterpolationMode.BICUBIC, antialias=False)[0]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    ("in_hw", "out_hw"), [((792, 612), (800, 800)), ((1684, 1191), (800, 800)), ((37, 53), (80, 30))]
)
def test_resize_matches_torchvision_cpu_kernel(device, in_hw, out_hw):
    chw = torch.randint(0, 256, (3, *in_hw), dtype=torch.uint8, generator=torch.Generator().manual_seed(1))
    got = _resize.resize(chw.unsqueeze(0).to(device), *out_hw)[0].cpu()
    assert torch.equal(got, _tv_resize(chw, *out_hw))


@pytest.mark.parametrize("device", DEVICES)
def test_exact_on_gpu_checks_once_per_size(device):
    chw = torch.full((3, 90, 70), 255, dtype=torch.uint8)
    assert _resize.exact_on_gpu(chw, 40, 40, device)
    assert (90, 70, 40, 40) in _resize._checked


def _boxes(n: int, seed: int) -> list[BoundingBox]:
    rnd = random.Random(seed)  # noqa: S311
    out = []
    for _ in range(n):
        left, top = rnd.randint(0, 50), rnd.randint(0, 50)
        out.append(BoundingBox(l=left, t=top, r=left + rnd.randint(0, 20), b=top + rnd.randint(0, 20)))
    return out


def test_list_index_returns_the_rtree_ids():
    boxes = _boxes(60, 2)
    rtree, plain = BoundingBoxSpatialIndex(), _index.ListSpatialIndex()
    for i, box in enumerate(boxes):
        rtree.insert(i, box)
        plain.insert(i, box)
    for i in range(0, 60, 7):
        rtree.delete(i, boxes[i])
        plain.delete(i, boxes[i])
    # touching boxes count as intersecting in both
    queries = [*_boxes(40, 3), BoundingBox(l=10, t=10, r=10, b=10)]
    for q in queries:
        assert sorted(plain.intersection(q)) == sorted(rtree.intersection(q))


def _model(processor: PPDocLayoutV3ImageProcessor, *, fast: bool) -> PPDocLayoutV3Model:
    model = object.__new__(PPDocLayoutV3Model)
    model.options = PPDocLayoutV3Options(fast_paths=fast, confidence_threshold=0.3)
    model._device = "cpu"
    model._image_processor = processor
    model._id2label = dict(enumerate(["text", "table", "image", "doc_title", "reference"]))
    model._names = [model._id2label[i] for i in range(5)]
    model._std255 = torch.tensor(1.0 / processor.rescale_factor, dtype=torch.float32)
    return model


class _Net(torch.nn.Module):
    """Stands in for PP-DocLayout-V3: fixed random outputs; it records the pixel values both paths give it."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[torch.Tensor] = []

    def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        self.seen.append(pixel_values)
        g = torch.Generator().manual_seed(4)
        b, q = pixel_values.shape[0], 300
        boxes = torch.rand(b, q, 4, generator=g) * 0.4 + 0.1
        return SimpleNamespace(
            pred_boxes=boxes,
            logits=torch.randn(b, q, 5, generator=g) - 2.5,
            out_masks=torch.randn(b, q, 200, 200, generator=g),
            order_logits=torch.randn(b, q, q, generator=g),
        )


def test_fast_paths_give_the_processor_detections():
    processor = PPDocLayoutV3ImageProcessor()
    rng = np.random.default_rng(5)
    images = [Image.fromarray(rng.integers(0, 256, (792, 612, 3), dtype=np.uint8)) for _ in range(2)]
    results = {}
    for fast in (False, True):
        model = _model(processor, fast=fast)
        model._model = _Net()
        results[fast] = model._run_inference(images)
        pixels = model._model.seen[0]
        results[f"pixels-{fast}"] = pixels
    assert torch.equal(results["pixels-True"], results["pixels-False"])
    assert results[True] == results[False]
    assert any(results[True])  # the random outputs keep some detections
