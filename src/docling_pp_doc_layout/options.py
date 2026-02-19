"""Configuration model for the PP-DocLayout-V3 layout engine."""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal

from docling.datamodel.pipeline_options import LayoutOptions
from pydantic import ConfigDict, Field


class PPDocLayoutV3Options(LayoutOptions):
    """Options for the PP-DocLayout-V3 layout detection engine.

    Uses a HuggingFace-hosted PP-DocLayout-V3 model to detect document
    layout elements (text, tables, figures, headers, etc.) in page images.

    Attributes:
        model_name: HuggingFace model repository ID.
        confidence_threshold: Minimum confidence score for detections.
    """

    kind: ClassVar[Literal["ppdoclayout-v3"]] = "ppdoclayout-v3"

    model_name: Annotated[
        str,
        Field(description="HuggingFace model repository ID for PP-DocLayout-V3."),
    ] = "PaddlePaddle/PP-DocLayoutV3_safetensors"

    confidence_threshold: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description="Minimum confidence score to keep a detection.",
        ),
    ] = 0.5

    model_config = ConfigDict(extra="forbid")
