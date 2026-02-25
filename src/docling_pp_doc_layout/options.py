"""Configuration model for the PP-DocLayout-V3 layout engine."""

from __future__ import annotations

import os
from typing import Annotated, ClassVar, Literal

from docling.datamodel.pipeline_options import LayoutOptions
from pydantic import ConfigDict, Field


def _parse_bool(value: str) -> bool:
    """Parse a string environment variable value as a boolean.

    Args:
        value: The string to parse.  Case-insensitive ``"true"``, ``"1"``,
            and ``"yes"`` are truthy; everything else is falsy.

    Returns:
        ``True`` if *value* is a recognised truthy string, ``False`` otherwise.
    """
    return value.lower() in ("true", "1", "yes")


class PPDocLayoutV3Options(LayoutOptions):
    """Options for the PP-DocLayout-V3 layout detection engine.

    Uses a HuggingFace-hosted PP-DocLayout-V3 model to detect document
    layout elements (text, tables, figures, headers, etc.) in page images.

    All options fall back to environment variables when not set explicitly,
    allowing configuration without code changes (e.g. in Docker / Compose
    deployments).

    Attributes:
        model_name: HuggingFace model repository ID.
            Falls back to the ``PP_DOC_LAYOUT_MODEL_NAME`` env var.
        confidence_threshold: Minimum confidence score for detections.
            Falls back to the ``PP_DOC_LAYOUT_CONFIDENCE_THRESHOLD`` env var.
        batch_size: Number of pages per inference batch.
            Falls back to the ``PP_DOC_LAYOUT_BATCH_SIZE`` env var.
        create_orphan_clusters: Create clusters for orphaned elements.
            Falls back to the ``PP_DOC_LAYOUT_CREATE_ORPHAN_CLUSTERS`` env var.
        keep_empty_clusters: Retain empty clusters in results.
            Falls back to the ``PP_DOC_LAYOUT_KEEP_EMPTY_CLUSTERS`` env var.
        skip_cell_assignment: Skip table-cell assignment during layout analysis.
            Falls back to the ``PP_DOC_LAYOUT_SKIP_CELL_ASSIGNMENT`` env var.
    """

    kind: ClassVar[Literal["ppdoclayout-v3"]] = "ppdoclayout-v3"

    model_name: Annotated[
        str,
        Field(description="HuggingFace model repository ID for PP-DocLayout-V3."),
    ] = Field(
        default_factory=lambda: os.environ.get(
            "PP_DOC_LAYOUT_MODEL_NAME",
            "PaddlePaddle/PP-DocLayoutV3_safetensors",
        )
    )

    confidence_threshold: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description="Minimum confidence score to keep a detection.",
        ),
    ] = Field(default_factory=lambda: float(os.environ.get("PP_DOC_LAYOUT_CONFIDENCE_THRESHOLD", "0.5")))

    batch_size: Annotated[
        int,
        Field(
            gt=0,
            description="Batch size for layout inference.",
        ),
    ] = Field(default_factory=lambda: int(os.environ.get("PP_DOC_LAYOUT_BATCH_SIZE", "8")))

    # Override inherited boolean fields to add environment-variable support.
    create_orphan_clusters: Annotated[
        bool,
        Field(
            description=(
                "Create clusters for orphaned elements not assigned to any structure. "
                "Falls back to PP_DOC_LAYOUT_CREATE_ORPHAN_CLUSTERS env var."
            )
        ),
    ] = Field(default_factory=lambda: _parse_bool(os.environ.get("PP_DOC_LAYOUT_CREATE_ORPHAN_CLUSTERS", "true")))

    keep_empty_clusters: Annotated[
        bool,
        Field(
            description=(
                "Retain empty clusters in layout analysis results. "
                "Falls back to PP_DOC_LAYOUT_KEEP_EMPTY_CLUSTERS env var."
            )
        ),
    ] = Field(default_factory=lambda: _parse_bool(os.environ.get("PP_DOC_LAYOUT_KEEP_EMPTY_CLUSTERS", "false")))

    skip_cell_assignment: Annotated[
        bool,
        Field(
            description=(
                "Skip assignment of cells to table structures during layout analysis. "
                "Falls back to PP_DOC_LAYOUT_SKIP_CELL_ASSIGNMENT env var."
            )
        ),
    ] = Field(default_factory=lambda: _parse_bool(os.environ.get("PP_DOC_LAYOUT_SKIP_CELL_ASSIGNMENT", "false")))

    model_config = ConfigDict(extra="forbid")
