"""Docling plugin entry point registering the PP-DocLayout-V3 layout engine."""

from __future__ import annotations

from typing import Any

from docling_pp_doc_layout.model import PPDocLayoutV3Model


def layout_engines() -> dict[str, Any]:
    """Return layout engine classes provided by this plugin."""
    return {"layout_engines": [PPDocLayoutV3Model]}
