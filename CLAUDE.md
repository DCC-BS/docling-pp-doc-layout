# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`docling-pp-doc-layout` is a Docling plugin that integrates PaddlePaddle's PP-DocLayout-V3 model for document layout detection. It detects layout elements (text, tables, figures, headers, etc.) in page images and feeds predictions back into Docling's standard pipeline.

## Commands

This project uses `uv` as the package manager and `make` to orchestrate common tasks.

```bash
make install       # Install dependencies + pre-commit hooks
make check         # Run all quality checks (lint, format, type-check)
make lint          # Run ruff linter with auto-fix
make format        # Run ruff formatter
make type-check    # Run ty type checker (src/ only)
make test          # Run pytest with terminal coverage report
make test-xml      # Run pytest and produce XML coverage report
make build         # Build distribution packages
make publish       # Publish to PyPI
```

To run a single test file or test function:
```bash
uv run pytest tests/test_model.py
uv run pytest tests/test_model.py::test_function_name
```

## Architecture

The plugin has four core modules under `src/docling_pp_doc_layout/`:

- **`plugin.py`** — Entry point. Exposes `layout_engines()` which returns a dict mapping the engine name to `PPDocLayoutV3Model`. Docling discovers plugins via this function.

- **`options.py`** — `PPDocLayoutV3Options` (Pydantic model). Configures the model name, confidence threshold (0.0–1.0, default 0.5), and inherits cluster options from Docling. `kind = "ppdoclayout-v3"` identifies this engine type.

- **`model.py`** — `PPDocLayoutV3Model`, the core detection class. Inherits from `BaseLayoutModel`. Key flow: `predict_layout()` extracts PIL images from pages → `_run_inference()` runs HuggingFace transformers batch inference → maps raw labels via `label_mapping.py` → applies `LayoutPostprocessor` → returns `LayoutPrediction` objects to Docling.

- **`label_mapping.py`** — Maps 21 raw PP-DocLayout-V3 class names to `DocItemLabel` values. **Critical constraint:** every label mapped here must exist in `LayoutPostprocessor.CONFIDENCE_THRESHOLDS`. The test suite enforces this.

## Testing Notes

- `tests/conftest.py` mocks the `transformers` module when it is not installed, enabling pure-logic unit tests without GPU/model access.
- The `transformers_available` fixture skips tests that require actual model inference.
- 80% coverage minimum is enforced; CI reports to Codecov.

## Code Style

- Line length: 120 characters
- String quotes: double
- Docstring convention: Google style
- Type checking: `ty` (not mypy)
- Ruff enforces a large ruleset (ANN, I, C90, BLE, TRY, S, SIM, PT, and more) — run `make check` before committing.
