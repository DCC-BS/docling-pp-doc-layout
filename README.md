# docling-pp-doc-layout

A Docling plugin for PaddlePaddle PP-DocLayout-V3 model document layout detection.

---

<p align="center">
  <a href="https://github.com/DCC-BS/docling-pp-doc-layout">GitHub</a>
  &nbsp;|&nbsp;
  <a href="https://pypi.org/project/docling-pp-doc-layout/">PyPI</a>
</p>

---

[![PyPI version](https://img.shields.io/pypi/v/docling-pp-doc-layout.svg)](https://pypi.org/project/docling-pp-doc-layout/)
[![Python versions](https://img.shields.io/pypi/pyversions/docling-pp-doc-layout.svg)](https://pypi.org/project/docling-pp-doc-layout/)
[![License](https://img.shields.io/github/license/DCC-BS/docling-pp-doc-layout)](https://github.com/DCC-BS/docling-pp-doc-layout/blob/main/LICENSE)
[![CI](https://github.com/DCC-BS/docling-pp-doc-layout/actions/workflows/main.yml/badge.svg)](https://github.com/DCC-BS/docling-pp-doc-layout/actions/workflows/main.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Coverage](https://codecov.io/gh/DCC-BS/docling-pp-doc-layout/graph/badge.svg)](https://codecov.io/gh/DCC-BS/docling-pp-doc-layout)


## Overview

> TODO: Add a short paragraph describing what this package does and who it is for.

## Requirements

- Python 3.13+

## Installation

```bash
# with uv (recommended)
uv add docling-pp-doc-layout

# with pip
pip install docling-pp-doc-layout
```

## Usage

```python
from docling_pp_doc_layout import greet

result = greet("World")
print(result)  # Hello, World!
```

> TODO: Replace the example above with real usage examples for your package.

## API Reference

### `greet(name: str) -> str`

Returns a greeting string.

| Parameter | Type  | Description              |
|-----------|-------|--------------------------|
| `name`    | `str` | The name to greet.       |

**Returns:** `str` – Greeting message.

**Example:**

```python
from docling_pp_doc_layout import greet

greet("Alice")  # "Hello, Alice!"
```

## Development

### Setup

```bash
git clone https://github.com/DCC-BS/docling-pp-doc-layout.git
cd docling-pp-doc-layout
make install
```

### Available commands

```
make install     Install dependencies and pre-commit hooks
make check       Run all quality checks (ruff lint, format, ty type check)
make test        Run tests with coverage report
make build       Build distribution packages
make publish     Publish to PyPI
```

### Running tests

```bash
make test
```

Tests are in `tests/` and use [pytest](https://pytest.org).
Coverage reports are generated at `coverage.xml` and printed to the terminal.

### Code quality

This project uses:

- **[ruff](https://github.com/astral-sh/ruff)** – linting and formatting
- **[ty](https://github.com/astral-sh/ty)** – type checking
- **[pre-commit](https://pre-commit.com/)** – pre-commit hooks

Run all checks:

```bash
make check
```

### Releasing

Releases are published to PyPI automatically.
Update the version in `pyproject.toml`, then trigger the **Publish** workflow from GitHub Actions:

```
GitHub → Actions → Publish to PyPI → Run workflow
```

The workflow tags the commit, builds the package, and publishes to PyPI via trusted publishing.

## License

[MIT](LICENSE) © Yanick Schraner
