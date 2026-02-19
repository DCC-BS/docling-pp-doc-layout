"""Pytest configuration for docling_pp_doc_layout tests.

Mocks ``transformers`` when it is not importable (version conflict with
docling) so that pure-logic tests can still run.
"""

import sys
from unittest.mock import MagicMock

import pytest


def _ensure_mock(name: str) -> bool:
    """Insert a ``MagicMock`` into *sys.modules* if *name* cannot be imported.

    Returns ``True`` if a mock was injected.
    """
    if name in sys.modules:
        return isinstance(sys.modules[name], MagicMock)
    try:
        __import__(name)
    except ImportError:
        sys.modules[name] = MagicMock()
        return True
    else:
        return False


_transformers_mocked = _ensure_mock("transformers")


@pytest.fixture
def transformers_available():
    """Skip the test when transformers is mocked rather than real."""
    if _transformers_mocked:
        pytest.skip("transformers not installed (mocked)")
