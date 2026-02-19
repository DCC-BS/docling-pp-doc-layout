"""Tests for PPDocLayoutV3Options."""

from __future__ import annotations

from typing import ClassVar

import pytest
from pydantic import ValidationError

from docling_pp_doc_layout.options import PPDocLayoutV3Options


class TestKind:
    def test_kind_value(self):
        assert PPDocLayoutV3Options.kind == "ppdoclayout-v3"

    def test_kind_is_class_var(self):
        opts = PPDocLayoutV3Options()
        assert "kind" not in opts.model_fields


class TestDefaults:
    def test_default_model_name(self):
        opts = PPDocLayoutV3Options()
        assert opts.model_name == "PaddlePaddle/PP-DocLayoutV3_safetensors"

    def test_default_confidence_threshold(self):
        opts = PPDocLayoutV3Options()
        assert opts.confidence_threshold == 0.5

    def test_default_create_orphan_clusters(self):
        opts = PPDocLayoutV3Options()
        assert opts.create_orphan_clusters is True

    def test_default_skip_cell_assignment(self):
        opts = PPDocLayoutV3Options()
        assert opts.skip_cell_assignment is False

    def test_default_keep_empty_clusters(self):
        opts = PPDocLayoutV3Options()
        assert opts.keep_empty_clusters is False


class TestCustomValues:
    def test_custom_model_name(self):
        opts = PPDocLayoutV3Options(model_name="my/model")
        assert opts.model_name == "my/model"

    def test_custom_confidence_threshold(self):
        opts = PPDocLayoutV3Options(confidence_threshold=0.3)
        assert opts.confidence_threshold == 0.3

    def test_custom_create_orphan_clusters_false(self):
        opts = PPDocLayoutV3Options(create_orphan_clusters=False)
        assert opts.create_orphan_clusters is False

    def test_custom_skip_cell_assignment_true(self):
        opts = PPDocLayoutV3Options(skip_cell_assignment=True)
        assert opts.skip_cell_assignment is True


class TestValidation:
    def test_confidence_below_zero_rejected(self):
        with pytest.raises(ValidationError):
            PPDocLayoutV3Options(confidence_threshold=-0.1)

    def test_confidence_above_one_rejected(self):
        with pytest.raises(ValidationError):
            PPDocLayoutV3Options(confidence_threshold=1.1)

    def test_confidence_boundary_zero(self):
        opts = PPDocLayoutV3Options(confidence_threshold=0.0)
        assert opts.confidence_threshold == 0.0

    def test_confidence_boundary_one(self):
        opts = PPDocLayoutV3Options(confidence_threshold=1.0)
        assert opts.confidence_threshold == 1.0

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            PPDocLayoutV3Options()


class TestPostprocessorCompatibility:
    """Verify that PPDocLayoutV3Options exposes every attribute that
    ``LayoutPostprocessor`` reads from its ``options`` argument.
    """

    REQUIRED_ATTRS: ClassVar[list[str]] = [
        "create_orphan_clusters",
        "skip_cell_assignment",
        "keep_empty_clusters",
    ]

    def test_has_all_postprocessor_attrs(self):
        opts = PPDocLayoutV3Options()
        for attr in self.REQUIRED_ATTRS:
            assert hasattr(opts, attr), (
                f"PPDocLayoutV3Options is missing '{attr}' -- LayoutPostprocessor will raise AttributeError"
            )

    def test_postprocessor_attrs_have_correct_types(self):
        opts = PPDocLayoutV3Options()
        for attr in self.REQUIRED_ATTRS:
            assert isinstance(getattr(opts, attr), bool)
