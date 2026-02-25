"""Tests for PPDocLayoutV3Options."""

from __future__ import annotations

import os
from typing import ClassVar
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from docling_pp_doc_layout.options import PPDocLayoutV3Options


class TestKind:
    def test_kind_value(self):
        assert PPDocLayoutV3Options.kind == "ppdoclayout-v3"

    def test_kind_is_class_var(self):
        assert "kind" not in PPDocLayoutV3Options.model_fields


class TestDefaults:
    def test_default_model_name(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PP_DOC_LAYOUT_MODEL_NAME", None)
            opts = PPDocLayoutV3Options()
            assert opts.model_name == "PaddlePaddle/PP-DocLayoutV3_safetensors"

    def test_default_confidence_threshold(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PP_DOC_LAYOUT_CONFIDENCE_THRESHOLD", None)
            opts = PPDocLayoutV3Options()
            assert opts.confidence_threshold == 0.5

    def test_default_batch_size(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PP_DOC_LAYOUT_BATCH_SIZE", None)
            opts = PPDocLayoutV3Options()
            assert opts.batch_size == 8

    def test_default_create_orphan_clusters(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PP_DOC_LAYOUT_CREATE_ORPHAN_CLUSTERS", None)
            opts = PPDocLayoutV3Options()
            assert opts.create_orphan_clusters is True

    def test_default_skip_cell_assignment(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PP_DOC_LAYOUT_SKIP_CELL_ASSIGNMENT", None)
            opts = PPDocLayoutV3Options()
            assert opts.skip_cell_assignment is False

    def test_default_keep_empty_clusters(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PP_DOC_LAYOUT_KEEP_EMPTY_CLUSTERS", None)
            opts = PPDocLayoutV3Options()
            assert opts.keep_empty_clusters is False


class TestEnvVars:
    def test_env_model_name(self):
        with patch.dict(os.environ, {"PP_DOC_LAYOUT_MODEL_NAME": "my-org/my-layout-model"}):
            opts = PPDocLayoutV3Options()
            assert opts.model_name == "my-org/my-layout-model"

    def test_env_confidence_threshold(self):
        with patch.dict(os.environ, {"PP_DOC_LAYOUT_CONFIDENCE_THRESHOLD": "0.7"}):
            opts = PPDocLayoutV3Options()
            assert opts.confidence_threshold == pytest.approx(0.7)

    def test_env_batch_size(self):
        with patch.dict(os.environ, {"PP_DOC_LAYOUT_BATCH_SIZE": "16"}):
            opts = PPDocLayoutV3Options()
            assert opts.batch_size == 16

    def test_env_create_orphan_clusters_false(self):
        with patch.dict(os.environ, {"PP_DOC_LAYOUT_CREATE_ORPHAN_CLUSTERS": "false"}):
            opts = PPDocLayoutV3Options()
            assert opts.create_orphan_clusters is False

    def test_env_create_orphan_clusters_true(self):
        with patch.dict(os.environ, {"PP_DOC_LAYOUT_CREATE_ORPHAN_CLUSTERS": "true"}):
            opts = PPDocLayoutV3Options()
            assert opts.create_orphan_clusters is True

    def test_env_create_orphan_clusters_one(self):
        with patch.dict(os.environ, {"PP_DOC_LAYOUT_CREATE_ORPHAN_CLUSTERS": "1"}):
            opts = PPDocLayoutV3Options()
            assert opts.create_orphan_clusters is True

    def test_env_create_orphan_clusters_yes(self):
        with patch.dict(os.environ, {"PP_DOC_LAYOUT_CREATE_ORPHAN_CLUSTERS": "yes"}):
            opts = PPDocLayoutV3Options()
            assert opts.create_orphan_clusters is True

    def test_env_create_orphan_clusters_case_insensitive(self):
        with patch.dict(os.environ, {"PP_DOC_LAYOUT_CREATE_ORPHAN_CLUSTERS": "TRUE"}):
            opts = PPDocLayoutV3Options()
            assert opts.create_orphan_clusters is True

    def test_env_skip_cell_assignment_true(self):
        with patch.dict(os.environ, {"PP_DOC_LAYOUT_SKIP_CELL_ASSIGNMENT": "true"}):
            opts = PPDocLayoutV3Options()
            assert opts.skip_cell_assignment is True

    def test_env_keep_empty_clusters_true(self):
        with patch.dict(os.environ, {"PP_DOC_LAYOUT_KEEP_EMPTY_CLUSTERS": "true"}):
            opts = PPDocLayoutV3Options()
            assert opts.keep_empty_clusters is True

    def test_env_var_overridden_by_explicit_arg_str(self):
        with patch.dict(os.environ, {"PP_DOC_LAYOUT_MODEL_NAME": "env-org/env-model"}):
            opts = PPDocLayoutV3Options(model_name="explicit/model")
            assert opts.model_name == "explicit/model"

    def test_env_var_overridden_by_explicit_arg_float(self):
        with patch.dict(os.environ, {"PP_DOC_LAYOUT_CONFIDENCE_THRESHOLD": "0.9"}):
            opts = PPDocLayoutV3Options(confidence_threshold=0.1)
            assert opts.confidence_threshold == pytest.approx(0.1)

    def test_env_var_overridden_by_explicit_arg_int(self):
        with patch.dict(os.environ, {"PP_DOC_LAYOUT_BATCH_SIZE": "32"}):
            opts = PPDocLayoutV3Options(batch_size=1)
            assert opts.batch_size == 1

    def test_env_var_overridden_by_explicit_arg_bool(self):
        with patch.dict(os.environ, {"PP_DOC_LAYOUT_CREATE_ORPHAN_CLUSTERS": "false"}):
            opts = PPDocLayoutV3Options(create_orphan_clusters=True)
            assert opts.create_orphan_clusters is True


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
            PPDocLayoutV3Options(unknown_field="oops")


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
