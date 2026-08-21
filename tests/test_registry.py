"""Unit tests for pipeline/registry.py's config validation logic.

Uses a couple of the real, existing configs/models/*.yaml files as fixtures (read-only --
never modifies them) to confirm valid configs of each backend kind load cleanly, plus
hand-built temp configs to confirm each ValueError path actually fires.
"""
import os
import sys

import pytest
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from pipeline import registry  # noqa: E402

CONFIGS_DIR = os.path.join(REPO_ROOT, "configs", "models")


def _write_config(tmp_path, data, name="test.yaml"):
    path = tmp_path / name
    with open(path, "w") as f:
        yaml.safe_dump(data, f)
    return str(path)


def test_load_all_real_configs_without_raising():
    # configs/models/*.yaml are this project's real, pinned model registry -- every one of
    # them must satisfy registry.py's own schema, or a run would fail at load time. Deliberately
    # not asserting an exact/minimum count: a fresh clone of the published repo has ~17
    # curated configs (see configs/README.md's "Published vs. full local registry" section),
    # while this project's own full local working copy has 58 -- both must pass this check.
    configs = registry.load_all(CONFIGS_DIR)
    assert len(configs) >= 1  # nosec B101
    model_ids = {c["model_id"] for c in configs}
    # gemma-2-2b-it (the sanity gate) is expected in both the curated and full registry.
    assert "gemma-2-2b-it" in model_ids  # nosec B101


def test_local_backend_requires_checkpoint_and_revision(tmp_path):
    cfg = {
        "model_id": "test-local",
        "serving_backend": "hf",
        "apply_chat_template": True,
        "num_fewshot": 5,
        "tasks": ["milu"],
        # missing hf_checkpoint / revision
    }
    path = _write_config(tmp_path, cfg)
    with pytest.raises(ValueError, match="local"):
        registry.load_model_config(path)


def test_api_backend_requires_endpoint_and_access_date(tmp_path):
    cfg = {
        "model_id": "test-api",
        "serving_backend": "openai-chat-completions",
        "apply_chat_template": True,
        "num_fewshot": 0,
        "tasks": ["milu_English_api"],
        # missing api_endpoint / access_date
    }
    path = _write_config(tmp_path, cfg)
    with pytest.raises(ValueError, match="API"):
        registry.load_model_config(path)


def test_api_backend_rejects_loglikelihood_protocol(tmp_path):
    cfg = {
        "model_id": "test-api-loglik",
        "serving_backend": "openai-chat-completions",
        "apply_chat_template": True,
        "num_fewshot": 0,
        "tasks": ["milu_English_api"],
        "api_endpoint": "https://example.invalid/v1/chat/completions",
        "access_date": "2026-08-11",
        "protocol": "loglikelihood",
    }
    path = _write_config(tmp_path, cfg)
    with pytest.raises(ValueError, match="loglikelihood"):
        registry.load_model_config(path)


def test_local_backend_defaults_to_loglikelihood_protocol(tmp_path):
    cfg = {
        "model_id": "test-local-default-protocol",
        "serving_backend": "hf",
        "apply_chat_template": True,
        "num_fewshot": 5,
        "tasks": ["milu"],
        "hf_checkpoint": "org/model",
        "revision": "abc123",
    }
    path = _write_config(tmp_path, cfg)
    loaded = registry.load_model_config(path)
    assert loaded["protocol"] == "loglikelihood"  # nosec B101
