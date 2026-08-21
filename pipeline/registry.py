"""Model registry: load pinned model configs from configs/models/*.yaml."""
import glob
import os

import yaml

CONFIGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "models")

COMMON_REQUIRED_FIELDS = [
    "model_id",
    "serving_backend",
    "apply_chat_template",
    "num_fewshot",
    "tasks",
]

# Local (open-weight) backends must pin an exact checkpoint + revision hash.
# API backends have no HF checkpoint -- they're pinned by API endpoint plus access
# date instead, since API model weights aren't pinned by us and can change
# silently server-side.
LOCAL_BACKENDS = {"hf", "hf-causal-multimodal", "vllm"}
API_REQUIRED_FIELDS = ["api_endpoint", "access_date"]
LOCAL_REQUIRED_FIELDS = ["hf_checkpoint", "revision"]

VALID_PROTOCOLS = {"loglikelihood", "generate_until"}


def load_model_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)

    missing = [field for field in COMMON_REQUIRED_FIELDS if field not in cfg]
    if missing:
        raise ValueError(f"{path} is missing required fields: {missing}")

    is_local = cfg["serving_backend"] in LOCAL_BACKENDS
    backend_required = LOCAL_REQUIRED_FIELDS if is_local else API_REQUIRED_FIELDS
    missing_backend_fields = [f for f in backend_required if f not in cfg]
    if missing_backend_fields:
        kind = "local" if is_local else "API"
        raise ValueError(f"{path}: {kind} model ({cfg['serving_backend']}) missing required fields: {missing_backend_fields}")

    protocol = cfg.setdefault("protocol", "loglikelihood" if is_local else "generate_until")
    if protocol not in VALID_PROTOCOLS:
        raise ValueError(f"{path}: protocol must be one of {VALID_PROTOCOLS}, got {protocol!r}")
    if not is_local and protocol == "loglikelihood":
        raise ValueError(
            f"{path}: API backend {cfg['serving_backend']!r} cannot use protocol='loglikelihood' "
            "-- chat-completions APIs don't expose prompt logprobs (see openai_completions.py's "
            "own NotImplementedError). Use 'generate_until' with a *_api task."
        )

    cfg["_config_path"] = path
    return cfg


def load_all(configs_dir: str = CONFIGS_DIR) -> list:
    paths = sorted(glob.glob(os.path.join(configs_dir, "*.yaml")))
    if not paths:
        raise FileNotFoundError(f"No model configs found in {configs_dir}")
    return [load_model_config(p) for p in paths]
