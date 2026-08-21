"""Run logs: every run records its config and data-version hash to MLflow
(self-hosted, local file store under ./mlruns -- no external account needed).
"""
import os

import mlflow

MLRUNS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mlruns")
MLFLOW_DB_PATH = os.path.join(MLRUNS_DIR, "mlflow.db")


def log_run(model_cfg: dict, milu_commit: str, dataset_revision: str, run_timestamp: str, scores: dict, run_id: str,
            real_cost: dict = None):
    os.makedirs(MLRUNS_DIR, exist_ok=True)
    # mlflow's plain filesystem store is in maintenance mode as of mlflow 3.x;
    # a local sqlite backend is still fully self-hosted, no external account needed.
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    mlflow.set_experiment("milu")

    with mlflow.start_run(run_name=run_id):
        mlflow.log_params({
            "model_id": model_cfg["model_id"],
            "serving_backend": model_cfg.get("serving_backend"),
            "protocol": model_cfg.get("protocol", "loglikelihood"),
            "hf_checkpoint": model_cfg.get("hf_checkpoint"),
            "revision": model_cfg.get("revision"),
            "api_endpoint": model_cfg.get("api_endpoint"),
            "access_date": model_cfg.get("access_date"),
            "apply_chat_template": model_cfg["apply_chat_template"],
            "num_fewshot": model_cfg["num_fewshot"],
            "quantization": model_cfg.get("quantization", "none"),
            "dtype": model_cfg.get("dtype"),
            "milu_repo_commit": milu_commit,
            "hf_dataset_revision": dataset_revision,
            "run_timestamp": run_timestamp,
        })

        def log_score(prefix, s):
            if s["accuracy"] is not None:
                mlflow.log_metric(f"acc_{prefix}", s["accuracy"])
            mlflow.log_metric(f"n_items_{prefix}", s["n_items"])
            if s["malformed_rate"] is not None:
                mlflow.log_metric(f"malformed_rate_{prefix}", s["malformed_rate"])

        log_score("overall", scores["overall"])
        for lang, s in scores["by_language"].items():
            log_score(f"lang_{lang}", s)
        for dom, s in scores["by_domain"].items():
            safe_dom = dom.replace(" ", "_").replace("&", "and") if dom else "unknown"
            log_score(f"domain_{safe_dom}", s)

        if real_cost is not None:
            mlflow.log_metric("real_prompt_tokens", real_cost["prompt_tokens"])
            mlflow.log_metric("real_completion_tokens", real_cost["completion_tokens"])
            mlflow.log_metric("real_cost_usd", real_cost["cost_usd"])
