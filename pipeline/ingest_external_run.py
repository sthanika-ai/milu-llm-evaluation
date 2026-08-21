"""Ingest an lm-eval-harness output directory that was produced outside
pipeline/run.py (e.g. an ad-hoc CLI run during debugging) into the real
raw-output store / scorer / results DB / MLflow -- so it's traceable like any
other run, instead of living disconnected in a scratch directory.

Usage: python -m pipeline.ingest_external_run <model_id> <lm_eval_output_dir> \
           --dataset-revision <sha> [--run-timestamp <ISO8601-ish>]
"""
import argparse
import os
import re

from pipeline import mlflow_logging, registry, results_db, run as run_mod, scorer, store


def infer_run_timestamp(lm_eval_output_dir: str) -> str:
    results_dir = _find_results_dir(lm_eval_output_dir)
    for fname in os.listdir(results_dir):
        m = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d+)", fname)
        if m:
            return m.group(1)
    raise FileNotFoundError(f"Could not infer a run timestamp from filenames under {lm_eval_output_dir}")


def _find_results_dir(lm_eval_output_dir: str) -> str:
    for root, _, files in os.walk(lm_eval_output_dir):
        if any(f.startswith("results_") for f in files):
            return root
    raise FileNotFoundError(f"No results_*.json found under {lm_eval_output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id")
    parser.add_argument("lm_eval_output_dir")
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--run-timestamp", default=None)
    args = parser.parse_args()

    model_cfg = next(c for c in registry.load_all() if c["model_id"] == args.model_id)
    milu_commit = run_mod.get_milu_commit()
    run_timestamp = args.run_timestamp or infer_run_timestamp(args.lm_eval_output_dir)

    raw_items_path = store.normalize_run(model_cfg, args.lm_eval_output_dir, milu_commit, args.dataset_revision, run_timestamp)
    scores = scorer.score_run(raw_items_path)

    run_id = f"{args.model_id}_{run_timestamp}_external"
    conn = results_db.get_connection()
    results_db.insert_scored_run(conn, run_id, args.model_id, "milu", run_timestamp, scores)
    mlflow_logging.log_run(model_cfg, milu_commit, args.dataset_revision, run_timestamp, scores, run_id)

    acc = scores["overall"]["accuracy"]
    print(f"[ingest] {args.model_id} ({run_id}): accuracy={acc:.4f} n_items={scores['overall']['n_items']} "
          f"malformed={scores['overall']['n_malformed']}")
    return scores


if __name__ == "__main__":
    main()
