"""Ingest a custom-format JSONL generation output (produced by internal tooling not
included in this repository) into the SAME results layer every other model uses
(results_db.py + MLflow) -- mirrors run.py's run_model() orchestration (store ->
score -> results_db -> mlflow), just adapted to a pre-existing raw generation file
instead of invoking lm_eval as a subprocess.

Convention match with store.py's schema: truncated/no-extracted-answer items are
marked is_malformed (excluded from the accuracy denominator, tracked separately via
malformed_rate) -- same treatment store.py already gives API-model refusals/format
failures, not scored as wrong answers.

Can be re-run safely: results_db's insert_scored_run uses INSERT OR REPLACE keyed on
(run_id, breakdown_type, breakdown_key), so re-ingesting after later phases finish
(e.g. English done, then the remaining 10 languages) just overwrites the same run_id's
rows with the more-complete scores -- no duplicate/stale rows.
"""
import argparse
import glob
import json
import os

from pipeline import mlflow_logging, registry, results_db, run as run_module, scorer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_STORE_DIR = os.path.join(REPO_ROOT, "data_raw_outputs")


def normalize_thinkmode_jsonl(input_paths, model_cfg, milu_commit, dataset_revision, run_timestamp):
    model_id = model_cfg["model_id"]
    out_dir = os.path.join(RAW_STORE_DIR, model_id, run_timestamp)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "raw_items.jsonl")

    generation_config = model_cfg.get("generation_config", {})
    n_written = 0
    n_malformed = 0
    with open(out_path, "w") as out_f:
        for input_path in input_paths:
            with open(input_path) as in_f:
                for line in in_f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    pred = rec.get("pred")
                    # Truncated/unparseable generations are scored wrong (correct=False,
                    # since pred=None never equals the target letter), NOT excluded from
                    # the accuracy denominator via is_malformed -- unlike a genuine API
                    # refusal, a truncation here means the model ran out of its token
                    # budget mid-reasoning and never produced an answer at all, which is a
                    # real failure to answer the question, not a separate failure mode to
                    # carve out. Excluding these previously inflated accuracy (e.g. sarvam-m
                    # 85.97% excl. vs the honest 84.68%/81.98% incl.-truncation numbers).
                    is_malformed = False
                    letter_to_index = {"A": 0, "B": 1, "C": 2, "D": 3}

                    normalized = {
                        "model_id": model_id,
                        "hf_checkpoint": model_cfg.get("hf_checkpoint"),
                        "revision": model_cfg.get("revision"),
                        "api_endpoint": model_cfg.get("api_endpoint"),
                        "access_date": model_cfg.get("access_date"),
                        "benchmark": "milu",
                        "protocol": "generate_until",
                        "benchmark_version": {
                            "milu_repo_commit": milu_commit,
                            "hf_dataset_revision": dataset_revision,
                        },
                        "num_fewshot": model_cfg["num_fewshot"],
                        "apply_chat_template": model_cfg["apply_chat_template"],
                        "generation_config": generation_config,
                        "run_timestamp": run_timestamp,
                        "source_file": input_path,
                        "item_id": f"{rec['language']}:{rec['doc_index']}",
                        "language": rec["language"],
                        "domain": rec.get("domain"),
                        "subject": rec.get("subject"),
                        "prompt_hash": None,
                        "doc_hash": None,
                        "target_index": letter_to_index.get(rec["target"]),
                        "predicted_index": letter_to_index.get(pred),
                        "predicted_raw": pred,
                        "correct": bool(rec["correct"]),
                        "is_malformed": is_malformed,
                        "option_logprobs": None,
                    }
                    out_f.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                    n_written += 1
                    n_malformed += int(is_malformed)

    print(f"[ingest] wrote {n_written} normalized items to {out_path} "
          f"({n_malformed} malformed/truncated, excluded from accuracy denominator)")
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="sarvam-m-thinkmode-0shot-json")
    p.add_argument("--inputs", nargs="+", required=True, help="One or more custom-format JSONL run outputs")
    p.add_argument("--run-timestamp", required=True, help="Stable across re-ingests of the same logical run "
                                                            "(e.g. English-then-rest phases) so results_db rows "
                                                            "update in place rather than creating a new run_id")
    args = p.parse_args()

    model_cfg = registry.load_model_config(
        os.path.join(REPO_ROOT, "configs", "models", f"{args.model_id}.yaml")
    )
    milu_commit = run_module.get_milu_commit()
    dataset_revision = run_module.get_dataset_revision()

    raw_items_path = normalize_thinkmode_jsonl(args.inputs, model_cfg, milu_commit, dataset_revision, args.run_timestamp)
    scores = scorer.score_run(raw_items_path)

    run_id = f"{args.model_id}_{args.run_timestamp}"
    conn = results_db.get_connection()
    results_db.insert_scored_run(conn, run_id, args.model_id, "milu", args.run_timestamp, scores)
    mlflow_logging.log_run(model_cfg, milu_commit, dataset_revision, args.run_timestamp, scores, run_id)

    print(f"[ingest] run_id={run_id}")
    print(f"[ingest] overall: {scores['overall']}")
    for lang, s in scores["by_language"].items():
        print(f"[ingest]   {lang}: {s}")


if __name__ == "__main__":
    main()
