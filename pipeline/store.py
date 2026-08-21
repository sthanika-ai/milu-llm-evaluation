"""Raw-output store: normalize lm-eval-harness --log_samples output into our schema.

Never welds scoring to inference: this just re-shapes lm-eval's per-item verbatim
output into one JSONL per run with the fields needed for traceability (model,
benchmark version, item ID, prompt hash, generation config, timestamp). scorer.py
reads this file, not lm-eval's raw output directly.

Handles both protocols lm-eval-harness uses for MILU, since local (open-weight) and
API (closed/frontier) models can't be scored the same way -- API chat-completions
endpoints don't expose prompt loglikelihoods (see vendor/MILU's own
OpenAIChatCompletion.loglikelihood, which raises NotImplementedError for exactly
this reason):
  - "loglikelihood": local HF models via the `milu` / `milu_<Language>` tasks.
    lm-eval scores all 4 options' continuations and argmaxes.
  - "generate_until": API models via the `milu_api` / `milu_<Language>_api` tasks.
    Model free-generates a JSON answer, a regex filter extracts a letter (or the
    filter's "[invalid]" fallback if nothing matched) -- that fallback is what lets
    us log refusals/format failures separately from wrong answers.
"""
import glob
import json
import os

RAW_STORE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_raw_outputs")

MALFORMED_MARKERS = {"[invalid]", "", None}


def _normalize_loglikelihood_record(rec: dict, language: str) -> dict:
    doc = rec["doc"]
    option_logprobs = [float(r[0][0]) for r in rec["resps"]]
    target_index = int(rec["target"])
    predicted_index = max(range(len(option_logprobs)), key=lambda i: option_logprobs[i])
    return {
        "item_id": f"{language}:{rec['doc_id']}",
        "language": doc.get("language", language),
        "domain": doc.get("domain"),
        "subject": doc.get("subject"),
        "prompt_hash": rec["prompt_hash"],
        "doc_hash": rec["doc_hash"],
        "target_index": target_index,
        "predicted_index": predicted_index,
        "predicted_raw": None,
        "correct": bool(rec["acc"]),
        "is_malformed": False,  # loglikelihood scoring always picks one of the 4 options
        "option_logprobs": option_logprobs,
    }


def _normalize_generate_until_record(rec: dict, language: str) -> dict:
    doc = rec["doc"]
    # filtered_resps: list of [filter_name-selected string] per lm-eval's filter_list;
    # "get-answer" filter (regex -> take_first) yields a single extracted letter, or
    # the filter's fallback string ("[invalid]" by default) if nothing matched.
    filtered = rec.get("filtered_resps") or rec.get("resps")
    predicted_raw = filtered[0] if filtered else None
    if isinstance(predicted_raw, list):
        predicted_raw = predicted_raw[0] if predicted_raw else None

    is_malformed = predicted_raw in MALFORMED_MARKERS
    target_letter = rec.get("target")
    correct = bool(rec.get("exact_match", 0)) and not is_malformed

    letter_to_index = {"A": 0, "B": 1, "C": 2, "D": 3}
    return {
        "item_id": f"{language}:{rec['doc_id']}",
        "language": doc.get("language", language),
        "domain": doc.get("domain"),
        "subject": doc.get("subject"),
        "prompt_hash": rec.get("prompt_hash"),
        "doc_hash": rec.get("doc_hash"),
        "target_index": letter_to_index.get(target_letter),
        "predicted_index": letter_to_index.get(predicted_raw),
        "predicted_raw": predicted_raw,
        "correct": correct,
        "is_malformed": is_malformed,
        "option_logprobs": None,
    }


def normalize_run(model_cfg: dict, lm_eval_output_dir: str, milu_commit: str, dataset_revision: str, run_timestamp: str) -> str:
    """Reads all samples_milu_*.jsonl files lm-eval wrote for this model's run and
    writes a single normalized raw_items.jsonl under data_raw_outputs/<model_id>/<run_timestamp>/.
    Returns the path to the written file.
    """
    model_id = model_cfg["model_id"]
    protocol = model_cfg.get("protocol", "loglikelihood")
    if protocol not in ("loglikelihood", "generate_until"):
        raise ValueError(f"{model_id}: unknown protocol {protocol!r} (expected 'loglikelihood' or 'generate_until')")

    sample_files = sorted(glob.glob(os.path.join(lm_eval_output_dir, "**", "samples_milu_*.jsonl"), recursive=True))
    if not sample_files:
        raise FileNotFoundError(f"No lm-eval sample files found under {lm_eval_output_dir}")

    out_dir = os.path.join(RAW_STORE_DIR, model_id, run_timestamp)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "raw_items.jsonl")

    generation_config = model_cfg.get("generation_config", {})

    n_written = 0
    n_malformed = 0
    with open(out_path, "w") as out_f:
        for sample_file in sample_files:
            # lm-eval names files samples_<taskname>_<timestamp>.jsonl; taskname is
            # milu_<Language> (loglikelihood) or milu_<Language>_api (generate_until).
            basename = os.path.basename(sample_file)
            task_part = basename[len("samples_milu_"):].rsplit("_", 1)[0]
            language = task_part[:-len("_api")] if task_part.endswith("_api") else task_part
            with open(sample_file) as in_f:
                for line in in_f:
                    rec = json.loads(line)
                    if protocol == "loglikelihood":
                        fields = _normalize_loglikelihood_record(rec, language)
                    else:
                        fields = _normalize_generate_until_record(rec, language)

                    normalized = {
                        "model_id": model_id,
                        "hf_checkpoint": model_cfg.get("hf_checkpoint"),
                        "revision": model_cfg.get("revision"),
                        "api_endpoint": model_cfg.get("api_endpoint"),
                        "access_date": model_cfg.get("access_date"),
                        "benchmark": "milu",
                        "protocol": protocol,
                        "benchmark_version": {
                            "milu_repo_commit": milu_commit,
                            "hf_dataset_revision": dataset_revision,
                        },
                        "num_fewshot": model_cfg["num_fewshot"],
                        "apply_chat_template": model_cfg["apply_chat_template"],
                        "generation_config": generation_config,
                        "run_timestamp": run_timestamp,
                        "source_file": sample_file,
                        **fields,
                    }
                    out_f.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                    n_written += 1
                    n_malformed += int(normalized["is_malformed"])

    print(f"[store] wrote {n_written} normalized items to {out_path} ({n_malformed} malformed/refused)")
    return out_path


def load_raw_items(raw_items_path: str):
    with open(raw_items_path) as f:
        for line in f:
            yield json.loads(line)
