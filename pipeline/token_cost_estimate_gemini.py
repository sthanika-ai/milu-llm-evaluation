"""Dry-run token estimator for Gemini models, proxied with the Gemma tokenizer.

Same 0-shot protocol as token_cost_estimate.py (system instruction + question,
short JSON answer), but counts tokens with the Gemma SentencePiece tokenizer
(google/gemma-3-27b-it, 262144-token vocab) instead of tiktoken. Gemini and
Gemma share the same tokenizer family, so this is a much closer proxy for
Gemini's real token count than tiktoken would be -- particularly for Indic
scripts, where tokenizer choice matters most. Still not exact: Gemini's own
tokenizer may differ from the Gemma checkpoints' in minor ways not reflected
here.

No API calls are made -- this tokenizer runs fully offline.
"""
import json
import os

import pandas as pd
from transformers import AutoTokenizer

from pipeline.token_cost_estimate import (
    LANGUAGES,
    SYSTEM_INSTRUCTION,
    ASSUMED_OUTPUT_JSON,
    doc_to_text,
    target_letter,
    load_split,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Public HF model identifier and git commit hash -- neither is a credential. Pinned
# to the same revision used for this checkpoint elsewhere in the roster
# (configs/models/gemma-3-27b-it.yaml).
GEMMA_MODEL_ID = "google/gemma-3-27b-it"  # nosemgrep
GEMMA_MODEL_REVISION = "005ad3404e59d6023443cb575daa05336842228a"  # nosemgrep # gitleaks:allow
tok = AutoTokenizer.from_pretrained(GEMMA_MODEL_ID, revision=GEMMA_MODEL_REVISION)


def n_tokens(text: str) -> int:
    return len(tok.encode(text, add_special_tokens=False))


def estimate_language(language: str) -> dict:
    test_df = load_split(language, "test")

    system_tokens = n_tokens(SYSTEM_INSTRUCTION)
    output_tokens_per_item = n_tokens(ASSUMED_OUTPUT_JSON)

    per_item_question_tokens = [n_tokens(doc_to_text(row)) for _, row in test_df.iterrows()]
    n_items = len(test_df)
    avg_question_tokens = sum(per_item_question_tokens) / n_items

    zero_shot_input_per_item = system_tokens + avg_question_tokens
    zero_shot_total_input = zero_shot_input_per_item * n_items
    zero_shot_total_output = output_tokens_per_item * n_items

    return {
        "language": language,
        "n_items": n_items,
        "avg_question_tokens": round(avg_question_tokens, 1),
        "zero_shot": {
            "input_tokens_total": round(zero_shot_total_input),
            "output_tokens_total": round(zero_shot_total_output),
        },
    }


def main():
    results = [estimate_language(lang) for lang in LANGUAGES]

    print(f"Tokenizer proxy: {GEMMA_MODEL_ID} (Gemini/Gemma shared family)")
    print(f"{'Language':<12} {'Items':>7} {'0-shot in':>12} {'0-shot out':>11}")
    totals = {"n_items": 0, "zs_in": 0, "zs_out": 0}
    for r in results:
        zs = r["zero_shot"]
        print(f"{r['language']:<12} {r['n_items']:>7} {zs['input_tokens_total']:>12,} "
              f"{zs['output_tokens_total']:>11,}")
        totals["n_items"] += r["n_items"]
        totals["zs_in"] += zs["input_tokens_total"]
        totals["zs_out"] += zs["output_tokens_total"]

    print("-" * 50)
    print(f"{'TOTAL':<12} {totals['n_items']:>7} {totals['zs_in']:>12,} {totals['zs_out']:>11,}")

    out_path = os.path.join(REPO_ROOT, "reports", "token_cost_estimate_gemini.json")
    with open(out_path, "w") as f:
        json.dump({"tokenizer_checkpoint": GEMMA_MODEL_ID, "per_language": results, "totals": totals},
                   f, indent=2, ensure_ascii=False)
    print(f"\nWrote detailed per-language breakdown to {out_path}")


if __name__ == "__main__":
    main()
