"""Dry-run token estimator for running MILU on paid frontier APIs.

Per the paper (arXiv:2411.02538, Sec 4): API/closed models can't get prompt
loglikelihoods, so they're evaluated generatively -- prompted to emit a
structured JSON answer, 0-shot only ("due to the high costs involved").
We estimate both that 0-shot protocol and a 5-shot alternative (which lets us
match our own open-model protocol and exploit prompt caching on the fixed
few-shot block, an option not really available to the paper's authors).

No API calls are made. Token counts are proxied with tiktoken's o200k_base
encoding (GPT-4o's exact tokenizer; a close-but-not-exact proxy for Gemini and
Claude, which use different tokenizers -- especially for Indic scripts).
"""
import glob
import json
import os

import pandas as pd
import tiktoken

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF_CACHE_SNAPSHOT_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--ai4bharat--MILU/snapshots/946c423e72cd2657674a7a65d739e212e9a5f876"
)
LANGUAGES = [
    "Bengali", "English", "Gujarati", "Hindi", "Kannada", "Malayalam",
    "Marathi", "Odia", "Punjabi", "Tamil", "Telugu",
]

SYSTEM_INSTRUCTION = (
    "You are answering a multiple-choice question. Read the question and the four "
    "options, then respond with ONLY a JSON object of the form "
    '{"answer": "A"} (or "B", "C", "D") -- no other text.'
)
ASSUMED_OUTPUT_JSON = '{"answer": "C"}'

enc = tiktoken.get_encoding("o200k_base")


def n_tokens(text: str) -> int:
    return len(enc.encode(text))


def doc_to_text(doc) -> str:
    choices = [doc["option1"], doc["option2"], doc["option3"], doc["option4"]]
    option_choices = {"A": choices[0], "B": choices[1], "C": choices[2], "D": choices[3]}
    prompt = "Question: " + doc["question"] + "\nChoices:\n"
    for choice, option in option_choices.items():
        prompt += f"{choice.upper()}. {option}\n"
    prompt += "Answer:"
    return prompt


def target_letter(doc) -> str:
    option_number = ["1", "2", "3", "4"].index(doc["target"].split("option")[1])
    return "ABCD"[option_number]


def load_split(language: str, split: str) -> pd.DataFrame:
    path = os.path.join(HF_CACHE_SNAPSHOT_GLOB, language, f"{split}-00000-of-00001.parquet")
    return pd.read_parquet(path)


def estimate_language(language: str) -> dict:
    test_df = load_split(language, "test")
    val_df = load_split(language, "validation")

    fewshot_docs = [row for _, row in val_df.head(5).iterrows()]
    fewshot_block = "\n\n".join(
        doc_to_text(ex) + " " + target_letter(ex) for ex in fewshot_docs
    )

    system_tokens = n_tokens(SYSTEM_INSTRUCTION)
    fewshot_tokens = n_tokens(fewshot_block)
    output_tokens_per_item = n_tokens(ASSUMED_OUTPUT_JSON)

    per_item_question_tokens = []
    for _, row in test_df.iterrows():
        per_item_question_tokens.append(n_tokens(doc_to_text(row)))

    n_items = len(test_df)
    avg_question_tokens = sum(per_item_question_tokens) / n_items

    # 0-shot: every request pays system + question in full, nothing cacheable
    # (system instruction is tiny and identical every time, but most providers
    # only start discounting cached prefixes above ~1024 tokens, so we count
    # it as fresh here to stay conservative).
    zero_shot_input_per_item = system_tokens + avg_question_tokens
    zero_shot_total_input = zero_shot_input_per_item * n_items
    zero_shot_total_output = output_tokens_per_item * n_items

    # 5-shot: system + fixed fewshot block is IDENTICAL for every item in this
    # language -> cacheable after the first request. Only the per-item question
    # is genuinely fresh every time.
    shared_prefix_tokens = system_tokens + fewshot_tokens
    five_shot_fresh_input = shared_prefix_tokens + avg_question_tokens * n_items
    five_shot_cached_input = shared_prefix_tokens * max(n_items - 1, 0)
    five_shot_total_output = output_tokens_per_item * n_items

    return {
        "language": language,
        "n_items": n_items,
        "avg_question_tokens": round(avg_question_tokens, 1),
        "fewshot_block_tokens": fewshot_tokens,
        "zero_shot": {
            "input_tokens_total": round(zero_shot_total_input),
            "output_tokens_total": round(zero_shot_total_output),
            "cached_tokens_total": 0,
        },
        "five_shot": {
            "fresh_input_tokens_total": round(five_shot_fresh_input),
            "cached_input_tokens_total": round(five_shot_cached_input),
            "output_tokens_total": round(five_shot_total_output),
        },
    }


def main():
    results = [estimate_language(lang) for lang in LANGUAGES]

    print(f"{'Language':<12} {'Items':>7} {'0-shot in':>12} {'0-shot out':>11} "
          f"{'5-shot fresh':>13} {'5-shot cached':>14} {'5-shot out':>11}")
    totals = {"n_items": 0, "zs_in": 0, "zs_out": 0, "fs_fresh": 0, "fs_cached": 0, "fs_out": 0}
    for r in results:
        zs, fs = r["zero_shot"], r["five_shot"]
        print(f"{r['language']:<12} {r['n_items']:>7} {zs['input_tokens_total']:>12,} "
              f"{zs['output_tokens_total']:>11,} {fs['fresh_input_tokens_total']:>13,} "
              f"{fs['cached_input_tokens_total']:>14,} {fs['output_tokens_total']:>11,}")
        totals["n_items"] += r["n_items"]
        totals["zs_in"] += zs["input_tokens_total"]
        totals["zs_out"] += zs["output_tokens_total"]
        totals["fs_fresh"] += fs["fresh_input_tokens_total"]
        totals["fs_cached"] += fs["cached_input_tokens_total"]
        totals["fs_out"] += fs["output_tokens_total"]

    print("-" * 95)
    print(f"{'TOTAL':<12} {totals['n_items']:>7} {totals['zs_in']:>12,} {totals['zs_out']:>11,} "
          f"{totals['fs_fresh']:>13,} {totals['fs_cached']:>14,} {totals['fs_out']:>11,}")

    out_path = os.path.join(REPO_ROOT, "reports", "token_cost_estimate.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"per_language": results, "totals": totals}, f, indent=2, ensure_ascii=False)
    print(f"\nWrote detailed per-language breakdown to {out_path}")


if __name__ == "__main__":
    main()
