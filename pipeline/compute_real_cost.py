"""Sums a run's real token-usage log (written by the patched
vendor/MILU/lm_eval/models/openai_completions.py) into a real $ cost --
replaces the upfront tiktoken-proxy estimate (token_cost_estimate.py) with
what the API actually billed, once a run has happened.
"""
import argparse
import json

from pipeline.api_pricing import estimate_cost_usd


def sum_usage(usage_log_path: str) -> dict:
    prompt_tokens = 0
    completion_tokens = 0
    n_calls = 0
    with open(usage_log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            prompt_tokens += rec.get("prompt_tokens") or 0
            completion_tokens += rec.get("completion_tokens") or 0
            n_calls += 1
    return {"n_calls": n_calls, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--usage-log", required=True)
    p.add_argument("--model-id", required=True, help="Key into pipeline/api_pricing.py's PRICING dict")
    args = p.parse_args()

    usage = sum_usage(args.usage_log)
    cost = estimate_cost_usd(args.model_id, usage["prompt_tokens"], usage["completion_tokens"])

    print(f"[real-cost] {args.model_id}: {usage['n_calls']} calls, "
          f"{usage['prompt_tokens']:,} prompt tokens, {usage['completion_tokens']:,} completion tokens")
    print(f"[real-cost] {args.model_id}: ${cost:.4f}")
    return {**usage, "cost_usd": cost}


if __name__ == "__main__":
    main()
