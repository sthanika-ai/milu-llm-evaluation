"""Estimates MILU cost for closed frontier models we did NOT run ourselves.

These are the models cited as comparison points in Claude Opus 4.7's, Claude
Sonnet 5's, and Claude Sonnet 4.6's own system cards' MILU tables -- not just
the three Claude models, but everything each card benchmarked itself against
(Gemini, GPT). We have no real API usage log for any of them, so we use each
vendor's own self-reported MILU accuracy and estimate cost from token counts.

Token counts come from whichever proxy tokenizer is the closest available
match to that model's real one:
  - "tiktoken": tiktoken's o200k_base encoding (token_cost_estimate.py). Exact
    for OpenAI's GPT-4o/5.x family; an approximation for Claude, which uses a
    different tokenizer -- see that file's own docstring for the caveat.
  - "gemma": a local Gemma tokenizer (token_cost_estimate_gemini.py). Gemini
    and Gemma share the same SentencePiece tokenizer family, so this is a
    materially closer proxy for Gemini than tiktoken would be, particularly
    on Indic scripts -- though still not guaranteed identical to Gemini's own.

Both proxies apply the project's standard 0-shot JSON-answer protocol -- the
same protocol already used for this project's other generative/API rows
(deepseek-v4-flash, etc).

Caveat this script does NOT attempt to correct for: every vendor's reported
MILU accuracy below used that model's own reasoning/thinking mode at a high
or maximum setting (see each source's own eval-conditions note). Thinking
tokens are billed as output tokens and are not included in this estimate --
so this number is a lower bound on what it would actually cost to reproduce
the vendor's exact accuracy figure, not an apples-to-apples cost for that
accuracy level. It IS an apples-to-apples cost for the same 0-shot
direct-answer protocol this project uses for its other API rows.

Where the same model has two different published MILU scores (Claude Opus
4.6 and Claude Sonnet 4.6 each appear in two system cards, evaluated at two
different thinking-effort settings -- see each entry's "note" below), we use
the higher, adaptive-max-effort figure from the model's own "home" card.
"""
import json
import os

from pipeline.api_pricing import PRICING

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INR_PER_USD = 95.29  # matches the rate used elsewhere in this project's cost methodology

# Vendor's own self-reported MILU accuracy (English + 10 Indic languages,
# equal-weighted average across languages -- NOT item-count-weighted like
# this project's own pipeline). Not run by this project.
FRONTIER_MODELS = {
    "claude-opus-4-7": {
        "milu_accuracy_pct": 89.9,
        "tokenizer": "tiktoken",
        "source": "Claude Opus 4.7 System Card, Table 8.12.2.B (own card)",
    },
    "claude-opus-4-8": {
        "milu_accuracy_pct": 91.1,
        "tokenizer": "tiktoken",
        "source": "Claude Sonnet 5 System Card, Figure 8.13.2.A",
    },
    "claude-opus-4-6": {
        "milu_accuracy_pct": 89.6,
        "tokenizer": "tiktoken",
        "source": "Claude Sonnet 4.6 System Card, Table 2.19.2.A -- also scored 87.6% in "
        "the Opus 4.7 card's table, which ran it at a lower medium thinking budget",
    },
    "claude-sonnet-5": {
        "milu_accuracy_pct": 89.3,
        "tokenizer": "tiktoken",
        "source": "Claude Sonnet 5 System Card, Figure 8.13.2.A (own card)",
    },
    "claude-sonnet-4-6": {
        "milu_accuracy_pct": 89.6,
        "tokenizer": "tiktoken",
        "source": "Claude Sonnet 4.6 System Card, Table 2.19.2.A (own card) -- also scored "
        "87.1% in the Opus 4.7 card's table, which ran it at a lower medium thinking budget",
    },
    "claude-sonnet-4-5": {
        "milu_accuracy_pct": 87.6,
        "tokenizer": "tiktoken",
        "source": "Claude Sonnet 4.6 System Card, Table 2.19.2.A",
    },
    "claude-mythos-preview": {
        "milu_accuracy_pct": 92.7,
        "tokenizer": "tiktoken",
        "source": "Claude Sonnet 5 System Card, Figure 8.13.2.A",
    },
    "gpt-5.4": {
        "milu_accuracy_pct": 90.6,
        "tokenizer": "tiktoken",
        "source": "Claude Opus 4.7 System Card, Table 8.12.2.B",
    },
    "gpt-5.2-pro": {
        "milu_accuracy_pct": 89.2,
        "tokenizer": "tiktoken",
        "source": "Claude Sonnet 4.6 System Card, Table 2.19.2.A",
    },
    "gemini-3.1-pro": {
        "milu_accuracy_pct": 93.6,
        "tokenizer": "gemma",
        "source": "Claude Opus 4.7 System Card, Table 8.12.2.B",
    },
    "gemini-3-pro": {
        "milu_accuracy_pct": 93.2,
        "tokenizer": "gemma",
        "source": "Claude Sonnet 4.6 System Card, Table 2.19.2.A",
    },
}


def main():
    with open(os.path.join(REPO_ROOT, "reports", "token_cost_estimate.json")) as f:
        tiktoken_totals = json.load(f)["totals"]
    with open(os.path.join(REPO_ROOT, "reports", "token_cost_estimate_gemini.json")) as f:
        gemma_totals = json.load(f)["totals"]

    token_totals = {
        "tiktoken": (tiktoken_totals["n_items"], tiktoken_totals["zs_in"], tiktoken_totals["zs_out"]),
        "gemma": (gemma_totals["n_items"], gemma_totals["zs_in"], gemma_totals["zs_out"]),
    }

    for tokenizer, (n_items, input_tokens, output_tokens) in token_totals.items():
        print(f"0-shot token totals across all {n_items:,} MILU items ({tokenizer} proxy):")
        print(f"  input:  {input_tokens:,}")
        print(f"  output: {output_tokens:,}")
    print()

    for model_id, info in FRONTIER_MODELS.items():
        n_items, input_tokens, output_tokens = token_totals[info["tokenizer"]]
        pricing = PRICING[model_id]
        usd = (input_tokens / 1_000_000) * pricing["input_per_1m"] + (
            output_tokens / 1_000_000
        ) * pricing["output_per_1m"]
        inr = usd * INR_PER_USD
        inr_per_1k = inr / (n_items / 1000)
        print(f"{model_id}: MILU accuracy {info['milu_accuracy_pct']}% ({info['source']})")
        print(f"  tokenizer proxy: {info['tokenizer']}")
        print(f"  estimated full-run cost: ${usd:.2f} = Rs.{inr:.2f}")
        print(f"  estimated Rs./1K answered questions: Rs.{inr_per_1k:.2f}")
        print()


if __name__ == "__main__":
    main()
