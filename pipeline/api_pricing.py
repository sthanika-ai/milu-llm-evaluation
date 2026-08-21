"""Per-model API pricing, $ per 1M tokens. Manually maintained -- pricing pages
change without notice, so each entry records where/when it was checked rather
than trusting a stale number silently.

Cache-hit rates exist for some providers (repeated prompt prefixes billed far
cheaper) but aren't applied here: our real usage log (see openai_completions.py's
_log_usage) only gets prompt_tokens/completion_tokens/total_tokens back from a
standard OpenAI-compatible response, not a cache-hit/miss split, so real-cost
computation conservatively prices all input at the cache-miss rate.
"""

PRICING = {
    "deepseek-v4-flash": {
        "input_per_1m": 0.14,
        "output_per_1m": 0.28,
        "source": "https://deepseek.ai/pricing, checked 2026-08-01",
    },
    "deepseek-v4-pro": {
        "input_per_1m": 0.435,
        "output_per_1m": 0.87,
        "source": "https://deepseek.ai/pricing, checked 2026-08-01",
    },
    "qwen3.8-max": {
        "input_per_1m": 2.00,
        "output_per_1m": 6.00,
        "source": "https://openrouter.ai/qwen/qwen3.8-max, checked 2026-08-06",
    },
    "sarvam-105b": {
        "input_per_1m": 0.3073,
        "output_per_1m": 0.7683,
        "source": "https://docs.sarvam.ai/api/getting-started/pricing, checked 2026-08-06 -- "
        "listed in INR (input Rs.29.28/1M, output Rs.73.2/1M), converted at 1 USD = "
        "95.28 INR (rate as given by user 2026-08-06). Sarvam also lists a cheaper "
        "cached-input rate (Rs.10.98/1M) not applied here -- same reason as every "
        "other entry in this file: our real usage log only gets prompt/completion/"
        "total token counts back, no cache-hit/miss split, so real-cost computation "
        "conservatively prices all input at the cache-miss rate.",
    },
    # Not run by this project -- pricing added only to estimate cost against
    # each vendor's own self-reported MILU accuracy for these models. See
    # pipeline/estimate_frontier_model_cost.py. Standard (post-introductory)
    # per-token rate, <=200K context tier where a provider has tiered pricing.
    "claude-opus-4-7": {
        "input_per_1m": 5.00,
        "output_per_1m": 25.00,
        "source": "Anthropic model catalog / pricing page, checked 2026-08-06",
    },
    "claude-opus-4-8": {
        "input_per_1m": 5.00,
        "output_per_1m": 25.00,
        "source": "Anthropic model catalog / pricing page, checked 2026-08-06 -- same Opus-tier rate as Opus 4.7",
    },
    "claude-opus-4-6": {
        "input_per_1m": 5.00,
        "output_per_1m": 25.00,
        "source": "Anthropic model catalog / pricing page, checked 2026-08-06 -- same Opus-tier rate as Opus 4.7",
    },
    "claude-sonnet-5": {
        "input_per_1m": 3.00,
        "output_per_1m": 15.00,
        "source": "Anthropic model catalog / pricing page, checked 2026-08-06 -- "
        "standard rate; introductory rate $2.00/$10.00 per 1M in effect through "
        "2026-08-31",
    },
    "claude-sonnet-4-6": {
        "input_per_1m": 3.00,
        "output_per_1m": 15.00,
        "source": "Anthropic model catalog / pricing page, checked 2026-08-06",
    },
    "claude-sonnet-4-5": {
        "input_per_1m": 3.00,
        "output_per_1m": 15.00,
        "source": "Assumed same Sonnet-tier rate as Sonnet 4.6/5 -- not independently "
        "confirmed for this legacy model, checked 2026-08-06",
    },
    "claude-mythos-preview": {
        "input_per_1m": 10.00,
        "output_per_1m": 50.00,
        "source": "Assumed same top tier as Claude Mythos 5/Fable 5 -- Mythos Preview was "
        "invitation-only (Project Glasswing) with no public pricing page, so this is "
        "an assumption, not a confirmed rate. Checked 2026-08-06",
    },
    "gpt-5.4": {
        "input_per_1m": 2.50,
        "output_per_1m": 15.00,
        "source": "OpenAI API pricing aggregators (OpenRouter/pricepertoken), checked "
        "2026-08-06 -- standard tier, <=200K context (long-context tier $5.00/$22.50)",
    },
    "gpt-5.2-pro": {
        "input_per_1m": 10.50,
        "output_per_1m": 84.00,
        "source": "OpenAI API pricing aggregator (pricepertoken), checked 2026-08-06",
    },
    "gemini-3.1-pro": {
        "input_per_1m": 2.00,
        "output_per_1m": 12.00,
        "source": "Google Gemini API pricing aggregators, checked 2026-08-06 -- standard "
        "tier, <=200K context (over 200K: $4.00/$18.00)",
    },
    "gemini-3-pro": {
        "input_per_1m": 2.00,
        "output_per_1m": 12.00,
        "source": "Google Gemini API pricing aggregators, checked 2026-08-06 -- standard "
        "tier, <=200K context (over 200K: $4.00/$18.00)",
    },
}


def estimate_cost_usd(model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
    if model_id not in PRICING:
        raise KeyError(f"No pricing entry for {model_id!r} -- add one to PRICING first.")
    p = PRICING[model_id]
    return (prompt_tokens / 1_000_000) * p["input_per_1m"] + (completion_tokens / 1_000_000) * p["output_per_1m"]
