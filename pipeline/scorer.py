"""Scorer: offline pass over the raw-output store. MCQ accuracy overall,
per-language, and per-domain. Never re-runs inference -- only reads what
store.py already wrote.

Refusals and format failures are logged separately instead of being scored as
wrong. Malformed/refused items (store.py's is_malformed flag -- always False for
local loglikelihood scoring, only possible for generate_until/API scoring) are
excluded from the accuracy denominator and reported as their own malformed_rate
stat instead of being silently counted as wrong answers.
"""
from collections import defaultdict

from pipeline.store import load_raw_items


def _score(items):
    n = len(items)
    if n == 0:
        return {"accuracy": None, "n_items": 0, "n_malformed": 0, "malformed_rate": None}
    n_malformed = sum(1 for it in items if it["is_malformed"])
    scoreable = n - n_malformed
    correct = sum(1 for it in items if it["correct"])
    accuracy = correct / scoreable if scoreable else None
    return {
        "accuracy": accuracy,
        "n_items": n,
        "n_malformed": n_malformed,
        "malformed_rate": n_malformed / n,
    }


def score_run(raw_items_path: str) -> dict:
    items = list(load_raw_items(raw_items_path))

    by_language = defaultdict(list)
    by_domain = defaultdict(list)
    for it in items:
        by_language[it["language"]].append(it)
        by_domain[it["domain"]].append(it)

    return {
        "overall": _score(items),
        "by_language": {lang: _score(its) for lang, its in sorted(by_language.items())},
        "by_domain": {dom: _score(its) for dom, its in sorted(by_domain.items())},
    }
