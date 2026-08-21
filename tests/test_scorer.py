"""Unit tests for pipeline/scorer.py -- pure logic, no GPU/API/network dependency.

Exercises _score() directly and score_run() against a small synthetic raw_items.jsonl,
in the same schema pipeline/store.py actually writes (one JSON object per line: language,
domain, correct, is_malformed -- see pipeline/store.py's _normalize_* functions).
"""
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from pipeline.scorer import _score, score_run  # noqa: E402


def _item(language, domain, correct, is_malformed=False):
    return {"language": language, "domain": domain, "correct": correct, "is_malformed": is_malformed}


def test_score_empty():
    result = _score([])
    assert result == {"accuracy": None, "n_items": 0, "n_malformed": 0, "malformed_rate": None}  # nosec B101


def test_score_all_correct():
    items = [_item("English", "Science", True) for _ in range(4)]
    result = _score(items)
    assert result["accuracy"] == 1.0  # nosec B101
    assert result["n_items"] == 4  # nosec B101
    assert result["n_malformed"] == 0  # nosec B101
    assert result["malformed_rate"] == 0.0  # nosec B101


def test_score_malformed_excluded_from_accuracy_denominator():
    # 2 correct, 1 wrong, 1 malformed -- malformed must not count as wrong.
    items = [
        _item("English", "Science", True),
        _item("English", "Science", True),
        _item("English", "Science", False),
        _item("English", "Science", False, is_malformed=True),
    ]
    result = _score(items)
    assert result["n_items"] == 4  # nosec B101
    assert result["n_malformed"] == 1  # nosec B101
    assert result["malformed_rate"] == 0.25  # nosec B101
    # accuracy = correct / (n - n_malformed) = 2 / 3, not 2 / 4
    assert abs(result["accuracy"] - (2 / 3)) < 1e-9  # nosec B101


def test_score_all_malformed_gives_none_accuracy():
    items = [_item("English", "Science", False, is_malformed=True) for _ in range(3)]
    result = _score(items)
    assert result["accuracy"] is None  # nosec B101
    assert result["n_malformed"] == 3  # nosec B101


def test_score_run_breaks_down_by_language_and_domain(tmp_path):
    items = [
        _item("English", "Science", True),
        _item("English", "Science", False),
        _item("Hindi", "Arts", True),
        _item("Hindi", "Arts", True),
    ]
    raw_path = tmp_path / "raw_items.jsonl"
    with open(raw_path, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")

    scores = score_run(str(raw_path))

    assert scores["overall"]["n_items"] == 4  # nosec B101
    assert scores["overall"]["accuracy"] == 0.75  # 3 correct (1 English + 2 Hindi) of 4  # nosec B101

    assert set(scores["by_language"]) == {"English", "Hindi"}  # nosec B101
    assert scores["by_language"]["English"]["accuracy"] == 0.5  # nosec B101
    assert scores["by_language"]["Hindi"]["accuracy"] == 1.0  # nosec B101

    assert set(scores["by_domain"]) == {"Science", "Arts"}  # nosec B101
    assert scores["by_domain"]["Science"]["n_items"] == 2  # nosec B101
    assert scores["by_domain"]["Arts"]["n_items"] == 2  # nosec B101
