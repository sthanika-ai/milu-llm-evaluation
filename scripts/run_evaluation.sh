#!/usr/bin/env bash
# Thin wrapper around `python -m pipeline.run`. Exists only to document the real
# invocation in one copy-pasteable place -- it does not add any behavior pipeline/run.py
# doesn't already have. All actual argument parsing/dispatch logic lives in
# pipeline/run.py:main() (--models, --cuda-visible-devices, --limit).
#
# IMPORTANT: pipeline/run.py dispatches most models to lm_eval as a subprocess under one
# of the venvs in requirements/README.md, chosen per-model via each config's `venv:` field
# -- but not every model's config is run through this script. Sarvam-M thinkmode, for
# example, bypasses this dispatcher entirely and was run via internal tooling not
# included in this repository (see that config's own notes: field for what it needed).
#
# Usage:
#   scripts/run_evaluation.sh                                  # every config in configs/models/
#   scripts/run_evaluation.sh --models gemma-2-2b-it            # sanity gate
#   scripts/run_evaluation.sh --models gemma-2-2b-it --limit 20 # smoke test, a few items
#   scripts/run_evaluation.sh --cuda-visible-devices 0 --models gemma-3-27b-it
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
python -m pipeline.run "$@"
