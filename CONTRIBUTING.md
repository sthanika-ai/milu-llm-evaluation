# Contributing

This started as an internal evaluation campaign and is published so the pipeline and results
are reproducible and auditable. It isn't run as a large open community project, but
contributions are welcome, especially:

- **New model configs** — the highest-value contribution. Add
  `configs/models/<model_id>.yaml` following the schema in `configs/README.md` and
  `pipeline/registry.py`'s required-fields lists. Run the model, then include the scored
  summary alongside your PR so the result is traceable.
- **Fixes to the scoring/pipeline code** (`pipeline/`) — please add or update a test in
  `tests/` for any change to `scorer.py` or `registry.py`'s logic.
- **Corrections to methodology** — if you find a scoring bug, document it clearly: symptom,
  root cause, fix, and before/after numbers, rather than silently changing a reported figure.

## Development setup

For pipeline-only changes that don't need a GPU or the vendored harness, the
`requirements/default.freeze.txt` environment plus `pytest tests/` is enough to iterate. For
full environment setup (venvs, vendored harness, model access), see the root `README.md`.

## Before opening a PR

- `pytest tests/`
- If you touched `pipeline/scorer.py`, `pipeline/registry.py`, or `pipeline/results_db.py`,
  confirm `scripts/export_results_tables.py` still runs cleanly against `results/results.db`.
- Do not commit `.env`, real API keys, or anything under the gitignored paths listed in
  `.gitignore` (venvs, `vendor/`, `data_raw_outputs/`, `mlruns/`, `results/`).
