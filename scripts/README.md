# Scripts

- **`setup_vendor.sh`** — one-time setup: clones `vendor/MILU` and `vendor/llama.cpp` at the
  pinned commits and applies this project's local patch to `vendor/MILU` (see
  `../patches/README.md`). Run this once after cloning, before anything else.
- **`run_evaluation.sh`** — thin wrapper around `python -m pipeline.run`.
- **`export_results_tables.py`** — exports `results/results.db` (gitignored, machine-local,
  produced by running the pipeline yourself) into `results/tables/*.csv`, a diff-friendly
  summary. Run this after any evaluation run to get a readable view of your results.

  `league_table.csv` is a raw dump of every scored run in `results.db` — including smoke
  tests, single-language investigation runs, and any superseded pre-fix rows you've run
  locally. It is not automatically deduplicated down to one authoritative row per model;
  cross-reference by `model_id` + `run_timestamp` if you need to trace a specific number back
  to a specific run.
