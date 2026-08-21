# Reproducing this project's results

A from-scratch walkthrough, assuming a Linux box with an NVIDIA GPU for the local-model rows
(API-model rows need no GPU).

## 0. Before you start

- Python 3.12 (this project ran on 3.12.9).
- An NVIDIA GPU for local models — this project ran on 2× A100 80GB PCIe (see
  `requirements/README.md`'s hardware note). Smaller GPUs can run the smaller models (Gemma 3
  4B, Sarvam-1) but not the 24B+ ones at full precision.
- A Hugging Face account, for two reasons: **the MILU dataset itself is gated** (request access
  at [huggingface.co/datasets/ai4bharat/MILU](https://huggingface.co/datasets/ai4bharat/MILU) —
  do this first, approval can take time), and **some checkpoints are also gated** (e.g. Gemma).
  Authenticate with `huggingface-cli login` or `HF_TOKEN`.
- API keys for any API-backed models you want to run — copy `.env.example` to `.env` and fill
  in what you need.

## 1. Set up the vendored harness

```bash
bash scripts/setup_vendor.sh
```

Clones `AI4Bharat/MILU` at the pinned commit into `vendor/MILU`, applies this project's own
patch (`patches/vendor-milu.diff` + `patches/vendor-milu-new-files/` — see
`patches/README.md`), then clones and builds `vendor/llama.cpp` (only needed for the
Sarvam-30B GGUF path).

## 2. Set up the Python environment(s)

This project uses **five separate venvs** for different model architectures/backends — real,
load-bearing complexity, documented in full in `requirements/README.md` (read it before
proceeding, especially the disclosed `transformers` version drift). Minimally, to run any local
model:

```bash
python3.12 -m venv .venv-multimodal
source .venv-multimodal/bin/activate
pip install -r requirements/multimodal.freeze.txt
pip install mlflow tiktoken   # not in any freeze file -- see requirements/README.md
```

Repeat for whichever other venv(s) your target model needs — `requirements/README.md`'s table
maps each config's `venv:` field to a freeze file.

## 3. Run the sanity gate first

```bash
python -m pipeline.run --models gemma-2-2b-it
```

Expect ~28-29% on Gujarati, matching the MILU paper's own published 29.25% within about a
point. If it doesn't land close, stop and debug the environment before running anything else —
see `docs/troubleshooting.md`.

## 4. Run an evaluation

```bash
python -m pipeline.run --models gemma-3-27b-it            # one model, all 11 languages
python -m pipeline.run --models gemma-2-2b-it --limit 20   # smoke test, a handful of items
python -m pipeline.run                                     # every config in configs/models/
```

For each matching config, `pipeline/run.py` invokes the vendored `lm_eval` CLI as a subprocess
under that config's `venv:` environment, normalizes the output into
`data_raw_outputs/<model_id>/<timestamp>/raw_items.jsonl` (`pipeline/store.py`), scores it
(`pipeline/scorer.py`), and writes rows into `results/results.db`
(`pipeline/results_db.py`) plus an MLflow run (`pipeline/mlflow_logging.py`).

An externally-produced run (e.g. from a different harness invocation) can be scored and ingested
the same way with `python -m pipeline.ingest_external_run <model_id> <lm_eval_output_dir>
--dataset-revision <sha>`.

### Three configs need extra steps beyond `pipeline.run`

- **`sarvam-30b-llamacpp-gguf`** and **`qwen3.6-27b-llamacpp-gguf`** each need a hand-started
  `llama-server` (built by `scripts/setup_vendor.sh`) serving their own Q4_K_M GGUF checkpoint
  on the `base_url` the config specifies, before `pipeline.run` can dispatch to it.
- **`sarvam-m-thinkmode-0shot-json`** doesn't go through `pipeline.run`'s dispatcher at all —
  its own `notes:` field explains why and gives the direct invocation. That config, as
  committed, is a 200-item English-only check; the full 11-language production number
  (82.44%) was produced with additional internal tooling not included in this repository.

The same applies to the malformed-item retry passes behind `gpt-oss-20b-thinkmode-vllm`'s
72.22% and `sarvam-30b-llamacpp-gguf`'s 72.99%: the base run via `pipeline.run` gets most of
the way there, and the retry step that recovers the rest used internal scripts that aren't part
of this repository. Each affected config's own `notes:` field says so and gives the retry
parameters (`max_gen_toks`, `--reasoning-budget`) so the gap is disclosed, not hidden. See
`DEVELOPER_NOTES.md` §§1-3 for the full account of every fix, and `configs/README.md` for which
configs this applies to.

## 5. Export and compare results

```bash
python scripts/export_results_tables.py
```

Writes `results/tables/league_table.csv`, `by_language.csv`, `by_domain.csv` from
`results/results.db`. This CSV dump includes every scored run in your local database (smoke
tests, investigation variants included) — cross-reference against the curated roster in the
root `README.md`.

## 6. Read the rest

The root `README.md` has the full results table. `DEVELOPER_NOTES.md` is the engineering log
behind it — every scoring correction and infrastructure fix, with root causes and before/after
numbers.
