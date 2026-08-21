# MILU Evaluation of Newer LLMs

A reproducible pipeline and results for evaluating large language models on
[MILU](https://arxiv.org/abs/2411.02538) — [AI4Bharat](https://ai4bharat.iitm.ac.in/) and
IBM's Multi-task Indic Language Understanding benchmark: ~85,000 multiple-choice questions
across 8 domains and 41 subjects, in 11 Indic languages, drawn from Indian regional and state
exams (NAACL 2025). MILU's own paper evaluated 42+ LLMs current as of late 2024; the top
score (GPT-4o) was ~74%.

This repository presents a completed evaluation campaign: 18 models, all 11 MILU languages,
end to end. Every reported figure traces back to a raw model output through the pipeline in
this repository, so results can be independently reproduced or audited rather than taken on
faith.

**MILU is AI4Bharat and IBM's benchmark. We run it as adopters, not authors.** This repo's
contribution is: coverage of newer models (2025-2026 checkpoints the original paper
predates), a cost-per-rupee lens the original paper doesn't report, and whatever findings
fall out of that — complementary to AI4Bharat/IBM's work, not competitive with it. See
[Credit](#credit) below.

**What's novel here, beyond re-running the paper's own models:** a reusable evaluation
pipeline (model-major scheduling, a raw-output store fully decoupled from scoring, a
queryable results DB); real, disclosed discovery and correction of a scoring-protocol bug
that silently breaks four different hybrid-"thinking" model families (see
[Thinking/reasoning configuration](#8-thinkingreasoning-configuration) below) — a bug that,
uncorrected, would have inverted two models' rank in the league table; a genuine
thinking-on-vs-off head-to-head for one model (Qwen3.6-27B) showing a 16.84-point swing from
that single setting alone; and a real cost-per-1K-questions figure alongside every accuracy
number, not just accuracy in isolation.

## 1. Results

Overall accuracy plus real measured cost, for the models fully evaluated so far (GPU-hours
are priced at an **illustrative** $1.80/hr reference rate, not a real bill — this box is
privately provisioned, not metered per model).

| Rank | Model | Protocol | Overall | Cost | Note |
|---|---|---|---|---|---|
| 1 | Qwen3.8-Max (API) | generative, 0-shot (API) | **89.67%** | $156.60 (real) | highest accuracy in the roster; reasoning is mandatory on this endpoint (can't be disabled), so cost isn't directly comparable to fully-disabled-reasoning rows — see that config's `notes:` field |
| 2 | Qwen3.6-27B (thinking on, llama.cpp GGUF) | generative, 0-shot | 83.84% | $134.83 (74.90 GPU-hr) | +16.84 points over the thinking-off row below, from that setting alone — see §8 below |
| 3 | Sarvam-M 24B (thinkmode) | generative, 0-shot | 82.44% | $34.96 (19.42 GPU-hr) | corrected row, see §8 below |
| 4 | DeepSeek V4-Flash | generative, 0-shot (API) | 79.51% | $2.05 (real) | complete, 0% malformed |
| 5 | Sarvam-30B (llama.cpp GGUF) | generative, 0-shot | 72.99% | $137.87 (76.60 GPU-hr) | most compute-intensive model in the roster |
| 6 | gpt-oss-20b (thinkmode, vLLM) | generative, 0-shot | 72.22% | $16.47 (9.15 GPU-hr) | corrected row, see §8 below |
| 7 | Qwen3.6-27B (thinking off) | generative, 0-shot | 67.00% | $6.82 (3.79 GPU-hr) | corrected row, see §8 below |
| 8 | Mistral Small 3.1 24B | generative, 0-shot | 64.04% | $1.33 (0.74 GPU-hr) | |
| 9 | Gemma 3 27B | loglikelihood, 5-shot | 63.95% | $23.85 (13.25 GPU-hr) | |
| 10 | Llama 4 Scout (17B active/109B total MoE) | loglikelihood, 5-shot | 63.06% | $39.60 (22.00 GPU-hr) | |
| 11 | Gemma 4 12B | loglikelihood, 5-shot | 61.12% | $12.92 (7.18 GPU-hr) | |
| 12 | Gemma 3 12B | loglikelihood, 5-shot | 56.41% | $12.08 (6.71 GPU-hr) | |
| 13 | Gemma 3 12B INT4 | loglikelihood, 5-shot | 53.94% | $11.66 (6.48 GPU-hr) | |
| 14 | Qwen3-VL 8B Instruct | loglikelihood, 5-shot | 50.33% | $15.70 (8.72 GPU-hr) | multimodal architecture, text-only used |
| 15 | Phi-4 14B | loglikelihood, 5-shot | 47.55% | $26.26 (14.59 GPU-hr) | |
| 16 | Gemma 3 4B | loglikelihood, 5-shot | 43.03% | $4.52 (2.51 GPU-hr) | |
| 17 | Qwen2.5 7B Instruct | loglikelihood, 5-shot | 41.81% | $14.26 (7.92 GPU-hr) | |
| 18 | Sarvam-1 2B | loglikelihood, 5-shot | 28.63% | $2.65 (1.47 GPU-hr) | |

Also in `configs/models/` but not in the table above: `gemma-2-2b-it` (a sanity-gate spot
check against the paper's own published number, not a full roster row — see §7). 11
frontier-model reference points (Claude/GPT/Gemini, cited from vendor system cards, **not run
by us**) round out the fuller internal comparison but aren't reproduced here.

Regenerate a machine-readable version of this table yourself after running the pipeline via
`python scripts/export_results_tables.py` (see §9) — the underlying `results/results.db` and
its CSV exports aren't included in this repo.

## 2. Models evaluated

19 curated, published configs in `configs/models/` — one canonical, full-11-language,
preferred-method config per model with a row in the table above, plus the sanity-gate config
— spanning local open-weight models (Gemma 2/3/4, Qwen 2.5/3.6, Sarvam-1/M/30B, Llama,
Mistral Small 3.1, Phi-4, gpt-oss-20b, Qwen3-VL) and two API models (DeepSeek V4-Flash,
Qwen3.8-Max) evaluated end-to-end. Qwen3.6-27B has two of those configs, not one — thinking
forced off and thinking left on are both genuine, disclosed configurations with materially
different results (§8), so both are published rather than picking one. This project's own full
investigation history (53 configs — smoke tests, staging runs, superseded pre-bugfix rows)
isn't published; see
`configs/README.md`'s "Published vs. full local registry" section for exactly what's excluded
and why. Checkpoint revision, quantization, protocol rationale, and any library patches a
given model needed are documented in that model's own config `notes:` field — see
`configs/README.md` for the schema.

## 3. Repository structure

```
configs/models/*.yaml   19 curated, published model configs (checkpoint, revision, backend, generation config)
configs/README.md       config schema + naming conventions
pipeline/                evaluation pipeline: registry, scheduler, raw-output store, scorer,
                          results DB, MLflow logging, cost-estimation utilities
patches/                 vendor/MILU's local modifications, captured for reproducibility
requirements/            per-venv pip-freeze snapshots + the 5-venv strategy writeup
scripts/                 setup_vendor.sh, run_evaluation.sh, export_results_tables.py
tests/                   unit tests for pipeline/scorer.py and pipeline/registry.py
```

`vendor/MILU` and `vendor/llama.cpp` (third-party, cloned by `scripts/setup_vendor.sh`, not
committed) and `data_raw_outputs/`, `mlruns/`, `results/results.db`, `.venv*` (this project's
own large, regeneratable, or machine-specific artifacts, gitignored) exist locally but aren't
part of the published repo — see [Known limitations](#10-known-limitations).

## 4. Installation

```bash
git clone <this-repo-url>
cd milu-llm-evaluation
bash scripts/setup_vendor.sh          # clones + patches vendor/MILU, clones + builds vendor/llama.cpp

python3.12 -m venv .venv-multimodal
source .venv-multimodal/bin/activate
pip install -r requirements/multimodal.freeze.txt
pip install mlflow tiktoken           # gap in the lock files -- see requirements/README.md
```

This project needs **five separate venvs** for different model backends — this is real,
load-bearing complexity (see `requirements/README.md`), not something to collapse into one
`requirements.txt`. Set up only the venv(s) your target model needs.

## 5. Configuration

```bash
cp .env.example .env
# fill in whichever of these you need -- see .env.example's comments for which model uses which
```

Local/open-weight models need no API key, but several checkpoints (Gemma, etc.) and the MILU
dataset itself are **gated on Hugging Face**. Request access to
[`ai4bharat/MILU`](https://huggingface.co/datasets/ai4bharat/MILU) (can take time — do this
first) and to any gated checkpoint you plan to run, then authenticate locally with
`huggingface-cli login` or `export HF_TOKEN=...` — this project relies on `huggingface_hub`'s
standard authentication, not a custom token variable.

## 6. Running an evaluation

```bash
python -m pipeline.run --models gemma-2-2b-it --limit 20   # smoke test
python -m pipeline.run --models gemma-3-27b-it              # one model, all 11 languages
python -m pipeline.run                                      # every config in configs/models/
```

or the equivalent wrapper: `bash scripts/run_evaluation.sh --models gemma-2-2b-it`.

`pipeline/run.py` loads all matching configs from `configs/models/*.yaml`, and for each one:
invokes the vendored `lm_eval` CLI as a subprocess under that config's `venv:` environment,
normalizes the output (`pipeline/store.py`), scores it (`pipeline/scorer.py`), and writes rows
into `results/results.db` (`pipeline/results_db.py`) plus an MLflow run
(`pipeline/mlflow_logging.py`). Note that the top few rows in §1 (Sarvam-M thinkmode,
Sarvam-30B, gpt-oss-20b thinkmode) needed additional targeted-retry/replication steps beyond
this one command to reach their final reported numbers — those auxiliary scripts aren't
included in this initial commit.

## 7. Reproducing results

1. `bash scripts/setup_vendor.sh`
2. Set up the venv(s) you need (§4 above), request MILU dataset access, configure `.env` (§5).
3. **Run the sanity gate first**: `python -m pipeline.run --models gemma-2-2b-it`. Expect
   ~28-29% on Gujarati, within about a point of the MILU paper's own published 29.25% for
   this checkpoint. Don't trust anything else until this passes.
4. Run your target model(s): `python -m pipeline.run --models <model_id>`.
5. `python scripts/export_results_tables.py` to get a CSV summary of everything you've run.

## 8. Thinking/reasoning configuration

Four model families in this roster have chat templates that can silently enable a "thinking"
mode, which — if not accounted for — breaks the standard loglikelihood MCQ-scoring protocol
entirely (measuring "how likely is the model to jump from an empty `<think>` tag straight to
an answer," a distribution the model was never trained to produce). This is the single most
consequential methodology finding in this project: uncorrected, it hid two of the strongest
models in the roster at the bottom of the table (Sarvam-M and gpt-oss-20b scored 48.02% and
30.45% under naive loglikelihood — bottom-half — vs. 82.44% and 72.22% corrected — now #3 and
#6 of the roster).

A fifth row below isn't a scoring-protocol correction at all: Qwen3.6-27B is the one model in
this roster evaluated with thinking **both** on and off, as two separate, equally-valid
production configs. The 16.84-point gap between them is a genuine capability difference, not a
bug — see `configs/models/qwen3.6-27b-llamacpp-gguf.yaml`'s own `notes:` field.

| Model | Thinking enabled | Reasoning config | Effect |
|---|---|---|---|
| Sarvam-M 24B | Yes, exercised | 0-shot generative, `max_new_tokens≈1536`, temperature 0 | 48.02% (broken) → **82.44%** (reported) |
| Qwen3.6-27B (thinking off) | No, forced off | `enable_thinking=False` patch + 0-shot generative | 37.82% (broken) → **67.00%** (reported) |
| Qwen3.6-27B (thinking on) | Yes, uncapped | 0-shot generative via llama.cpp GGUF, no stop sequence, `max_gen_toks=3072`, no reasoning-budget cap | **83.84%** (reported); +16.84 points over the thinking-off row, from that one setting |
| gpt-oss-20b | Yes (Harmony format) | 0-shot generative, no stop sequence, `max_gen_toks=8192` after malformed-item retry | 30.45% (broken, smoke-scale) → **72.22%** (reported) |
| Sarvam-30B | Yes, budget-capped | `llama.cpp --reasoning-budget` hard cap on the thinking phase | **72.99%** (reported) |
| Qwen3.8-Max | Yes, mandatory (can't be disabled) | `openrouter_reasoning_effort: minimal` — the lowest allocation the endpoint accepts, not a true disable | **89.67%** (reported); cost reflects mandatory reasoning tokens on every call, not directly comparable to fully-disabled-reasoning rows |

"Visible/final output handling" here means: extract the model's own stated final answer via a
regex filter, score only that — no hidden chain-of-thought is inspected beyond what each model
itself returned in its own output stream. Per-model detail (exact token budgets, why each
protocol was chosen, any library patch involved) is in that model's own config `notes:` field
in `configs/models/`.

## 9. Results and analysis

Running the pipeline yourself (§6/§7) produces:
- **`results/results.db`** (gitignored, machine-local) — the queryable results layer, one row
  per (run, breakdown_type, breakdown_key). See `pipeline/results_db.py`'s schema.
- **`data_raw_outputs/<model_id>/<run_timestamp>/raw_items.jsonl`** (gitignored, machine-local)
  — verbatim per-item model output, prompt hash, and generation config. Scoring is a fully
  separate offline pass over this store (`pipeline/scorer.py`) — never welded to inference —
  so any number can be re-derived or disputed without re-running the model.
- **`python scripts/export_results_tables.py`** — exports `results/results.db` into
  `results/tables/league_table.csv`, `by_language.csv`, `by_domain.csv` for anything readable
  outside SQLite.

None of these generated artifacts are included in this repo — only the code that produces
them.

## 10. Known limitations

- **Frontier APIs (GPT, Gemini, Claude) are cited, not run by this project.** The figures for
  those models come from each vendor's own published system card, not from an evaluation run
  in this repository.
- **`transformers` version drift, unresolved.** The default venv's actual installed
  `transformers` (`5.14.1`) doesn't match the version its own lock file pins and the sanity
  gate was validated against (`4.46.3`) — most loglikelihood-protocol rows ran under this
  venv. Not yet re-verified against the sanity gate. See `requirements/README.md`.
- **GPU-hour costs are illustrative, not billed.** Local-model cost figures apply a reference
  $1.80/hr rate to measured GPU-hours; the actual hardware is privately provisioned and
  shared, not metered per model or hour.
- **API/provider non-determinism.** DeepSeek V4-Flash's numbers reflect real API responses at
  the time of the run; provider-side model updates aren't pinned by us the way local
  checkpoints are (an access date substitutes for a revision hash).
- **Rate limits are real and provider-specific.** DeepSeek V4-Flash needed an explicit
  dispatch-rate limiter beyond `num_concurrent`.
- **The MILU dataset must be obtained separately** (HF-gated, distributed under CC BY 4.0 by
  AI4Bharat/IBM) — not bundled in this repo.
- **Hardware**: this project ran on 2× NVIDIA A100 80GB PCIe. Smaller GPUs can run the
  smaller models in the roster but not the 24B+ ones at full precision — see
  `requirements/README.md`.
- **Several real scoring/infrastructure bugs were found and fixed during this project**
  (a tokenizer round-trip bug, a silent RoPE-base truncation bug, a mass-timeout harness bug,
  a first-vs-last-match regex extraction bug) — each is disclosed in the relevant model
  config's own `notes:` field and in `patches/README.md`, not silently patched over.

## 11. Citation

If you use this pipeline, its configs, or its results, please cite this repository (see
`CITATION.cff`). **If you use the MILU benchmark itself, cite the original paper:**

```bibtex
@inproceedings{verma-etal-2025-milu,
    title = "{MILU}: A Multi-task {I}ndic Language Understanding Benchmark",
    author = "Verma, Sshubam and Khan, Mohammed Safi Ur Rahman and Kumar, Vishwajeet and
              Murthy, Rudra and Sen, Jaydeep",
    booktitle = "Proceedings of the 2025 Conference of the Nations of the Americas Chapter of
                  the Association for Computational Linguistics: Human Language Technologies
                  (Volume 1: Long Papers)",
    year = "2025",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.naacl-long.507/",
    doi = "10.18653/v1/2025.naacl-long.507",
}
```

## Credit

MILU is [AI4Bharat](https://ai4bharat.iitm.ac.in/) and IBM's benchmark, published at NAACL
2025 ([arXiv:2411.02538](https://arxiv.org/abs/2411.02538), code at
[github.com/AI4Bharat/MILU](https://github.com/AI4Bharat/MILU)). This repository runs their
benchmark as adopters — full credit to AI4Bharat and IBM for the dataset, task design, and
original evaluation. This project's own contribution is coverage of newer models, a
cost-per-rupee lens, and whatever findings fall out of that; it's complementary to their work,
not competitive with it. The evaluation harness is EleutherAI's
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), via AI4Bharat's
fork.

## License

This project's own code, configs, scripts, and docs are MIT-licensed — see `LICENSE`.
Vendored dependencies (`vendor/MILU`, `vendor/llama.cpp`, cloned by `scripts/setup_vendor.sh`,
not committed) remain under their own upstream MIT licenses — see `patches/README.md`. The
MILU dataset is distributed separately by AI4Bharat/IBM under CC BY 4.0 and is not bundled in
this repo; request access on Hugging Face (§5).
