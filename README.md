# MILU Evaluation of Newer LLMs

A reproducible pipeline and results for evaluating 18 newer LLMs on MILU, AI4Bharat and IBM's Multi-task Indic Language Understanding benchmark, with measured GPU-hours next to every accuracy number.

[![License: MIT](https://img.shields.io/badge/license-MIT-56BF4F?style=flat-square&labelColor=1E281F)](LICENSE)
[![Dataset: MILU](https://img.shields.io/badge/data-MILU%20(CC%20BY%204.0)-FFD21E?style=flat-square&labelColor=1E281F)](https://huggingface.co/datasets/ai4bharat/MILU)
[![Report](https://img.shields.io/badge/report-sthanika.ai-56BF4F?style=flat-square&labelColor=1E281F&logo=firefox&logoColor=white)](https://sthanika.ai/research/milu-2026)

## What it measures

MILU is about 85,000 multiple-choice questions across 8 domains and 41 subjects in 11 Indic languages, drawn from Indian regional and state exams (NAACL 2025). The original paper evaluated 42+ LLMs current as of late 2024, with a top score (GPT-4o) of about 74%. This repo runs it as an adopter, not an author, and adds:

- **Newer models:** 2025–2026 checkpoints the paper predates, 18 models across all 11 languages.
- **A compute lens:** wall-clock GPU-hours alongside every accuracy number, so the accuracy-per-compute tradeoff is visible.
- **A scoring-protocol finding:** a bug that silently breaks four hybrid "thinking" model families under the standard loglikelihood protocol and, uncorrected, would have inverted two models' ranks.
- **A thinking on/off head-to-head:** Qwen3.6-27B shows a 16.84-point swing from that one setting alone.
- **A reusable pipeline:** model-major scheduling, a raw-output store fully decoupled from scoring, and a queryable results DB.

Every reported figure traces back to a raw model output through this pipeline. Report: [sthanika.ai](https://sthanika.ai/research/milu-2026)

## Quickstart

Local models need no API key, but the MILU dataset and several checkpoints (Gemma and others) are gated on Hugging Face. Request access to `ai4bharat/MILU` first (it can take time), then authenticate with `huggingface-cli login` or `export HF_TOKEN=...`.

```bash
git clone https://github.com/sthanika-ai/milu-llm-evaluation.git
cd milu-llm-evaluation
bash scripts/setup_vendor.sh          # clones + patches vendor/MILU, clones + builds vendor/llama.cpp
cp .env.example .env                  # fill in the keys your target model needs

python3.12 -m venv .venv-multimodal
source .venv-multimodal/bin/activate
pip install -r requirements/multimodal.freeze.txt
pip install mlflow tiktoken           # gap in the lock files, see requirements/README.md

# sanity gate first: expect ~28-29% on Gujarati, within about a point of the paper's 29.25%
python -m pipeline.run --models gemma-2-2b-it

python -m pipeline.run --models gemma-3-27b-it     # one model, all 11 languages
python -m pipeline.run                             # every config in configs/models/
python scripts/export_results_tables.py            # CSV summary of everything you've run
```

Notes:

- **Five venvs.** Different model backends need separate venvs. This is real complexity (see `requirements/README.md`), not something to collapse into one `requirements.txt`. Set up only the venv(s) your target model needs.
- **What `pipeline.run` does.** For each matching config in `configs/models/*.yaml`, it invokes the vendored `lm_eval` CLI under that config's venv, normalises the output (`pipeline/store.py`), scores it (`pipeline/scorer.py`), and writes `results/results.db` plus an MLflow run. `bash scripts/run_evaluation.sh --models <id>` is an equivalent wrapper.
- **Extra steps for some rows.** Sarvam-M (thinkmode), Sarvam-30B and gpt-oss-20b (thinkmode) needed targeted-retry and replication steps beyond this one command to reach their reported numbers. Those auxiliary scripts are not in the initial commit.
- **Configs.** `configs/models/*.yaml` has 19 curated configs (checkpoint, revision, backend, generation config), the 18 roster models plus the `gemma-2-2b-it` sanity gate. Each model's own `notes:` field documents quantization, protocol rationale and any library patches. See `configs/README.md` for the schema.
- **Not published.** The project's full investigation history (53 configs: smoke tests, staging runs, superseded pre-bugfix rows) is not published. The generated artifacts (`data_raw_outputs/`, `results/results.db`, `mlruns/`) are gitignored and machine-local, so only the code that produces them is included.

## Results

Overall accuracy plus measured compute. Compute is wall-clock GPU-hours on privately provisioned, shared hardware, and API models are marked as such.

| rank | model | protocol | overall | compute | note |
|---|---|---|---|---|---|
| 1 | Qwen3.8-Max (API) | generative, 0-shot (API) | 89.67% | API (hosted) | reasoning is mandatory on this endpoint, so compute is not comparable to fully-disabled-reasoning rows |
| 2 | Qwen3.6-27B (thinking on, llama.cpp GGUF) | generative, 0-shot | 83.84% | 74.90 GPU-hr | +16.84 points over thinking off |
| 3 | Sarvam-M 24B (thinkmode) | generative, 0-shot | 82.44% | 19.42 GPU-hr | corrected row |
| 4 | DeepSeek V4-Flash | generative, 0-shot (API) | 79.51% | API (hosted) | 0% malformed |
| 5 | Sarvam-30B (llama.cpp GGUF) | generative, 0-shot | 72.99% | 76.60 GPU-hr | most compute-intensive in the roster |
| 6 | gpt-oss-20b (thinkmode, vLLM) | generative, 0-shot | 72.22% | 9.15 GPU-hr | corrected row |
| 7 | Qwen3.6-27B (thinking off) | generative, 0-shot | 67.00% | 3.79 GPU-hr | corrected row |
| 8 | Mistral Small 3.1 24B | generative, 0-shot | 64.04% | 0.74 GPU-hr | |
| 9 | Gemma 3 27B | loglikelihood, 5-shot | 63.95% | 13.25 GPU-hr | |
| 10 | Llama 4 Scout (17B active / 109B MoE) | loglikelihood, 5-shot | 63.06% | 22.00 GPU-hr | |
| 11 | Gemma 4 12B | loglikelihood, 5-shot | 61.12% | 7.18 GPU-hr | |
| 12 | Gemma 3 12B | loglikelihood, 5-shot | 56.41% | 6.71 GPU-hr | |
| 13 | Gemma 3 12B INT4 | loglikelihood, 5-shot | 53.94% | 6.48 GPU-hr | |
| 14 | Qwen3-VL 8B Instruct | loglikelihood, 5-shot | 50.33% | 8.72 GPU-hr | multimodal architecture, text only used |
| 15 | Phi-4 14B | loglikelihood, 5-shot | 47.55% | 14.59 GPU-hr | |
| 16 | Gemma 3 4B | loglikelihood, 5-shot | 43.03% | 2.51 GPU-hr | |
| 17 | Qwen2.5 7B Instruct | loglikelihood, 5-shot | 41.81% | 7.92 GPU-hr | |
| 18 | Sarvam-1 2B | loglikelihood, 5-shot | 28.63% | 1.47 GPU-hr | |

**The thinking-mode finding.** Four model families have chat templates that can silently enable "thinking", which breaks standard loglikelihood MCQ scoring (it measures how likely the model is to jump from an empty `<think>` tag straight to an answer, a distribution it was never trained to produce). Uncorrected, it hid two of the strongest models near the bottom of the table.

| model | thinking | reasoning config | effect |
|---|---|---|---|
| Sarvam-M 24B | yes | 0-shot generative, `max_new_tokens`≈1536, temperature 0 | 48.02% (broken) → 82.44% |
| Qwen3.6-27B (off) | forced off | `enable_thinking=False` patch + 0-shot generative | 37.82% (broken) → 67.00% |
| Qwen3.6-27B (on) | yes, uncapped | 0-shot generative via llama.cpp GGUF, `max_gen_toks=3072`, no budget cap | 83.84% (+16.84 over off) |
| gpt-oss-20b | yes (Harmony) | 0-shot generative, `max_gen_toks=8192` after malformed-item retry | 30.45% (broken, smoke-scale) → 72.22% |
| Sarvam-30B | yes, budget-capped | llama.cpp `--reasoning-budget` hard cap | 72.99% |
| Qwen3.8-Max | yes, mandatory | `openrouter_reasoning_effort: minimal`, the lowest the endpoint accepts | 89.67% |

Only each model's own stated final answer is extracted (regex filter) and scored. The Qwen3.6-27B gap is a genuine capability difference, not a bug.

Caveats:

- Frontier APIs (GPT, Gemini, Claude) are cited from vendor system cards, not run here.
- The default venv's installed `transformers` (5.14.1) does not match the version its lock file pins and the sanity gate was validated against (4.46.3). Most loglikelihood rows ran under it and have not yet been re-verified. See `requirements/README.md`.
- GPU-hours are indicative, not a controlled benchmark.
- API results reflect real responses at run time. Provider-side updates are not pinned, so an access date substitutes for a revision hash. DeepSeek V4-Flash needed an explicit dispatch-rate limiter.
- Hardware was 2× NVIDIA A100 80GB PCIe. Smaller GPUs can run the smaller models but not the 24B+ ones at full precision.
- Several bugs were found and fixed (a tokenizer round-trip bug, a silent RoPE-base truncation, a mass-timeout harness bug, a first-vs-last-match regex extraction bug). Each is disclosed in the relevant config's `notes:` field and in `patches/README.md`.

Full report: [sthanika.ai](https://sthanika.ai/research/milu-2026)

## Citation

Cite this repository as in [`CITATION.cff`](CITATION.cff). If you use the MILU benchmark itself, cite the original paper:

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

## License

Code, configs, scripts and docs are MIT, see [LICENSE](LICENSE). Vendored dependencies (`vendor/MILU`, `vendor/llama.cpp`, cloned by `scripts/setup_vendor.sh`, not committed) keep their upstream MIT licenses, see `patches/README.md`. The MILU dataset is distributed separately by AI4Bharat/IBM under CC BY 4.0 and is not bundled.

## Related

- [MILU](https://github.com/AI4Bharat/MILU) (AI4Bharat and IBM, NAACL 2025, [arXiv:2411.02538](https://arxiv.org/abs/2411.02538)): the benchmark. Full credit to AI4Bharat and IBM for the dataset, task design and original evaluation. This repo is complementary to their work, not competitive with it.
- [`ai4bharat/MILU`](https://huggingface.co/datasets/ai4bharat/MILU): the dataset (gated)
- EleutherAI's lm-evaluation-harness, via AI4Bharat's fork: the evaluation harness
- Companion work from sthanika-ai: [CodeMixTax](https://github.com/sthanika-ai/CodeMixTax), [token_fertility](https://github.com/sthanika-ai/token_fertility)
- Site: [sthanika.ai](https://sthanika.ai)
