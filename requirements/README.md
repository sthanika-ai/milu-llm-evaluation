# Environments

This project uses **five separate virtual environments**, not one, because different model
architectures and serving backends need conflicting `transformers`/`torch`/`vllm` versions on
the same box. This is a real, load-bearing part of the methodology, documented at the top of
`pipeline/run.py` (`VENV_BIN` dict) — not incidental complexity. Mixing venvs up is a real
failure mode: a `transformers` version mismatch on one model produced a degenerate
repetition-loop output that initially looked like a model bug.

| Venv | Used for (`venv:` field in `configs/models/*.yaml`) | Real installed versions |
|---|---|---|
| `.venv` (default) | 8 configs — Gemma 2, Qwen 2.5, Sarvam-1, Sarvam-30B (early bf16/AWQ investigation configs), Sarvam-M loglikelihood | `transformers==5.14.1`, `torch==2.5.1+cu124` — **see drift note below** |
| `.venv-multimodal` | 12 configs — Gemma 3/4, Qwen3-VL, Mistral Small 3.1, Phi-4, gpt-oss-20b (`hf-causal-multimodal` backend) | `transformers==5.14.1`, `torch==2.5.1+cu124` |
| `.venv-sarvam` | 7 configs — Sarvam-30B bf16/GGUF investigation configs, Sarvam-M(-thinkmode) | `transformers==4.57.1`, `torch==2.5.1+cu124` (pinned specifically because Sarvam-30B's custom modeling code needs this exact version) |
| `.venv-vllm2` | 9 configs — Llama 4 Scout, later Phi-4/Mistral/gpt-oss vLLM configs | `transformers==5.14.1`, `torch==2.8.0+cu128`, `vllm==0.11.0` |
| `.venv-vllm` | 1 config (`sarvam-m-thinkmode-0shot-json.yaml`, `venv: vllm`) | `transformers==4.46.3`, `torch==2.5.1+cu124`, `vllm==0.6.6.post1` — **see note below** |

**Note on `.venv-vllm`:** `pipeline/run.py`'s current `VENV_BIN` dict defines
`default`/`multimodal`/`vllm2`/`sarvam` — it has no `"vllm"` entry, even though one config
(`sarvam-m-thinkmode-0shot-json.yaml`) sets `venv: vllm`. That's expected: this config doesn't
go through `pipeline/run.py`'s dispatcher at all — it's run directly via internal tooling that
drives vLLM with its own checkpoint/revision flags and doesn't consult `VENV_BIN`. See that
config's own `notes:` field for what it needed.

## Known drift

`.venv`'s own `requirements.lock.txt` (repo root) pins `transformers==4.46.3` — the version
the sanity gate was validated against. The actually installed version in that venv, confirmed
by running `.venv/bin/python -c "import transformers; print(transformers.__version__)"`, is
`5.14.1`. Most of the roster's 5-shot loglikelihood results ran under `venv: default`, and this
drift has not yet been re-verified against the sanity gate. Given confirmed
`transformers`-5.x version-skew issues found elsewhere in this project (a RoPE-base config
lookup and a tokenizer-caching method both moved location between versions — see the affected
configs' own `notes:` fields), treat any loglikelihood-protocol number from `venv: default` as
carrying this caveat until the sanity gate is re-run under `transformers==5.14.1` and confirmed
to still pass.

The `*.freeze.txt` files in this directory are real `pip freeze` snapshots of the five venvs'
actual installed state, captured directly from the live environments, so this drift is visible
rather than papered over by a lock file nobody has re-validated against what's really
installed.

## Recreating an environment

```bash
python3.12 -m venv .venv-multimodal
source .venv-multimodal/bin/activate
pip install -r requirements/multimodal.freeze.txt
```

Repeat for the other four, substituting the venv name and freeze file. The root
`requirements.lock.txt` is the original, hand-maintained lock file for `.venv` (the "default"
environment) — kept as-is; `requirements/default.freeze.txt` is a snapshot of what's actually
installed there today (see drift note above), not a replacement for it.

## A known gap: packages `pipeline/*.py` needs that aren't in `requirements.lock.txt`

`pipeline/mlflow_logging.py` imports `mlflow`, and `pipeline/token_cost_estimate.py` imports
`tiktoken` — neither appears in the root `requirements.lock.txt`. Whatever venv you run
`python -m pipeline.run` / the `pipeline.*` scripts from (not necessarily one of the five
model-serving venvs above — the orchestrator itself just needs `pyyaml`, `huggingface_hub`,
`mlflow`, `pandas`, and, for the cost-estimate scripts, `tiktoken`) needs these installed
separately, e.g. `pip install mlflow tiktoken`.

## Hardware this was run on

Ubuntu 22.04.5 LTS, Python 3.12.9, 2× NVIDIA A100 80GB PCIe (driver 550.54.15), CUDA 12.4
toolchain (per `torch==2.5.1+cu124`) for most venvs, CUDA 12.8 for `.venv-vllm2`
(`torch==2.8.0+cu128`). GPU work in this project was pinned to a single non-primary GPU via
`CUDA_VISIBLE_DEVICES` on a shared box (see `pipeline/run.py --cuda-visible-devices`), except
where a run was deliberately split across both GPUs in parallel for throughput.
