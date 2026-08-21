# Models evaluated

One row per config in `configs/models/`. Checkpoint, revision, and access date come straight
from that config's own `hf_checkpoint`/`revision`/`api_endpoint`/`access_date` fields — the
config is the authoritative source, so this table can't drift from what was actually run.

| Model (config name) | Checkpoint | Revision / endpoint | Accessed | Quantization |
|---|---|---|---|---|
| `gemma-2-2b-it` *(sanity-gate check only, not a roster row)* | `google/gemma-2-2b-it` | `299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8` | 2026-07-27 | none |
| `sarvam-1` | `sarvamai/sarvam-1` | `e9607337286ddf496d4a2562b194e489dcf3feea` | 2026-07-28 | none |
| `qwen2.5-7b-instruct` | `Qwen/Qwen2.5-7B-Instruct` | `a09a35458c702b33eeacc393d103063234e8bc28` | 2026-07-28 | none |
| `gemma-3-4b-it` | `google/gemma-3-4b-it` | `093f9f388b31de276ce2de164bdc2081324b9767` | 2026-07-28 | none |
| `gemma-3-12b-it` | `google/gemma-3-12b-it` | `96b6f1eccf38110c56df3a15bffe176da04bfd80` | 2026-07-28 | none |
| `gemma-3-12b-it-int4` | `unsloth/gemma-3-12b-it-qat-int4-bnb-4bit` | `8ca2c0834b8376778202ab03623ce89ab4b9b5a2` | 2026-07-28 | int4 |
| `gemma-4-12b-it` | `google/gemma-4-12B-it` | `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` | 2026-07-28 | none |
| `gemma-3-27b-it` | `google/gemma-3-27b-it` | `005ad3404e59d6023443cb575daa05336842228a` | 2026-07-28 | none |
| `qwen3.6-27b-genthinkoff` | `Qwen/Qwen3.6-27B` | `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` | 2026-08-05 | none |
| `qwen3.6-27b-llamacpp-gguf` | `ggml-org/Qwen3.6-27B-GGUF` (Q4_K_M) | served locally via `llama.cpp` — see `configs/models/qwen3.6-27b-llamacpp-gguf.yaml` | 2026-08-22 | Q4_K_M (GGUF) |
| `sarvam-m-thinkmode-0shot-json` | `sarvamai/sarvam-m` | `01534a53c46f2788e392dbb3d994e0fa8f04d3fd` | 2026-07-31 | none |
| `llama-4-scout` | `RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16` | `e8b8a7ca92e62e476ae2e7d3169bf11c7caba485` | 2026-08-01 | w4a16 |
| `qwen3-vl-8b-instruct` | `Qwen/Qwen3-VL-8B-Instruct` | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | 2026-07-28 | none |
| `mistral-small-3.1-24b` | `mistralai/Mistral-Small-3.1-24B-Instruct-2503` | `68faf511d618ef198fef186659617cfd2eb8e33a` | 2026-07-28 | none |
| `gpt-oss-20b-thinkmode-vllm` | `openai/gpt-oss-20b` | `6cee5e81ee83917806bbde320786a8fb61efebee` | 2026-08-03 | mxfp4 |
| `phi-4-vllm` | `microsoft/phi-4` | `2db69c1c3e91a05d2c64a3185acfbaf36f744e25` | 2026-08-04 | none |
| `sarvam-30b-llamacpp-gguf` | `sarvamai/sarvam-30b-gguf` (Q4_K_M) | served locally via `llama.cpp` — see `configs/models/sarvam-30b-llamacpp-gguf.yaml` | 2026-08-07 | Q4_K_M (GGUF) |
| `deepseek-v4` | DeepSeek API (`deepseek-v4-flash`) | n/a — API model, no local checkpoint | 2026-08-01 | n/a |
| `qwen3.8-max` | OpenRouter (`qwen/qwen3.8-max`) | n/a — API model, no local checkpoint | 2026-08-06 | n/a |

Every row's chat template is applied (`apply_chat_template: true`). None use a custom sampling
seed — all runs use greedy decoding (`temperature=0`/`do_sample=false`), so there's no seed to
record. Each config is a single production run; the models needing a retry pass on top of the
base run (`gpt-oss-20b-thinkmode-vllm`, `sarvam-30b-llamacpp-gguf`) have that noted in their
own config's `notes:` field.

## Thinking/reasoning configuration

Four model families in this roster have a chat template that can silently enable a
"thinking"/reasoning mode by default — see `DEVELOPER_NOTES.md` §1 for the full investigation
of why this matters. This table records only what setting was applied and its effect on the
reported score, not any model's chain-of-thought content.

Qwen3.6-27B is a fifth, different case: not a scoring-protocol bug, but a deliberate
thinking-on-vs-off comparison, published as two separate rows — see `DEVELOPER_NOTES.md` §8.

| Model | Thinking | Setting | Reported effect |
|---|---|---|---|
| `sarvam-m-thinkmode-0shot-json` | On (template default; exercised for real) | `max_new_tokens=1536`, temperature 0, no stop sequence | 82.44% (generative, reported) vs. 48.02% (loglikelihood, excluded) |
| `qwen3.6-27b-genthinkoff` | Forced off (`enable_thinking=False`, patch in `patches/vendor-milu.diff`) | `generate_until`, 0-shot | 67.00% (reported) vs. 37.82% (loglikelihood with thinking silently on, excluded) |
| `qwen3.6-27b-llamacpp-gguf` | On, uncapped | `max_gen_toks=3072`, no stop sequence, no reasoning-budget cap | 83.84% (reported); +16.84 points over the thinking-off row above |
| `gpt-oss-20b-thinkmode-vllm` | On (Harmony format's `analysis` channel) | `max_gen_toks=3072`, no stop sequence (`milu_api_nostop`) | 72.22% (reported, after malformed-item retry) vs. 30.45% (loglikelihood smoke test, excluded) |
| `sarvam-30b-llamacpp-gguf` | On, hard-capped via `llama.cpp`'s native `--reasoning-budget` flag | `max_gen_toks=1900`, reasoning phase capped separately | 72.99% (reported) |

Every row's final answer is extracted from the model's own visible output via a regex filter
(`pipeline/scorer.py`); truncated/unparseable generations are logged as malformed rather than
scored as wrong.
