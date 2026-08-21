# Methodology notes: scoring corrections and known limitations

This document records the scoring-protocol corrections and infrastructure fixes behind this
project's reported results. Each entry states the symptom, the root cause, the fix, and the
before/after numbers — and points at the committed file (`configs/models/`, `patches/`,
`pipeline/`) that reproduces it, so the fix isn't just a claim.

**Summary:**

- Four models (Sarvam-M, Qwen3.6-27B, gpt-oss-20b, Mistral Small 3.1) needed a corrected
  scoring protocol because a hidden chat-template behavior or tokenizer limitation broke
  standard loglikelihood scoring. Uncorrected, two of them would have ranked in the bottom
  half of the roster instead of the top (§1).
- Sarvam-30B and Qwen3.8-Max each needed a chain of infrastructure fixes — inference-serving
  bugs for the former, mandatory-reasoning cost accounting for the latter — before their
  results were trustworthy (§2, §3).
- Cost-per-question figures account for loglikelihood's 4x-per-item forward-pass cost, so a
  loglikelihood row and a generative row are comparable on the same basis (§6).
- One open item affects every loglikelihood result run under the default environment: a
  `transformers` version drift, not yet re-verified against the sanity gate (§7).
- Qwen3.6-27B is evaluated both with thinking forced off (67.00%) and left on (83.84%) — not a
  bug fix, a genuine 16.84-point capability delta from one setting, published as two rows (§8).

## 1. Hybrid-thinking chat templates

Four models broke standard 5-shot loglikelihood scoring because their chat template opens a
hidden "thinking" block by default. Each was caught by an anomalously low score, confirmed by
inspecting the raw rendered prompt.

### 1.1 Sarvam-M 24B

Every prompt ended in an unclosed `<think>` tag right before the scored continuation, so
loglikelihood was measuring a model jumping from an empty think-tag straight to an answer with
zero reasoning tokens.

- **Fix:** score 0-shot, generative, thinking allowed. `pipeline/ingest_thinkmode_run.py`
  scores truncated/unparseable generations as wrong rather than excluding them (an earlier pass
  excluded them, which inflated the score to 85.97%).
- **Result: 82.44% overall** (vs. 48.02% under loglikelihood). Config:
  `configs/models/sarvam-m-thinkmode-0shot-json.yaml`.

### 1.2 Qwen3.6-27B

Same unclosed-`<think>` pattern. Its loglikelihood English score was lower than several of its
own Indic-language scores — the reverse of every other model in the roster, a clear sign of a
measurement error rather than a genuine result.

- **Fix:** an `enable_thinking=False` patch to `apply_chat_template()`, captured in
  `patches/vendor-milu.diff` (`huggingface.py`, `vllm_causallms.py`), plus switching from
  loglikelihood to `generate_until` scoring.
- **Result: 67.00% overall, 0.004% malformed.** Config:
  `configs/models/qwen3.6-27b-genthinkoff.yaml`.
- With thinking off, generative scoring ran ~10x faster than loglikelihood for this
  architecture — both protocols pay the same per-item cost, but loglikelihood needs 4 forward
  passes per item (one per MCQ option) versus one short generation.

### 1.3 gpt-oss-20b

OpenAI's Harmony format opens an `analysis`-channel reasoning block before the `final`-channel
answer. The standard API task's stop-on-`"\n\n"` condition was cutting generation off
mid-reasoning, regardless of token budget.

- **Fix:** a stop-condition-free task variant, `milu_api_nostop`, committed under
  `patches/vendor-milu-new-files/lm_eval/tasks/milu/`; a retry pass on malformed items at a
  higher `max_gen_toks` (3072 → 8192) recovered most of the rest.
- **Result: 72.22% overall, 1.86% malformed** — the remainder are degenerate repetition loops
  that don't converge regardless of token budget. Config:
  `configs/models/gpt-oss-20b-thinkmode-vllm.yaml`.

### 1.4 Mistral Small 3.1 24B

Two independent tokenizer-level issues, not a hidden reasoning mode:

- **Issue A — tokenizer round-trip.** vLLM's `MistralTokenizer` strips special tokens on
  decode, and re-encoding the decoded string doesn't reproduce them — a mismatch with how the
  harness re-tokenizes chat-templated text. This only breaks loglikelihood scoring, so this
  model is scored 0-shot generative instead.
- **Issue B — RoPE base silently wrong.** vLLM registers this architecture as `LlamaForCausalLM`, whose
  RoPE lookup reads a flat `rope_theta` attribute; `transformers>=5.x` moved this checkpoint's
  value into a nested `rope_parameters` dict, so the flat lookup silently returned the wrong
  default (10000 instead of 1,000,000,000) with no error. Fixed by patching the installed
  `vllm` package to fall back to `rope_parameters["rope_theta"]` when the flat attribute is
  missing — this changes a pip-installed dependency rather than the vendored harness, so it
  isn't part of this repo's `patches/` overlay; re-apply it by hand to reproduce this model.
- **Result: 64.04% overall, 14 malformed (0.018%).** Config:
  `configs/models/mistral-small-3.1-24b.yaml`.

**Takeaway:** a wrong scoring protocol can invert rank, not just shift a number. Sarvam-M and
gpt-oss-20b scored 48.02%/30.45% under naive loglikelihood — bottom half of the roster.
Corrected, they score 82.44%/72.22% — top of the roster. Check whether a model's chat template
silently enables a reasoning mode before trusting a loglikelihood score.

## 2. Sarvam-30B: inference-serving fixes

Sarvam-30B (a 32B-total/6-active-per-token MoE) needed the most infrastructure work in the
roster:

1. **Repetition-loop collapse** under `transformers==5.14.1` (this project's default venv).
   **Fix:** run under `.venv-sarvam` (`transformers==4.57.1`) instead — see
   `requirements/README.md`.
2. **First-match instead of last-match answer extraction** — this model's reasoning traces
   mention and rule out multiple letters before concluding. **Fix:** a `group_select: -1` task
   variant, scoped to this model only.
3. **The `hf` backend was too slow at full scale** (21-35s/item); full bf16 doesn't fit one
   A100-80GB, and a community INT4/AWQ quantization was ~5x slower. **Fix:** serve the official
   GGUF release (Q4_K_M) via `llama.cpp` — 10-18x faster (1.4-2.75s/item).
4. **A mass request-timeout bug**, surfaced at two different scales: the timeout clock starts
   when a request is created, not when it gets a connection slot, so a backlog past
   `max_retries × timeout_seconds` fails even when the server is healthy. **Fix:** raise
   `timeout_seconds` per run.
5. **Malformed items scaled with reasoning-chain length by domain** (Social Sciences 42-43%
   malformed vs. Science 16-23%). **Fix:** `llama-server`'s `--reasoning-budget` flag, capping
   the thinking phase — recovered 99.9% of the remaining malformed pool.

**Result: 72.99% overall, 0.03% malformed.** Config:
`configs/models/sarvam-30b-llamacpp-gguf.yaml`.

**Cost: $137.87 (76.60 GPU-hr) → ₹165.03/1,000 questions** — mostly the two malformed-item
retry passes (36.08 GPU-hr) plus the initial 8-language batch run (32.47 GPU-hr).

The reported accuracy (72.99%) is lower than an intermediate reading taken mid-investigation
(83.35%, Gujarati only) that excluded malformed items from its denominator — a biased, easier
subset, since malformed items skew toward harder questions. Once the reasoning-budget fix
recovered nearly all of them, accuracy reflects the full dataset instead.

## 3. Qwen3.8-Max: mandatory reasoning and infrastructure fixes

Qwen3.8-Max (accessed via OpenRouter, config `configs/models/qwen3.8-max.yaml`) scores
**89.67% overall, 1.05% malformed** — the highest accuracy and smallest English-Indic gap in
the roster. Reaching a valid result needed:

1. **Reasoning is mandatory on this endpoint**, not merely default-on — no `"none"` option
   exists, and an explicit disable request returns a 400. `openrouter_reasoning_effort:
   minimal` is the real floor, not a true disable, so this row's cost isn't directly
   comparable to fully-disabled-reasoning rows like DeepSeek V4-Flash.
2. **`max_gen_toks` raised 100 → 500** to leave headroom for mandatory reasoning tokens (a
   100-token cap left no room for any answer content).
3. **A harness crash, not just a malformed item.** OpenRouter returns `content: null` when the
   model exhausts its budget before answering. **Fix:** coerce `None` to `""` in
   `openai_completions.py`, captured in `patches/vendor-milu.diff`.
4. **The same mass-timeout congestion bug as §2**, here in the OpenRouter path. **Fix:** raise
   `num_concurrent` 4→30 and lower `max_retries` 8→2.

**Cost: $156.60 (real metered spend) → ₹187.45/1,000 questions** — the highest in the roster,
driven by mandatory reasoning tokens on every call.

## 4. Rows excluded from the reported results

| Row | Reason |
|---|---|
| Sarvam-M (loglikelihood, 48.02%) | Superseded by the corrected result (82.44%), same model, corrected protocol — §1.1. |
| Qwen3.6-27B (loglikelihood, 37.82%) | Superseded by the corrected result (67.00%) — §1.2. |
| gpt-oss-20b (loglikelihood, 30.45%) | Smoke test only, never run at full scale. |
| `gemma-2-2b-it` | Sanity-gate check only, not a roster run. |
| Sarvam-30B early configs | Pre-fix investigation runs, superseded by 72.99% — §2. |
| Qwen3.8-Max early configs | Pre-fix runs at `max_gen_toks=100`, superseded by 89.67% — §3. |
| Qwen3.6-27B thinking-on via `hf-causal-multimodal` | Manually killed for being too slow (batch_size=1, 51-291s/item); superseded by the llama.cpp GGUF path, 10-18x faster — §8. |

## 5. Other infrastructure fixes

**Llama 4 Scout (MoE, `configs/models/llama-4-scout.yaml`).** vLLM's bitsandbytes path doesn't
support MoE models; switched to RedHatAI's `w4a16` (compressed-tensors INT4) checkpoint.
Loading it needed three local patches to the installed `vllm` package for
transformers-5.x/vLLM-0.11.0 version-skew (a config field, a tokenizer-caching method, and the
same `rope_parameters` issue as §1.4) — these modify a pip-installed dependency, not the
vendored harness, so they aren't part of this repo's `patches/` overlay. Odia needed a higher
`max_model_len` (5,120 vs. 2,048) since its longest prompt runs to 4,270 tokens.

**Qwen3-VL-8B (`configs/models/qwen3-vl-8b-instruct.yaml`).** Auto-batch-size detection doesn't
fire on the `hf-causal-multimodal` backend for this architecture; ran with a manually fixed
`batch_size` instead of patching vLLM.

**DeepSeek V4-Flash (API).** An elevated timeout rate that scaled with dispatch order turned
out to be an undocumented provider throttle on sustained dispatch rate. **Fix:** an explicit
rate limiter pacing dispatch, plus targeted retries of the affected items.

## 6. Cost computation methodology

Loglikelihood (5-shot) needs 4 forward passes per question (one per MCQ option); generative
(0-shot) needs 1. GPU-hours are the real wall-clock time of the whole run (the harness runs all
4 loglikelihood requests per question before returning — there's no separate per-request
timer), and the "/1,000 questions" denominator is the real question count, not the request
count. Loglikelihood's 4x cost is already inside the measured GPU-hours, so dividing by the
real question count is what makes a loglikelihood row and a generative row comparable.
Arithmetic check: $1.33 × 95.29 ÷ 79.608 = ₹1.59/1,000 (Mistral Small 3.1, matches the reported
figure).

**Known gap:** GPU-hours for local models are reconstructed manually from run-log timestamps,
not computed by an automated script. Correct for every model checked, but not yet enforced by
code.

## 7. Open items

- **Environment drift, unresolved.** The default venv has `transformers==5.14.1` installed,
  despite its lock file pinning `4.46.3` — the version the sanity gate was validated against.
  This hasn't been re-verified against the sanity gate under the actually-installed version.
  Given the confirmed `transformers`-5.x version-skew issues elsewhere in this project (§1.4,
  §5), treat any loglikelihood result from the default venv as carrying this caveat until the
  sanity gate is re-run under `transformers==5.14.1`.
- **Domain-level breakdowns for the corrected Sarvam-M and Qwen3.6-27B runs are stale** — the
  original 8-domain breakdown for both was measured under the superseded loglikelihood
  protocol and shouldn't be reused under the corrected model names until recomputed.

## 8. Qwen3.6-27B: thinking on vs. thinking off

The corrected §1.2 result (67.00%, thinking forced off) isn't the whole story for this model. A
separate full 11-language run with thinking left on — served via the official
`ggml-org/Qwen3.6-27B-GGUF` release (Q4_K_M) through llama.cpp, the same infrastructure pattern
as Sarvam-30B (§2) — scores dramatically higher.

**Result: 83.84% overall, all 11 languages, 79,608 items, 3.92% malformed (3,123 items).**
Config: `configs/models/qwen3.6-27b-llamacpp-gguf.yaml`.

| Language | Accuracy | Items | Malformed |
|---|---|---|---|
| English | 86.09% | 13,535 | 1.85% |
| Bengali | 85.54% | 6,637 | 3.01% |
| Kannada | 85.19% | 6,234 | 4.04% |
| Hindi | 84.77% | 14,831 | 3.57% |
| Telugu | 83.55% | 7,304 | 5.26% |
| Tamil | 83.23% | 6,372 | 4.44% |
| Gujarati | 82.89% | 4,826 | 4.58% |
| Marathi | 82.14% | 6,924 | 4.84% |
| Odia | 80.83% | 4,525 | 5.13% |
| Malayalam | 80.60% | 4,321 | 5.99% |
| Punjabi | 80.42% | 4,099 | 4.32% |

**+16.84 points over the thinking-off row (67.00%) — the largest thinking-on-vs-off gap
measured for any model in this roster.** Unlike the §1 hybrid-thinking family, this isn't a
scoring-protocol bug: both rows use the correct 0-shot generative protocol, correctly scored.
The gap is a genuine capability difference between letting this model reason and forcing it
not to — which is why both rows are published side by side rather than one superseding the
other (contrast with §4's excluded rows, which are broken measurements, not legitimate
comparisons).

Getting there needed one earlier attempt abandoned: an `hf-causal-multimodal` run with thinking
on was manually killed for being too slow (batch_size=1, 51-291s/item observed, 36/100 items in
58 minutes) before completing even a full-language pass. **Fix:** the same GGUF-via-llama.cpp
path already proven for Sarvam-30B (§2) — this checkpoint (`qwen35`/`qwen35moe` architecture)
was already natively supported by the llama.cpp build compiled for that model, no rebuild
needed.

**Malformed items are unmitigated reasoning-budget exhaustion, not a bug.** Confirmed directly
via the real per-request `finish_reason` log for Gujarati: 212 of 4,817 requests (4.4%) hit
`finish_reason: "length"` against the `max_gen_toks=3072` cap, accounting for 96% of that
language's 221 malformed items. No `--reasoning-budget` cap (llama.cpp's own flag — see §2 fix
#5) was applied for this run; doing so would likely recover most of these, matching the 99.9%
recovery Sarvam-30B saw from the same fix, but hasn't been attempted for this model yet.

**Cost: $134.83 (74.90 GPU-hr) → ₹161.39/1,000 answered questions.** Reconstructed from real
run-log timestamps, same methodology as §6.
