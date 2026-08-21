# Troubleshooting / known issues

Scannable index of real problems found and fixed during this project. Every entry links to
the full write-up in `DEVELOPER_NOTES.md` — this page is a map, not a replacement for it.

## "My score is suspiciously low (near or below random chance) for a reasoning/thinking model"

**You've hit the hybrid-thinking scoring bug.** Some chat templates (Sarvam-M, Qwen3.6-27B,
gpt-oss-20b all confirmed) default to opening a "thinking" block even when never explicitly
requested. Scoring loglikelihood right after a bare, unclosed `<think>\n` (or equivalent
channel marker) measures "how likely is the model to jump straight from zero reasoning to a
final answer" — not a distribution the model was trained to produce. Symptom: a score near or
below the 25% MCQ floor, or a model scoring *worse* in English than in low-resource languages
(backwards from every other model). **Fix:** switch to generative (0-shot) scoring with
thinking either explicitly enabled (let it reason) or explicitly disabled
(`enable_thinking=False`, see `patches/vendor-milu.diff`). Full writeup:
`DEVELOPER_NOTES.md` §1.

## "Generation gets cut off mid-reasoning / high malformed rate on a thinking model"

The standard MILU API task template stops generation on the first `"\n\n"` — thinking traces
routinely contain a blank line before the real answer, so this cuts generation off before an
answer is ever reached. **Fix:** use the `*_api_nostop` task variant (`until: []`, no stop
sequence) instead of the standard `*_api` task — see `patches/README.md`. The same task
variant is also used for Sarvam-30B (§2). Full writeup: `DEVELOPER_NOTES.md` §1.3.

## "Regex answer extraction grabs the wrong letter"

The default extraction filter takes the **first** regex match. A reasoning model that
mentions and rules out several candidate letters before its real final answer
("...so B... actually D... final answer D") gets scored on the first-mentioned letter, not
the real one. **Fix:** use a `group_select: -1` (last-match) task variant instead of the
default `group_select: 0`. Full writeup: `DEVELOPER_NOTES.md` §2 fix #2.

## "Score is near/below random chance and skews toward the last-listed choice, non-thinking model"

Check whether your tokenizer's `apply_chat_template()` can actually round-trip through this
harness's string-render-then-re-tokenize loglikelihood path. Mistral's tokenizer (vLLM's
`MistralTokenizer` wrapper) silently strips special tokens on decode, so the harness ends up
scoring an unwrapped, un-templated string without erroring. Confirm by decoding raw token ids
and checking whether re-encoding reproduces the same token count. **Fix:** this is a hard
architectural mismatch for loglikelihood specifically — switch that model to the 0-shot
generative protocol instead (which doesn't need the string round-trip). Full writeup:
`DEVELOPER_NOTES.md` §1.4, Issue A.

## "A model's positional encoding seems wrong, no crash, no warning"

Check whether the architecture reads `rope_theta` via a bare `getattr(config, "rope_theta",
10000)`. `transformers>=5.x` moved `rope_theta` into a nested `rope_parameters` dict for some
checkpoints — the old flat-attribute fallback silently returns the wrong default (10000)
instead of the real value (can be 1,000,000,000+) with no error. Affects any vLLM
architecture that shares `llama.py` (Mistral, Phi-4, Llama 4 Scout all hit variants of this).
**Fix:** fall back to `rope_parameters["rope_theta"]` when the flat attribute is missing.
Full writeup: `DEVELOPER_NOTES.md` §1.4, Issue B, and §5 (Llama 4 Scout).

## "A large batch run fails all at once near a timeout boundary, but the server logs show the requests succeeding"

The harness's `local-chat-completions`/API backend creates every request task in one
synchronous burst regardless of dataset size, but each task's timeout clock starts at
*creation*, not when it actually gets a connection from the (much smaller) concurrency pool.
If your dataset's real drain time exceeds `max_retries × timeout_seconds`, every
still-queued task fails simultaneously, even though the server is healthy and finishes those
exact requests moments later. **Fix:** raise `timeout_seconds` to comfortably exceed the
run's expected wall-clock time, and don't rely on `max_retries` to compensate (retrying
doesn't drain the backlog faster). This has to be re-derived per run — it doesn't scale
automatically with dataset size. Full writeup: `DEVELOPER_NOTES.md` §2 fix #4.

## "An API run's failure/timeout rate is highest for whichever language dispatched first"

Likely an undocumented server-side throttle on sustained request *dispatch rate*, separate
from any concurrent-connection limit. `num_concurrent` bounds requests in flight, not how
fast new ones are issued as slots free up. **Fix:** add an explicit rate limiter pacing
dispatch itself, independent of the concurrency cap. Full writeup: `DEVELOPER_NOTES.md` §5
(DeepSeek V4-Flash).

## "Malformed/truncated items cluster in specific domains, is that a bug?"

Not necessarily. In this project, Social Sciences / Law & Governance / Arts & Humanities
consistently showed a much higher malformed rate than Science / Health & Medicine, across
multiple languages and models — a real difference in how much reasoning discursive domains
need, not noise. Before assuming a bug, check whether raising the raw token budget alone
helps (diminishing returns) versus a dedicated reasoning-budget cap (see `DEVELOPER_NOTES.md`
§2 fix #5) — and be aware that excluding malformed items from your accuracy denominator biases
the result upward if malformed items correlate with difficulty (they usually do). Full
writeup: `DEVELOPER_NOTES.md` §2 fix #5 and its "Methodological note."

## Environment / dependency issues

- **`transformers` version drift.** The root `requirements.lock.txt` pins
  `transformers==4.46.3` for the default venv (`.venv`), the version the sanity gate was
  validated against — but the actually-installed version there is `5.14.1` (confirmed live,
  see `requirements/README.md`). This is a disclosed, unresolved risk, not something this
  repo silently fixes for you — re-run the sanity gate under whatever version is actually
  installed before trusting a loglikelihood number from that venv.
- **Missing packages in `requirements.lock.txt`.** `mlflow` and `tiktoken` are imported by
  `pipeline/mlflow_logging.py` / `pipeline/token_cost_estimate.py` but aren't in the lock
  file. `pip install mlflow tiktoken` into whichever venv runs the orchestrator.
- **`venv: vllm` config with no matching `VENV_BIN` entry.** One config
  (`sarvam-m-thinkmode-0shot-json.yaml`) sets `venv: vllm`, but `pipeline/run.py`'s
  `VENV_BIN` dict doesn't define a `"vllm"` key. Not a blocker: that config's own `notes:`
  field explains that it bypasses `pipeline/run.py` entirely — see `requirements/README.md`.
- **mxfp4 quantization crash on `transformers==5.14.1` + `torch==2.5.1`.** `gpt-oss-20b`'s
  native mxfp4 hits `Mxfp4Config.validate_environment()` calling
  `torch.accelerator.current_accelerator()`, which doesn't exist in this pinned torch version.
  Workaround: a forward-compatible shim, or load with `dequantize_mxfp4` to plain bf16.
  Full writeup: `DEVELOPER_NOTES.md` §1.3.
