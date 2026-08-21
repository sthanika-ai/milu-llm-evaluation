# `vendor/MILU` local patch set

`vendor/MILU` is a shallow clone of
[`AI4Bharat/MILU`](https://github.com/AI4Bharat/MILU) (itself a fork of
[EleutherAI's `lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness)
with MILU's tasks pre-registered), pinned at commit
`7d8e6c9102bf44ae9f9ee84cfabefb4cb8fa2e88`.

That commit alone is not sufficient to reproduce this project's results. This project also
carries a set of local modifications on top of it — 8 modified files and 31 new files,
captured here so a fresh clone of `AI4Bharat/MILU@7d8e6c9` plus this overlay reproduces the
harness state every published config in `configs/models/` actually runs under.

## Contents

- **`vendor-milu.diff`** — unified diff of the 8 modified tracked files:
  - `lm_eval/api/model.py`, `lm_eval/models/__init__.py`,
    `lm_eval/models/api_models.py`, `lm_eval/models/hf_vlms.py`,
    `lm_eval/models/huggingface.py`, `lm_eval/models/openai_completions.py`,
    `lm_eval/models/utils.py`, `lm_eval/models/vllm_causallms.py`
- **`vendor-milu-new-files/`** — verbatim copies of the new files, at the same relative
  path they need to land at under `vendor/MILU/`. 28 of the 31 are committed here (every one
  referenced by a `tasks:` field in `configs/models/`); 3 investigation-only task variants
  exist on disk but aren't part of the committed overlay — see `.gitignore`.
  - `lm_eval/models/hf_causal_multimodal.py` — new backend for natively-multimodal
    architectures (Gemma 3/4, Qwen3-VL, Mistral Small 3.1) that keeps MILU's text-only
    loglikelihood scoring on the plain-causal codepath while loading via
    `AutoModelForImageTextToText`.
  - `lm_eval/tasks/milu/utils_milu_api.py` + every committed `milu_<Language>_api*.yaml` /
    `_default_template_api*.yaml` / `_milu_api*.yaml` — the generative (`generate_until`)
    scoring path for API/closed models and for the hybrid-thinking local models that had to
    be moved off loglikelihood scoring (see below). No public task config existed for this
    before this project wrote one.

## Why each piece exists

| Change | Motivated by |
|---|---|
| `enable_thinking=False` patch in `huggingface.py` / `vllm_causallms.py` | Some chat templates (Sarvam-M, Qwen3.6-27B) default to thinking-mode-on even when never requested, which silently breaks loglikelihood scoring |
| `milu_*_api_nostop.yaml` variants (`until: []`, no stop sequence) | The standard API task's `"\n\n"` stop sequence cuts hybrid-thinking models off mid-reasoning before they reach an answer |
| `hf_causal_multimodal.py` | Natively-multimodal architectures need `AutoModelForImageTextToText`, not `AutoModelForCausalLM` |
| `api_models.py` / `openai_completions.py` changes (rate limiting, usage logging) | DeepSeek V4-Flash's dispatch-rate throttle needed explicit pacing; real per-request token usage logging feeds `pipeline/compute_real_cost.py` |

Model-specific rationale for each patch (why a given model needed it, what protocol it enables)
is in that model's own config `notes:` field in `configs/models/`.

## Applying this patch set

Handled automatically by [`scripts/setup_vendor.sh`](../scripts/setup_vendor.sh):

```bash
git clone https://github.com/AI4Bharat/MILU.git vendor/MILU
git -C vendor/MILU checkout 7d8e6c9102bf44ae9f9ee84cfabefb4cb8fa2e88
git -C vendor/MILU apply /path/to/patches/vendor-milu.diff
cp -r patches/vendor-milu-new-files/. vendor/MILU/
```

## License

`AI4Bharat/MILU` and `EleutherAI/lm-evaluation-harness` are MIT-licensed
(`vendor/MILU/LICENSE.md`). This patch set is a derivative modification of that MIT-licensed
code and is distributed under the same terms, as part of this project's own MIT-licensed
contribution (see the root [`LICENSE`](../LICENSE)).
