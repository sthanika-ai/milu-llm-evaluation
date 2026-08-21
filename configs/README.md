# Model configs

One YAML file per evaluated model in `configs/models/`. Each config pins an exact checkpoint
name, HF revision hash (or API endpoint plus access date), serving backend, quantization, and
generation config, so every reported result traces back to a specific, pinned artifact.
Loaded and validated by `pipeline/registry.py`.

There is no separate `configs/prompts/` directory. Prompt templates are not duplicated here —
they live in the vendored harness's own task configs
(`vendor/MILU/lm_eval/tasks/milu/milu_<Language>.yaml` for loglikelihood,
`milu_<Language>_api[_nostop].yaml` for generative, the latter added by this project — see
`patches/README.md`), which is what actually runs. Copying prompt text into a second location
here would risk drifting from what the harness really uses.

## Schema

Enforced by `pipeline/registry.py::load_model_config`. Required for every config:

- `model_id`, `serving_backend`, `apply_chat_template`, `num_fewshot`, `tasks`

Local backends (`serving_backend` in `hf`, `hf-causal-multimodal`, `vllm`) additionally
require:

- `hf_checkpoint`, `revision` — exact Hugging Face repo + commit SHA, never a moving tag.

API/remote backends additionally require:

- `api_endpoint`, `access_date` — API model weights aren't pinned by us and can change
  silently server-side, so an access date substitutes for a revision hash.

`protocol` defaults to `loglikelihood` for local backends and `generate_until` for API
backends, and `registry.py` rejects `loglikelihood` on an API backend (chat-completions APIs
don't expose prompt logprobs — see `vendor/MILU/lm_eval/models/openai_completions.py`'s own
`NotImplementedError`).

Common optional fields you'll see across these configs: `venv` (which of the five
environments in `requirements/README.md` to run this model under — see `pipeline/run.py`'s
`VENV_BIN`), `dtype`, `quantization`, `batch_size`, `generation_config` (temperature,
max_gen_toks/max_new_tokens), `api_key_env_var` (which `.env` variable holds this API's key),
`base_url` (for `local-chat-completions` configs pointing at a manually-started server, e.g.
the Sarvam-30B llama.cpp configs), and a free-text `notes` field recording why a setting was
chosen, plus any library patch that model's run needed — read it before copying a config as a
template.

## Published vs. full local registry

`configs/models/` here contains **19 curated configs**: one canonical, full-11-language
config per model with a reported result, plus the sanity-gate config — except Qwen3.6-27B,
which has two (thinking forced off and thinking left on are both genuine, disclosed
configurations with materially different results, so both are published; see the root
`README.md` §8). This project's full internal run history (investigation configs, smoke tests,
per-language staging splits, superseded pre-fix rows) is larger but isn't published, to keep
the registry to one config per model (two for Qwen3.6-27B) instead of requiring a newcomer to
guess which of several similarly-named variants is the authoritative one.

Every published config's `tasks:` field runs all 11 MILU languages in one command, using
either the standard `milu`/`milu_api`/`milu_api_nostop` group tasks or a consolidated task
list where a model's real run was originally split across languages for staging or hardware
reasons. Two of these — `sarvam-30b-llamacpp-gguf.yaml` and `gpt-oss-20b-thinkmode-vllm.yaml`
— reach their reported accuracy only after a malformed-item retry pass on top of the base run
(their own `notes:` fields explain why); everything else is reproducible in one command.

Superseded/broken-protocol configurations (early runs later found to use an incorrect scoring
protocol, corrected in a later config for the same model) are intentionally not part of the
published set.

## Naming conventions in the full internal registry

- `<model_id>.yaml` is the primary/official config for that model.
- `<model_id>-<language>.yaml` — a single-language run.
- `<model_id>-remaining10.yaml` — the other 10 languages, run separately after one was
  validated first.
- `-genthinkoff-`/`-genthinkon-`/`-thinkmode-` — whether a model's chat-template reasoning
  mode is forced off, left on, or explicitly exercised.
- `-gpu0-`/`-gpu1-`/`-2srv-` — a run deliberately split across multiple GPUs/servers for
  parallelism.

## Adding a new model

1. Copy the config closest to your model's architecture/backend as a starting point (e.g. a
   `hf-causal-multimodal` config for another multimodal model).
2. Pin the exact checkpoint revision — `huggingface-cli download <repo> --revision <sha>` or
   check the model page's commit history, never a branch name.
3. Run `python -m pipeline.run --models <model_id> --limit 20` first to confirm the config
   loads and scores before committing to a full run.
4. If it's a reasoning/thinking-capable model, check whether its chat template enables
   thinking by default before trusting a loglikelihood score — see §8 of the root `README.md`
   for why this matters.
