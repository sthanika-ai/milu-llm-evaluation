"""Run scheduler: model-major order. Local models load once, run the entire item
set, unload (subprocess exits => GPU freed), then the next model -- pinned to a
single GPU via CUDA_VISIBLE_DEVICES so we never touch the other user's job on
GPU 0. API models get async requests under rate limits instead --
lm-eval-harness's API backends already implement this internally (see
api_models.py's num_concurrent/max_retries), so the API path here just invokes
lm_eval with those model_args instead of building a bespoke async client.
"""
import argparse
import datetime
import os
# Only ever invoked below with fixed argv lists, never shell=True.
import subprocess  # nosec B404
import sys

from huggingface_hub import HfApi, scan_cache_dir

from pipeline import api_pricing, compute_real_cost, mlflow_logging, registry, results_db, scorer, store

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR_MILU_DIR = os.path.join(REPO_ROOT, "vendor", "MILU")
LM_EVAL_OUTPUT_ROOT = os.path.join(REPO_ROOT, "data_raw_outputs", "_lm_eval_raw")
RUN_LOGS_DIR = os.path.join(REPO_ROOT, "data_raw_outputs", "_run_logs")
ENV_FILE = os.path.join(REPO_ROOT, ".env")

# Local venvs: the default one is pinned to transformers==4.46.3 (the version
# the sanity gate was validated against -- kept untouched so that validation stays
# meaningful). Newer multimodal-architecture checkpoints (Gemma3/4, Qwen3-VL,
# Mistral Small 3.1 -- see hf_causal_multimodal.py) need transformers>=5.x, which
# lives in a separate venv so it can never silently affect the validated path.
# "sarvam" pins transformers==4.57.1 specifically -- confirmed via a direct A/B
# test (same checkpoint/prompt/GPU) that Sarvam-30B's custom
# modeling_sarvam_moe.py produces coherent generation on 4.57.1 (the version
# its own config.json says it was built against) but collapses into
# greedy-decoding repetition loops on the default venv's 5.14.1, regardless of
# enable_thinking -- a real cross-major-version API-drift bug in that custom
# code (transformers.modeling_attn_mask_utils, which the file imports, is
# already deprecated as of 5.10+), not a model or thinking-toggle limitation.
VENV_BIN = {
    "default": os.path.join(REPO_ROOT, ".venv", "bin"),
    "multimodal": os.path.join(REPO_ROOT, ".venv-multimodal", "bin"),
    "vllm2": os.path.join(REPO_ROOT, ".venv-vllm2", "bin"),
    "sarvam": os.path.join(REPO_ROOT, ".venv-sarvam", "bin"),
}

LOCAL_BACKENDS = {"hf", "hf-causal-multimodal", "vllm"}


def load_dotenv(path: str = ENV_FILE) -> None:
    """Minimal .env loader (KEY=VALUE per line) -- avoids pulling in a dependency
    for something this small. Never overrides an already-set env var."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def get_milu_commit() -> str:
    # Fixed argv list, no shell -- "git" resolved from PATH is standard practice
    # for a local dev-tooling script like this one.
    out = subprocess.run(  # nosec B603 B607
        ["git", "rev-parse", "HEAD"], cwd=VENDOR_MILU_DIR, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def get_dataset_revision() -> str:
    return HfApi().dataset_info("ai4bharat/MILU").sha


def _build_vllm_cmd(model_cfg: dict, output_dir: str) -> list:
    # lm-eval-harness's own vllm model class (lm_eval/models/vllm_causallms.py)
    # takes a different kwarg surface than the "hf" backend -- gpu_memory_utilization/
    # max_model_len/max_num_seqs instead of parallelize=True, and no on-the-fly
    # load_in_4bit flag: pre-quantized checkpoints (bitsandbytes/compressed-tensors)
    # are auto-detected by vLLM straight from the checkpoint's own config.json, so
    # quantization is never passed as a model_arg here (it's still recorded in the
    # yaml as a label, so every result still traces to a pinned artifact -- just
    # not fed to vLLM directly).
    model_args = (
        f"pretrained={model_cfg['hf_checkpoint']},"
        f"revision={model_cfg['revision']},"
        f"dtype={model_cfg.get('dtype', 'bfloat16')},"
        f"gpu_memory_utilization={model_cfg.get('gpu_memory_utilization', 0.85)},"
        f"max_model_len={model_cfg.get('max_model_len', 8192)},"
        f"max_num_seqs={model_cfg.get('max_num_seqs', 8)}"
    )
    if model_cfg.get("trust_remote_code"):
        model_args += ",trust_remote_code=True"
    if "enable_thinking" in model_cfg:
        # See vllm_causallms.py's patched apply_chat_template -- same
        # enable_thinking passthrough as the hf backend's, for thinking-toggle
        # chat templates (Qwen3.x, Sarvam's <|nothink|> tag).
        model_args += f",enable_thinking={model_cfg['enable_thinking']}"
    if model_cfg.get("tokenizer_mode"):
        # Mistral checkpoints ship a MistralCommonBackend tokenizer that vLLM's
        # generic get_tokenizer() AutoTokenizer.from_pretrained() path can't load
        # (passes tokenizer_revision/_from_auto kwargs the Mistral backend's
        # from_pretrained rejects) -- vLLM's own warning is explicit that Mistral
        # checkpoints need tokenizer_mode="mistral", which routes through
        # MistralTokenizer.from_pretrained() instead, a different code path that
        # doesn't hit this kwarg mismatch.
        model_args += f",tokenizer_mode={model_cfg['tokenizer_mode']}"
    cmd = [
        _lm_eval_bin(model_cfg),
        "--model", "vllm",
        "--model_args", model_args,
        "--tasks", ",".join(model_cfg["tasks"]),
        "--num_fewshot", str(model_cfg["num_fewshot"]),
        "--batch_size", str(model_cfg.get("batch_size", "auto")),
        "--log_samples",
        "--output_path", output_dir,
    ]
    # generate_until protocol (e.g. mistral-small-3.1-24b's chat-template
    # workaround, sarvam-m-thinkmode-style 0-shot JSON answer extraction) needs a
    # real max_gen_toks -- the harness's own default (256) can be too tight for
    # some tasks; same gen_kwargs mechanism _build_api_cmd already uses.
    gen_cfg = model_cfg.get("generation_config", {})
    if "max_gen_toks" in gen_cfg:
        cmd += ["--gen_kwargs", f"max_gen_toks={gen_cfg['max_gen_toks']}"]
    return cmd


def _build_local_cmd(model_cfg: dict, output_dir: str) -> list:
    if model_cfg["serving_backend"] == "vllm":
        return _build_vllm_cmd(model_cfg, output_dir)
    # temperature/top_p are NOT passed here: they're generation-only parameters
    # and MILU's local-model protocol is loglikelihood scoring (output_type:
    # multiple_choice) -- no sampling/generation ever happens, so these params
    # do nothing except flow through to the model's from_pretrained() as unused
    # kwargs. That's harmless for some architectures (Gemma2, GPT2) but a hard
    # TypeError for others (Llama, Qwen2, Qwen3-VL all reject unrecognized
    # kwargs at construction) -- discovered when the Phase 1 campaign crashed on
    # exactly this for sarvam-1/qwen2.5-7b-instruct/qwen3-vl-8b-instruct-int4.
    model_args = (
        f"pretrained={model_cfg['hf_checkpoint']},"
        f"revision={model_cfg['revision']},"
        f"dtype={model_cfg.get('dtype', 'bfloat16')},"
        f"parallelize=True"
    )
    if model_cfg.get("trust_remote_code"):
        model_args += ",trust_remote_code=True"
    if "enable_thinking" in model_cfg:
        # See huggingface.py's patched apply_chat_template -- lets a config
        # force Qwen3.x-style hybrid-thinking templates off (or on) for the
        # loglikelihood protocol, where a stray "<think>\n" before the scored
        # continuation otherwise silently corrupts the scored logprobs.
        model_args += f",enable_thinking={model_cfg['enable_thinking']}"
    if model_cfg.get("attn_implementation"):
        # Forwarded straight into from_pretrained(attn_implementation=...) --
        # a standard transformers kwarg, real effect only for architectures
        # whose custom modeling code actually plugs into
        # ALL_ATTENTION_FUNCTIONS/_attn_implementation (confirmed present in
        # sarvam-30b's modeling_sarvam_moe.py before using this).
        model_args += f",attn_implementation={model_cfg['attn_implementation']}"

    # quantization is always recorded in the registry as a label (brief: every
    # result must trace to a pinned artifact, including quantization). It only
    # translates into an on-the-fly bitsandbytes load flag when the checkpoint
    # itself isn't already pre-quantized (already_quantized: true) -- applying
    # load_in_4bit on top of an already-4bit checkpoint is redundant/invalid.
    quantization = model_cfg.get("quantization", "none")
    if quantization not in ("none", None, "int4", "int8", "mxfp4", "fp8"):
        raise ValueError(f"{model_cfg['model_id']}: unsupported quantization {quantization!r} (expected none/int4/int8/mxfp4/fp8)")
    if quantization in ("int4", "int8") and not model_cfg.get("already_quantized"):
        if quantization == "int4":
            model_args += ",load_in_4bit=True,bnb_4bit_compute_dtype=bfloat16"
        else:
            model_args += ",load_in_8bit=True"
    if model_cfg.get("dequantize_mxfp4"):
        # gpt-oss-20b's native mxfp4 checkpoint hits a real transformers 5.14.1 /
        # torch 2.5.1 bug (Mxfp4Config.validate_environment calls
        # torch.accelerator.current_accelerator(), an API this torch version
        # doesn't have) -- forcing dequantize=True short-circuits before that
        # code path, loading as plain bf16 instead. See huggingface.py's
        # dequantize_mxfp4 model_kwarg handling for where this actually applies.
        model_args += ",dequantize_mxfp4=True"
    # Matches AI4Bharat/MILU's own README invocation (model_args + batch_size auto:40 /
    # max_batch_size 64) so our run reproduces their exact protocol, not just an equivalent one.
    # batch_size is overridable per-model since auto:N has been observed to OOM by
    # over-scaling into a long-context batch on this shared/contended GPU.
    cmd = [
        _lm_eval_bin(model_cfg),
        "--model", model_cfg["serving_backend"],
        "--model_args", model_args,
        "--tasks", ",".join(model_cfg["tasks"]),
        "--num_fewshot", str(model_cfg["num_fewshot"]),
        "--batch_size", str(model_cfg.get("batch_size", "auto:40")),
        "--max_batch_size", "64",
        "--log_samples",
        "--output_path", output_dir,
    ]
    # generate_until protocol on the plain hf backend (e.g. gpt-oss-20b's
    # generative re-run, matching the workaround already used for
    # mistral-small-3.1-24b/sarvam-m-thinkmode) needs a real max_gen_toks --
    # reasoning-channel output can consume the harness's own default (20)
    # entirely before ever reaching a final-channel answer, same failure mode
    # already confirmed for deepseek-v4-flash.
    gen_cfg = model_cfg.get("generation_config", {})
    gen_kwargs_parts = []
    if "max_gen_toks" in gen_cfg:
        gen_kwargs_parts.append(f"max_gen_toks={gen_cfg['max_gen_toks']}")
    # repetition_penalty/no_repeat_ngram_size: forwarded through
    # huggingface.py's _model_generate(**generation_kwargs) straight into
    # transformers' native model.generate() -- standard GenerationConfig
    # fields, no extra package/version risk (unlike the missing-fused-kernel
    # situation this project hit for a different model's architecture).
    # Needed for models that degenerate into a token-repetition loop under
    # the required temperature=0/do_sample=False greedy decoding (confirmed
    # for sarvam-30b: identical looping behavior with thinking on AND off,
    # ruling out the chat template as the cause -- a real greedy-decoding
    # weakness in this model, not a prompt/protocol bug).
    if "repetition_penalty" in gen_cfg:
        gen_kwargs_parts.append(f"repetition_penalty={gen_cfg['repetition_penalty']}")
    if "no_repeat_ngram_size" in gen_cfg:
        gen_kwargs_parts.append(f"no_repeat_ngram_size={gen_cfg['no_repeat_ngram_size']}")
    if gen_kwargs_parts:
        cmd += ["--gen_kwargs", ",".join(gen_kwargs_parts)]
    return cmd


def _lm_eval_bin(model_cfg: dict) -> str:
    venv = model_cfg.get("venv", "default")
    if venv not in VENV_BIN:
        raise ValueError(f"{model_cfg['model_id']}: unknown venv {venv!r} (expected one of {list(VENV_BIN)})")
    return os.path.join(VENV_BIN[venv], "lm_eval")


def _build_api_cmd(model_cfg: dict, output_dir: str) -> list:
    gen_cfg = model_cfg.get("generation_config", {})
    model_args = (
        f"model={model_cfg['api_endpoint']},"
        f"num_concurrent={model_cfg.get('num_concurrent', 10)},"
        f"max_retries={model_cfg.get('max_retries', 3)},"
        f"timeout={model_cfg.get('timeout_seconds', 300)},"
        f"request_delay={model_cfg.get('request_delay_seconds', 0.0)},"
        f"temperature={gen_cfg.get('temperature', 0.0)},"
        # The API model name (e.g. "deepseek-v4-flash") isn't an HF repo -- without this,
        # TemplateAPI's default tokenizer_backend="huggingface" tries (and 404s) loading
        # one. generate_until scoring never needs local tokenization; the API renders
        # and tokenizes the prompt itself.
        f"tokenizer_backend=None,tokenized_requests=False"
    )
    if model_cfg.get("base_url"):
        model_args += f",base_url={model_cfg['base_url']}"
    # No retries on malformed/refused *content* (brief: "No retries. Log refusals
    # and format failures separately") -- max_retries above is transport-level
    # retry (timeouts/5xx) only, handled by lm-eval-harness itself, not a
    # content-quality retry loop.
    cmd = [
        _lm_eval_bin(model_cfg),
        "--model", model_cfg["serving_backend"],
        "--model_args", model_args,
        "--tasks", ",".join(model_cfg["tasks"]),
        "--num_fewshot", str(model_cfg["num_fewshot"]),
        "--log_samples",
        "--output_path", output_dir,
        # Per-request cache, stable across runs (keyed on model_id, NOT this run's
        # timestamped output_dir) -- --log_samples only writes scored output at the
        # very end of a *successful* run, so a crash mid-run (e.g. one connection
        # timeout exhausting retries, which killed a real 74,782-item/10,462-call-in
        # attempt) otherwise loses every completed call's paid-for result with no way
        # to recover it. Re-running with this same cache path skips already-completed
        # requests instead of re-paying for them.
        "--use_cache", os.path.join(REPO_ROOT, "data_raw_outputs", "_lm_eval_cache", model_cfg["model_id"]),
    ]
    # _default_template_api.yaml's own max_gen_toks=20 assumes a direct JSON-only
    # answer with no reasoning -- true for a plain API model, but confirmed false for
    # deepseek-v4-flash: reasoning tokens count against the same budget, consuming it
    # before any answer content is written (finish_reason="length", empty/truncated
    # content) unless raised. disable_thinking (validated via DeepSeek's documented
    # "thinking":{"type":"disabled"} field) keeps this model on the same "direct
    # knowledge test, no reasoning" footing as every other row in the league table,
    # and is far cheaper/faster -- see openai_completions.py's _create_payload patch,
    # which translates this flat CLI-passable flag into that real nested API field.
    gen_kwargs_parts = []
    if "max_gen_toks" in gen_cfg:
        gen_kwargs_parts.append(f"max_gen_toks={gen_cfg['max_gen_toks']}")
    if gen_cfg.get("disable_thinking"):
        gen_kwargs_parts.append("disable_thinking=true")
    if gen_cfg.get("disable_reasoning"):
        gen_kwargs_parts.append("disable_reasoning=true")
    if gen_cfg.get("openrouter_reasoning_effort"):
        gen_kwargs_parts.append(f"openrouter_reasoning_effort={gen_cfg['openrouter_reasoning_effort']}")
    if gen_cfg.get("openrouter_reasoning_max_tokens"):
        gen_kwargs_parts.append(f"openrouter_reasoning_max_tokens={gen_cfg['openrouter_reasoning_max_tokens']}")
    if gen_cfg.get("openrouter_exclude_reasoning"):
        gen_kwargs_parts.append("openrouter_exclude_reasoning=true")
    if gen_cfg.get("sarvam_reasoning_effort"):
        gen_kwargs_parts.append(f"sarvam_reasoning_effort={gen_cfg['sarvam_reasoning_effort']}")
    if gen_kwargs_parts:
        cmd += ["--gen_kwargs", ",".join(gen_kwargs_parts)]
    return cmd


def cleanup_model_cache(model_cfg: dict, all_model_configs: list) -> None:
    """Delete this model's HF cache entry (weights only, not our own results --
    those already live in data_raw_outputs/results_db/mlruns, none of which
    depend on the cached weights) once its run is done, UNLESS some other
    registered model config still points at the exact same checkpoint+revision
    (e.g. qwen3-vl-8b-instruct and qwen3-vl-8b-instruct-int4 share one repo).
    Several roster checkpoints are 40-55GB; leaving all of them cached
    simultaneously across a 15-model campaign risks running this shared box out
    of disk.
    """
    checkpoint = model_cfg.get("hf_checkpoint")
    revision = model_cfg.get("revision")
    if not checkpoint or not revision:
        return  # API models have no local cache to clean

    still_needed = any(
        c["model_id"] != model_cfg["model_id"] and c.get("hf_checkpoint") == checkpoint and c.get("revision") == revision
        for c in all_model_configs
    )
    if still_needed:
        print(f"[run] {model_cfg['model_id']}: leaving {checkpoint}@{revision[:10]} cached (shared with another registered model)")
        return

    cache_info = scan_cache_dir()
    matching_revisions = [
        rev.commit_hash
        for repo in cache_info.repos
        if repo.repo_id == checkpoint
        for rev in repo.revisions
        if rev.commit_hash == revision
    ]
    if not matching_revisions:
        return  # never downloaded, or already cleaned up

    strategy = cache_info.delete_revisions(*matching_revisions)
    freed_str = strategy.expected_freed_size_str
    strategy.execute()
    print(f"[run] {model_cfg['model_id']}: freed {freed_str} from HF cache ({checkpoint}@{revision[:10]})")


def run_model(model_cfg: dict, milu_commit: str, dataset_revision: str, cuda_visible_devices: str, all_model_configs: list,
              limit: int = None) -> None:
    model_id = model_cfg["model_id"]
    is_local = model_cfg["serving_backend"] in LOCAL_BACKENDS
    run_timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = os.path.join(LM_EVAL_OUTPUT_ROOT, model_id, run_timestamp)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(RUN_LOGS_DIR, exist_ok=True)
    os.makedirs(os.path.join(REPO_ROOT, "data_raw_outputs", "_lm_eval_cache"), exist_ok=True)

    cmd = _build_local_cmd(model_cfg, output_dir) if is_local else _build_api_cmd(model_cfg, output_dir)
    if model_cfg.get("apply_chat_template"):
        cmd.append("--apply_chat_template")
    if limit is not None:
        # Testing only -- lm-eval's own CLI warns real metrics shouldn't be computed
        # this way. Exists so a full pipeline run (real store -> scorer -> results_db
        # -> mlflow wiring, including real per-call token usage / cost for API models)
        # can be smoke-tested end-to-end on a handful of items before committing to a
        # full, unlimited (and for API models, real-money) run.
        cmd += ["--limit", str(limit)]

    env = dict(os.environ)
    # lm_eval's own stdout/stderr (redirected below to log_path, not a TTY) is
    # fully block-buffered by default -- for a long unattended API run, its
    # per-request progress/warning output can sit in an in-memory buffer for
    # a very long time before ever reaching disk, making the log file look
    # frozen/stale even while the run is actually progressing normally
    # (confirmed for real on a multi-hour sarvam-105b run: token_usage.jsonl
    # showed dozens of real completed calls while the log file hadn't grown a
    # single byte past its startup lines). PYTHONUNBUFFERED forces the child
    # process's own CPython interpreter to flush every write immediately.
    env["PYTHONUNBUFFERED"] = "1"
    usage_log_path = None
    if is_local:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    else:
        # Real per-call token usage, for a real $ cost after the run -- see
        # openai_completions.py's _log_usage patch. Upfront estimates
        # (token_cost_estimate.py) are a tiktoken proxy; this is what actually got billed.
        usage_log_path = os.path.join(output_dir, "token_usage.jsonl")
        env["MILU_TOKEN_USAGE_LOG"] = usage_log_path
        # local-chat-completions has no fixed provider, so no fixed API-key env var --
        # each API model config names which real env var (from .env) holds its key;
        # we just relay it under the generic name openai_completions.py's patched
        # LocalCompletionsAPI.api_key reads.
        api_key_env_var = model_cfg.get("api_key_env_var")
        if api_key_env_var:
            if api_key_env_var not in os.environ:
                raise ValueError(f"{model_id}: api_key_env_var={api_key_env_var!r} not set -- add it to .env")
            env["MILU_API_KEY"] = os.environ[api_key_env_var]

    log_path = os.path.join(RUN_LOGS_DIR, f"{model_id}_{run_timestamp}.log")
    print(f"[run] {model_id}: {' '.join(cmd)}")
    print(f"[run] {model_id}: logging to {log_path}")
    # cmd is a fixed argv list built entirely from this repo's own
    # configs/models/*.yaml fields, never shell-interpreted or user-supplied at
    # request time.
    with open(log_path, "w") as log_f:
        result = subprocess.run(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)  # nosec B603
    if result.returncode != 0:
        raise RuntimeError(f"lm_eval failed for {model_id} (exit {result.returncode}); see {log_path}")

    raw_items_path = store.normalize_run(model_cfg, output_dir, milu_commit, dataset_revision, run_timestamp)
    scores = scorer.score_run(raw_items_path)

    real_cost = None
    if usage_log_path and os.path.exists(usage_log_path) and model_id in api_pricing.PRICING:
        usage = compute_real_cost.sum_usage(usage_log_path)
        cost_usd = api_pricing.estimate_cost_usd(model_id, usage["prompt_tokens"], usage["completion_tokens"])
        real_cost = {**usage, "cost_usd": cost_usd}
        print(f"[run] {model_id}: real cost = ${cost_usd:.4f} "
              f"({usage['prompt_tokens']:,} prompt + {usage['completion_tokens']:,} completion tokens, "
              f"{usage['n_calls']} calls)")

    run_id = f"{model_id}_{run_timestamp}"
    conn = results_db.get_connection()
    results_db.insert_scored_run(conn, run_id, model_id, "milu", run_timestamp, scores,
                                  real_cost_usd=real_cost["cost_usd"] if real_cost else None)

    mlflow_logging.log_run(model_cfg, milu_commit, dataset_revision, run_timestamp, scores, run_id, real_cost=real_cost)

    cleanup_model_cache(model_cfg, all_model_configs)

    acc = scores["overall"]["accuracy"]
    acc_str = f"{acc:.4f}" if acc is not None else "N/A"
    print(f"[run] {model_id}: overall accuracy = {acc_str} ({scores['overall']['n_items']} items, "
          f"{scores['overall']['n_malformed']} malformed/refused)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda-visible-devices", default="1")
    parser.add_argument("--models", nargs="*", help="Subset of model_ids to run (default: all in configs/models)")
    parser.add_argument("--limit", type=int, default=None,
                         help="Testing only -- caps items per task. Still exercises the real "
                              "store/scorer/results_db/mlflow pipeline, just on a handful of items.")
    args = parser.parse_args()

    load_dotenv()

    milu_commit = get_milu_commit()
    dataset_revision = get_dataset_revision()
    print(f"[run] MILU repo commit: {milu_commit}")
    print(f"[run] ai4bharat/MILU dataset revision: {dataset_revision}")

    all_model_configs = registry.load_all()
    model_configs = all_model_configs
    if args.models:
        # preserve the order the caller asked for (e.g. smallest-first for a
        # long unattended campaign), not registry.load_all()'s alphabetical order
        by_id = {c["model_id"]: c for c in model_configs}
        missing = [m for m in args.models if m not in by_id]
        if missing:
            raise ValueError(f"Unknown model_id(s) requested: {missing}")
        model_configs = [by_id[m] for m in args.models]

    # One model's failure (OOM, missing dep, bad checkpoint, etc.) must not abort
    # the rest of an unattended multi-model campaign -- log it and move on so the
    # remaining roster still gets a fair shot.
    failed = []
    for cfg in model_configs:
        try:
            run_model(cfg, milu_commit, dataset_revision, args.cuda_visible_devices, all_model_configs, limit=args.limit)
        except Exception as e:
            failed.append(cfg["model_id"])
            print(f"[run] {cfg['model_id']}: FAILED ({type(e).__name__}: {e}) -- continuing with remaining models")

    print(f"[run] campaign done. {len(model_configs) - len(failed)}/{len(model_configs)} succeeded.")
    if failed:
        print(f"[run] failed model_ids: {failed}")


if __name__ == "__main__":
    sys.exit(main())
