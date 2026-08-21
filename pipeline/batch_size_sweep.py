"""One-off utility: empirically sweep fixed batch sizes for a model on a small
--limit slice, measure real it/s for each, and report the best. Not part of the
main pipeline -- run manually when a model's auto:N batch-size detection is
suspect or when we want a verified-fast fixed batch_size for a long campaign run.
"""
import argparse
import os
import re
# Only ever invoked below with fixed argv lists, never shell=True.
import subprocess  # nosec B404
import sys
import tempfile
import time


def measure(lm_eval_bin, model_args, task, limit, batch_size, cuda_visible_devices, apply_chat_template):
    with tempfile.TemporaryDirectory(prefix="batch_sweep_") as output_dir:
        cmd = [
            lm_eval_bin, "--model", "hf-causal-multimodal",
            "--model_args", model_args,
            "--tasks", task,
            "--num_fewshot", "5",
            "--limit", str(limit),
            "--batch_size", str(batch_size),
            "--output_path", output_dir,
        ]
        if apply_chat_template:
            cmd.append("--apply_chat_template")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": cuda_visible_devices}

        # cmd is a fixed list built from this script's own CLI args (run manually
        # by an operator), never shell-interpreted or attacker-supplied.
        start = time.time()
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)  # nosec B603
        elapsed = time.time() - start

    if proc.returncode != 0:
        tail = (proc.stdout + proc.stderr)[-1500:]
        return {"batch_size": batch_size, "ok": False, "elapsed": elapsed, "error_tail": tail}

    # parse the final tqdm rate reported in stderr, e.g. "...it/s]"
    combined = proc.stdout + proc.stderr
    rates = re.findall(r"([\d.]+)it/s\]", combined)
    final_rate = float(rates[-1]) if rates else None
    return {"batch_size": batch_size, "ok": True, "elapsed": round(elapsed, 1), "reported_rate": final_rate}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lm-eval-bin", required=True)
    p.add_argument("--model-args", required=True)
    p.add_argument("--task", default="milu_Gujarati")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--cuda-visible-devices", default="1")
    p.add_argument("--apply-chat-template", action="store_true")
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[8, 16, 24, 32, 48, 64])
    args = p.parse_args()

    results = []
    for bs in args.batch_sizes:
        print(f"[sweep] testing batch_size={bs} ...", flush=True)
        r = measure(args.lm_eval_bin, args.model_args, args.task, args.limit, bs,
                    args.cuda_visible_devices, args.apply_chat_template)
        results.append(r)
        if r["ok"]:
            print(f"[sweep] batch_size={bs}: rate={r['reported_rate']} it/s, wall={r['elapsed']}s")
        else:
            print(f"[sweep] batch_size={bs}: FAILED ({r['error_tail'][-300:]})")
            if "CUDA out of memory" in r["error_tail"]:
                print(f"[sweep] OOM at batch_size={bs}, stopping sweep (larger sizes will also OOM)")
                break

    ok_results = [r for r in results if r["ok"] and r["reported_rate"]]
    if ok_results:
        best = max(ok_results, key=lambda r: r["reported_rate"])
        print(f"\n[sweep] BEST: batch_size={best['batch_size']} at {best['reported_rate']} it/s")
    else:
        print("\n[sweep] no successful batch size found")
    return results


if __name__ == "__main__":
    sys.exit(main() and 0)
