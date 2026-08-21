#!/usr/bin/env bash
# Clones the two vendored third-party dependencies at the exact commits this project's
# results were produced with, then applies this project's own local patch to vendor/MILU
# (see patches/README.md for what's in that patch and why).
#
# Not run automatically by anything else in this repo -- vendor/ is gitignored (see
# .gitignore), so this is the one-time setup step a fresh clone needs before
# scripts/run_evaluation.sh / python -m pipeline.run will work.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MILU_COMMIT="7d8e6c9102bf44ae9f9ee84cfabefb4cb8fa2e88"
# llama.cpp is only needed for the Sarvam-30B GGUF path (configs/models/sarvam-30b-llamacpp-*.yaml).
# This commit is NOT independently pinned anywhere else in this project's docs -- it's
# recorded here as the HEAD of the locally-vendored clone as observed on 2026-08-10, the
# closest available record of what was actually used. If you need an exact reproduction of
# the Sarvam-30B GGUF numbers and this commit no longer builds cleanly, treat the llama.cpp
# version as an open item to re-verify, not a guaranteed-correct pin.
LLAMACPP_COMMIT="fc3f10b"

echo "[setup_vendor] cloning AI4Bharat/MILU @ ${MILU_COMMIT}"
if [ ! -d vendor/MILU/.git ]; then
    git clone https://github.com/AI4Bharat/MILU.git vendor/MILU
fi
git -C vendor/MILU fetch --depth 1 origin "${MILU_COMMIT}" || true
git -C vendor/MILU checkout "${MILU_COMMIT}"

echo "[setup_vendor] applying this project's local patch to vendor/MILU"
git -C vendor/MILU apply --check "${REPO_ROOT}/patches/vendor-milu.diff"
git -C vendor/MILU apply "${REPO_ROOT}/patches/vendor-milu.diff"
cp -r "${REPO_ROOT}/patches/vendor-milu-new-files/." vendor/MILU/
echo "[setup_vendor] vendor/MILU ready (pinned commit + local patch applied)"

echo "[setup_vendor] cloning ggml-org/llama.cpp @ ${LLAMACPP_COMMIT} (only needed for the Sarvam-30B GGUF path)"
if [ ! -d vendor/llama.cpp/.git ]; then
    git clone https://github.com/ggml-org/llama.cpp.git vendor/llama.cpp
fi
git -C vendor/llama.cpp fetch origin "${LLAMACPP_COMMIT}" || true
git -C vendor/llama.cpp checkout "${LLAMACPP_COMMIT}"

echo "[setup_vendor] building llama.cpp with CUDA support (skip this and pass --no-cuda if you have no GPU / need CPU-only)"
if [ "${1:-}" != "--no-cuda" ]; then
    cmake -B vendor/llama.cpp/build -S vendor/llama.cpp -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build vendor/llama.cpp/build --config Release -j "$(nproc)"
else
    cmake -B vendor/llama.cpp/build -S vendor/llama.cpp -DCMAKE_BUILD_TYPE=Release
    cmake --build vendor/llama.cpp/build --config Release -j "$(nproc)"
fi

echo "[setup_vendor] done. See requirements/README.md for the Python venv(s) still needed on top of this."
