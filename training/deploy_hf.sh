#!/usr/bin/env bash
# Create (or reuse) an HF Space that runs the GRPO training Dockerfile.
#
# Usage:
#   ./training/deploy_hf.sh <hf-username> [space-name] [hardware]
#     hardware ∈ {t4-medium, a10g-small, a10g-large, a100-large, zero-a10g}
#     default: a10g-small
#
# Steps:
#   1) hf repos create <user>/<space> --repo-type space --space-sdk docker
#   2) huggingface_hub.request_space_hardware(...)
#   3) hf upload <user>/<space> . . --repo-type space (excludes runs/, results/)
#
# Re-running this script is safe — repo creation is idempotent (errors on
# already-exists are ignored), and `hf upload` re-syncs the working copy.

set -euo pipefail

USER="${1:?usage: deploy_hf.sh <hf-username> [space-name] [hardware]}"
SPACE="${2:-fakegang-grpo}"
HW="${3:-a10g-small}"
REPO="${USER}/${SPACE}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[deploy] repo: $REPO  hardware: $HW  root: $ROOT"

# --- Step 1: create Space (Docker SDK so our Dockerfile is used) -------------
if ! hf repos create "$SPACE" --repo-type space --space-sdk docker --type space \
        2>/tmp/hf_create.err; then
    if grep -qi "exist" /tmp/hf_create.err; then
        echo "[deploy] Space already exists — continuing"
    else
        cat /tmp/hf_create.err >&2
        exit 1
    fi
fi

# --- Step 2: pin Dockerfile path + request hardware --------------------------
# HF Spaces expect the Dockerfile at repo root. We keep ours under training/ so
# we sync a small shim Dockerfile at root that just FROMs ours, OR we use the
# README front-matter `app_file`/Dockerfile pointer. Simplest: copy on upload.
echo "[deploy] syncing Dockerfile to repo root for the Space build"
cp -f training/Dockerfile ./Dockerfile

# README is required for HF Space metadata (sdk: docker, app_port).
cat > ./README.md.space <<'EOF'
---
title: fakegang-grpo
emoji: 🛡️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

GRPO training for the fake-gang content-moderation policy. See `training/RUNBOOK.md`.
EOF
mv -f ./README.md.space ./README.md

# --- Step 3: upload (exclude heavy/local-only paths) -------------------------
echo "[deploy] uploading working copy (heavy dirs excluded — env is remote)"
hf upload "$REPO" . . --repo-type space \
    --exclude "episodes/*" \
    --exclude "dashboard/*" \
    --exclude "wheels/*" \
    --exclude "assets/*" \
    --exclude "images/*" \
    --exclude "agent/*" \
    --exclude "tests/*" \
    --exclude "graphify-out/*" \
    --exclude "model-benchmark-logs/*" \
    --exclude "runs/*" \
    --exclude "training/runs/*" \
    --exclude "eval-models/results/*" \
    --exclude "memory/*" \
    --exclude "policy_cache/*" \
    --exclude "server/episodes/*" \
    --exclude "**/__pycache__/*" \
    --exclude "**/.pytest_cache/*" \
    --exclude "**/*.pyc" \
    --exclude "**/*.log" \
    --exclude "judge_log.txt" \
    --exclude "uv.lock" \
    --exclude ".git/*"

# --- Step 4: request GPU hardware --------------------------------------------
echo "[deploy] requesting hardware: $HW"
python3 - <<PY
from huggingface_hub import HfApi, SpaceHardware
hw_map = {
    "t4-medium":   SpaceHardware.T4_MEDIUM,
    "a10g-small":  SpaceHardware.A10G_SMALL,
    "a10g-large":  SpaceHardware.A10G_LARGE,
    "a100-large":  SpaceHardware.A100_LARGE,
    "zero-a10g":   SpaceHardware.ZERO_A10G,
}
hw = hw_map["${HW}"]
HfApi().request_space_hardware(repo_id="${REPO}", hardware=hw)
print(f"[deploy] hardware requested: {hw}")
PY

echo
echo "[deploy] done.  https://huggingface.co/spaces/${REPO}"
echo "[deploy] tail logs with:  hf spaces info ${REPO}"
