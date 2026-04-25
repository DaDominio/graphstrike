#!/usr/bin/env bash
# Auto-train on Space boot.
#
# The env server lives on a separate Space at $ENV_BASE_URL; we DO NOT start
# a local server here — we just hit it. Default points to the Pandago env Space.
#
# Knobs (set as Space variables / secrets):
#   PHASE          phase0 | phase1 | phase2 | phase3   (default phase1)
#   MODEL          HF model id                          (default Qwen/Qwen2.5-1.5B-Instruct)
#   PLATFORM       Instagram | Snapchat | LinkedIn | …  (default Instagram)
#   ENV_BASE_URL   env Space URL                        (default Pandago space)
#   WANDB_PROJECT  W&B project name                     (default fakegang-grpo)
#   WANDB_API_KEY  secret — required if W&B enabled
#   HF_TOKEN       secret — required for gated models / for pushing checkpoints
#   PUSH_TO_HUB    "1" to push final checkpoint, "0" to skip   (default 0)
#   PUSH_REPO_ID   <user>/<model-name>                          (required if PUSH_TO_HUB=1)
set -euo pipefail
cd /app

PHASE="${PHASE:-phase1}"
# Normalize: accept "0"/"1"/"2"/"3" as shorthand for phase0..phase3.
case "$PHASE" in
    0|1|2|3) PHASE="phase${PHASE}" ;;
esac
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
PLATFORM="${PLATFORM:-Instagram}"
ENV_BASE_URL="${ENV_BASE_URL:-https://pandago-graphstrike-model-training.hf.space}"
WANDB_PROJECT="${WANDB_PROJECT:-fakegang-grpo}"
PUSH_TO_HUB="${PUSH_TO_HUB:-0}"

echo "[entrypoint] phase=$PHASE model=$MODEL platform=$PLATFORM env=$ENV_BASE_URL"

# 1) Probe the remote env so we fail fast with a clear message.
echo "[entrypoint] probing remote env /health"
for i in $(seq 1 30); do
    if curl -fsS "${ENV_BASE_URL}/health" >/dev/null 2>&1; then
        echo "[entrypoint] env reachable (after ${i}s)"
        break
    fi
    sleep 2
    if [ "$i" = "30" ]; then
        echo "[entrypoint] FATAL: cannot reach $ENV_BASE_URL/health" >&2
        exit 1
    fi
done

# 2) Run the requested phase against the remote env.
EXTRA_ARGS=()
if [ -n "${WANDB_API_KEY:-}" ]; then
    EXTRA_ARGS+=( --wandb-project "$WANDB_PROJECT" )
fi

python -m training.train_grpo \
    --phase   "$PHASE" \
    --model   "$MODEL" \
    --platform "$PLATFORM" \
    --base-url "$ENV_BASE_URL" \
    "${EXTRA_ARGS[@]}"

# 3) Optional: push the trained checkpoint to the Hub.
if [ "$PUSH_TO_HUB" = "1" ] && [ -n "${PUSH_REPO_ID:-}" ]; then
    echo "[entrypoint] pushing training/runs/$PHASE → $PUSH_REPO_ID"
    hf upload "$PUSH_REPO_ID" "training/runs/$PHASE" --repo-type model || true
fi

echo "[entrypoint] done — sleeping so Space stays 'running' for log inspection"
# Spaces auto-sleep when the main process exits; keep alive for log retrieval.
tail -f /dev/null
