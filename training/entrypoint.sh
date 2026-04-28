#!/usr/bin/env bash
set -euo pipefail
cd /app

PHASE="${PHASE:-phase1}"
case "$PHASE" in
    0|1|2|3) PHASE="phase${PHASE}" ;;
esac
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
PLATFORM="${PLATFORM:-Instagram}"
PLATFORMS="${PLATFORMS:-Instagram,X,Snapchat}"
EVAL_PLATFORM="${EVAL_PLATFORM:-LinkedIn}"
ENV_BASE_URL="${ENV_BASE_URL:-https://pandago-graphstrike-model-training.hf.space}"
WANDB_PROJECT="${WANDB_PROJECT:-fakegang-grpo}"
PUSH_TO_HUB="${PUSH_TO_HUB:-0}"
RUN_TAG="${RUN_TAG:-$PHASE}"

echo "[entrypoint] phase=$PHASE model=$MODEL tag=$RUN_TAG platform=$PLATFORM env=$ENV_BASE_URL"

# ── 1. Health-check server on :7860 ──────────────────────────────────────────
# HF Spaces kills the container when nothing responds on the exposed port for
# ~30 min. This minimal server keeps the container alive for the full run.
python3 -c "
from http.server import HTTPServer, BaseHTTPRequestHandler
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'training in progress')
    def log_message(self, *a): pass
HTTPServer(('0.0.0.0', 7860), H).serve_forever()
" &
HEALTH_PID=$!
echo "[entrypoint] health server PID=$HEALTH_PID listening on :7860"

# ── 2. Probe remote env ───────────────────────────────────────────────────────
echo "[entrypoint] probing remote env /health"
for i in $(seq 1 30); do
    if curl -fsS "${ENV_BASE_URL}/health" >/dev/null 2>&1; then
        echo "[entrypoint] env reachable (after ${i}s)"
        break
    fi
    sleep 2
    if [ "$i" = "30" ]; then
        echo "[entrypoint] FATAL: cannot reach $ENV_BASE_URL/health" >&2
        kill "$HEALTH_PID" 2>/dev/null || true
        exit 1
    fi
done

# ── 3. Training ───────────────────────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXTRA_ARGS=()
if [ -n "${WANDB_API_KEY:-}" ]; then
    EXTRA_ARGS+=( --wandb-project "$WANDB_PROJECT" )
fi

if [ "$PHASE" = "phase3" ]; then
    PLAT_LIST=$(echo "$PLATFORMS" | tr ',' ' ')
    EXTRA_ARGS+=( --platforms $PLAT_LIST --eval-platform "$EVAL_PLATFORM" )
fi

python3 -m training.train_grpo \
    --phase    "$PHASE" \
    --model    "$MODEL" \
    --platform "$PLATFORM" \
    --base-url "$ENV_BASE_URL" \
    --out-dir  "/app/training/runs/$RUN_TAG" \
    "${EXTRA_ARGS[@]}"

# ── 4. Push checkpoint ────────────────────────────────────────────────────────
if [ "$PUSH_TO_HUB" = "1" ] && [ -n "${PUSH_REPO_ID:-}" ]; then
    echo "[entrypoint] pushing training/runs/$RUN_TAG → $PUSH_REPO_ID"
    python3 -c "
from huggingface_hub import HfApi
import os
api = HfApi(token=os.environ['HF_TOKEN'])
run_tag = os.environ.get('RUN_TAG', os.environ.get('PHASE', 'phase1'))
run_dir = f'/app/training/runs/{run_tag}'
api.upload_folder(
    folder_path=run_dir,
    repo_id=os.environ['PUSH_REPO_ID'],
    repo_type='model',
    commit_message=f'checkpoint: {run_tag} complete',
    path_in_repo=run_tag,
)
print('[entrypoint] weights pushed OK')
"
fi

echo "[entrypoint] training complete — health server keeping container alive"
wait "$HEALTH_PID"
