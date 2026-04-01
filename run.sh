#!/usr/bin/env bash
# Docker entrypoint: generates episodes, starts the OpenEnv server,
# then runs the LLM agent training loop — all inside the same container.
#
# Required environment variables (pass via docker run -e):
#   AWS_ACCESS_KEY_ID      — for Bedrock / Qwen3 access
#   AWS_SECRET_ACCESS_KEY  — for Bedrock / Qwen3 access
#
# Optional:
#   TRAIN_TASK      easy | medium | hard | (blank = curriculum)
#   TRAIN_EPISODES  default: 50
#   TRAIN_TEMP      default: 0.4
#   TRAIN_VERBOSE   set to "1" for per-step logging
#   SERVER_PORT     default: 8000
#
# Persistent volumes:
#   /app/memory/   — reflections + best trajectories
#   /app/runs/     — training metrics (JSONL)

set -euo pipefail

PORT="${SERVER_PORT:-8000}"
TASK_ARG="${TRAIN_TASK:-}"
EPISODES="${TRAIN_EPISODES:-50}"
TEMP="${TRAIN_TEMP:-0.4}"
VERBOSE="${TRAIN_VERBOSE:-0}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   Fake Gang Detection — OpenEnv RL Environment              ║"
echo "║   LLM Agent: Qwen3 via AWS Bedrock (Reflexion learning)     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Validate AWS credentials
if [[ -z "${AWS_ACCESS_KEY_ID:-}" || -z "${AWS_SECRET_ACCESS_KEY:-}" ]]; then
    echo "[ERROR] AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set."
    exit 1
fi

# ── 1. Generate episodes (fast, ~1s, skipped if already exist) ───────────────
echo "[1/3] Generating episodes..."
python server/generator.py
echo "      Done."
echo ""

# ── 2. Start the OpenEnv environment server ──────────────────────────────────
echo "[2/3] Starting OpenEnv environment server on port ${PORT}..."
python -m uvicorn server.app:app \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --workers 1 \
    --log-level warning &
SERVER_PID=$!

# Health check using Python (no curl needed)
echo "      Waiting for server..."
python - <<EOF
import time, urllib.request, urllib.error, sys
for i in range(30):
    try:
        urllib.request.urlopen("http://localhost:${PORT}/health", timeout=2)
        print("      Server ready ✓")
        sys.exit(0)
    except Exception:
        time.sleep(1)
print("[ERROR] Server did not start in 30s")
sys.exit(1)
EOF

echo ""

# ── 3. Run the LLM agent training loop ───────────────────────────────────────
echo "[3/3] Starting LLM agent training..."
echo "      Episodes : ${EPISODES}"
echo "      Task     : ${TASK_ARG:-curriculum (easy→medium→hard)}"
echo "      LLM temp : ${TEMP}"
echo ""

TRAIN_ARGS=(
    --env-url "http://localhost:${PORT}"
    --episodes "${EPISODES}"
    --temperature "${TEMP}"
    --log-dir runs
)

[[ -n "${TASK_ARG}" ]] && TRAIN_ARGS+=(--task "${TASK_ARG}")
[[ "${VERBOSE}" == "1" ]] && TRAIN_ARGS+=(--verbose)

python train.py "${TRAIN_ARGS[@]}"

echo ""
echo "Training done. Metrics → /app/runs/metrics.jsonl  Memory → /app/memory/"
wait "$SERVER_PID"
