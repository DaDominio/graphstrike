# ── GraphStrike — OpenEnv Environment Server ─────────────────────────────────
# This Dockerfile runs the environment server (FastAPI + Gradio UI + episodes).
# No training loop, no AWS credentials required.
#
# Used by:  openenv push / HuggingFace Spaces deployment
# Port:     7860 (HF Spaces default — configurable via PORT env var)
#
# For local training (Qwen3 + Reflexion + Hybrid Policy), use server/Dockerfile.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.12-slim

WORKDIR /app

# ── Install Python dependencies (network available on HF build workers) ───────
COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    "pydantic>=2.6.0" \
    requests \
    "openenv-core>=0.2.0" \
    "gradio>=4.0.0" \
    "openai>=1.0.0"

# ── Copy source code ──────────────────────────────────────────────────────────
COPY . .

# ── Pre-generate all 150 episodes at build time (~1 second, deterministic) ───
# This bakes the episodes into the image so the server starts instantly.
RUN python server/generator.py

# ── Dirs for optional persistent data (mounted as volumes on local Docker) ───
RUN mkdir -p /app/memory /app/runs

# ── Runtime config ────────────────────────────────────────────────────────────
ENV PORT=7860
ENV AWS_DEFAULT_REGION=us-east-1

# HF Spaces expects the app on port 7860 (override via PORT env var)
EXPOSE 7860

# Server only — the training loop is NOT started here.
# Judges evaluate the environment via the API endpoints:
#   GET  /health      → liveness check
#   GET  /tasks       → available tasks + action schema
#   POST /reset       → start an episode
#   POST /step        → take an action
#   GET  /grader      → get normalised score after SUBMIT
#   POST /baseline    → run the rule-based agent on all 3 tasks
CMD ["python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--log-level", "info"]
