FROM python:3.12-slim

WORKDIR /app

# ── Install deps from pre-downloaded wheels (no network needed, fast) ────────
COPY wheels/ wheels/
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=wheels/ \
    fastapi "uvicorn[standard]" "pydantic>=2.6.0" boto3 botocore requests \
 && rm -rf wheels/

# ── Source code (~87 KB, episodes excluded via .dockerignore) ────────────────
COPY . .

# ── Persistent dirs (mount as Docker volumes for persistence) ────────────────
RUN mkdir -p /app/memory /app/runs /app/episodes
VOLUME ["/app/memory", "/app/runs"]

ENV AWS_DEFAULT_REGION=us-east-1
ENV TRAIN_TASK=""
ENV TRAIN_EPISODES=50
ENV TRAIN_TEMP=0.4
ENV TRAIN_VERBOSE=0
ENV SERVER_PORT=8000

EXPOSE 8000

RUN chmod +x run.sh
CMD ["bash", "run.sh"]
