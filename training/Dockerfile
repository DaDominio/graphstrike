# HF Spaces GPU container for GRPO training.
#
# Build context = repo root (fake_gang_env/). The Space runs the env server +
# launches a training phase. The phase is selected via the PHASE env var.
#
# Default workflow on the Space:
#   1) start the env server in the background (uvicorn, port 8000)
#   2) wait for /health
#   3) run `python -m training.train_grpo --phase $PHASE ...`
#
# To run locally:  docker build -t fakegang-train -f training/Dockerfile .
#                  docker run --gpus all -e PHASE=phase1 fakegang-train

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PHASE=phase1 \
    MODEL=Qwen/Qwen2.5-1.5B-Instruct \
    PLATFORM=Instagram \
    ENV_BASE_URL=https://pandago-graphstrike-model-training.hf.space

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*
RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Install training deps (cached layer; env is remote so we don't need server deps).
COPY training/requirements.txt /tmp/training-requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /tmp/training-requirements.txt

# Copy the rest of the repo (client.py + eval-models/_round2_runner.py + training/).
COPY . /app

EXPOSE 7860

CMD ["bash", "training/entrypoint.sh"]
