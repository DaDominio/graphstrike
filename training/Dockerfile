FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    CC=/usr/bin/gcc \
    CXX=/usr/bin/g++ \
    PHASE=phase1 \
    MODEL=Qwen/Qwen2.5-1.5B-Instruct \
    PLATFORM=Instagram \
    ENV_BASE_URL=https://pandago-graphstrike-model-training.hf.space

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv git curl ca-certificates \
        build-essential gcc g++ && \
    ln -sf /usr/bin/gcc /usr/bin/cc && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

ARG CACHE_BUST=2026-04-25-f
RUN echo "cache-bust: $CACHE_BUST"

COPY training/requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir bitsandbytes --prefer-binary && \
    pip install --no-cache-dir -r /tmp/training-requirements.txt && \
    python3 -c "import bitsandbytes; print('bnb OK')" && \
    python3 -c "import torch; print('torch', torch.__version__)" && \
    python3 -c "import trl; print('trl', trl.__version__)"

COPY . /app

EXPOSE 7860

CMD ["bash", "training/entrypoint.sh"]
