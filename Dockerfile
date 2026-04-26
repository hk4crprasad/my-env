# ─────────────────────────────────────────────────────────────────────────────
#  Email Triage RL Environment — HF Spaces Docker (GPU: T4 small)
# ─────────────────────────────────────────────────────────────────────────────
# HF Spaces provides the NVIDIA driver via nvidia-container-runtime; we just
# need a Python image and torch wheels that bundle their own CUDA runtime.
# The cu121 wheels work on T4 (Turing, compute cap 7.5) and on any Ampere+ GPU.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface/hub

# System deps — curl for healthcheck, git for git-based pip installs (transformers source builds)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1️⃣  Install PyTorch with CUDA 12.1 runtime (T4-compatible) BEFORE other deps.
#     Latest cu121 wheel — works on T4 (Turing cc 7.5) and Ampere+.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch

# 2️⃣  Install the rest from PyPI (separate layer for cache-friendliness)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3️⃣  Copy application code last so code edits don't bust the heavy pip layers
COPY . .

# Cache dir must be writable by the Spaces runtime user
RUN mkdir -p $HF_HOME && chmod -R 777 /app/.cache

# HF Spaces requires port 7860
EXPOSE 7860

# Lightweight liveness probe — the OpenEnv API endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://localhost:7860/health || exit 1

# Entrypoint handles loading HF Spaces secrets (single ENV secret or individual ones)
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Single worker — the lazy-loaded LoRA adapter must live in one process to avoid
# duplicate VRAM use. Multi-worker would OOM on T4.
ENTRYPOINT ["/app/entrypoint.sh"]
