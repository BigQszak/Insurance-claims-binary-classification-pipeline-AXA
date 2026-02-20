# syntax=docker/dockerfile:1
# ============================================================
# Multi-stage Dockerfile for the AXA ML pipeline
# ============================================================

# --- Stage 1: Build (install dependencies via uv) -----------
FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install production dependencies only
RUN uv sync --frozen --no-dev --no-install-project

# Copy source code & install the project itself
COPY src/ src/
COPY configs/ configs/
RUN uv sync --frozen --no-dev

# --- Stage 2: Runtime ----------------------------------------
FROM python:3.11-slim

WORKDIR /app

# Copy virtual environment and source from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/configs /app/configs
COPY --from=builder /app/pyproject.toml /app/pyproject.toml

# Put the venv's bin on PATH
ENV PATH="/app/.venv/bin:$PATH"

# Default command: run the full pipeline
ENTRYPOINT ["axa-ml"]
CMD ["run", "--config", "configs/default.yaml"]
