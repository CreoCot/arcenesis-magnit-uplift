# Stage 1
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

WORKDIR /app

# Environment variables
ENV UV_COMPILE_BYTECODE=0 \
    UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-install-project --no-dev


# Stage 2
FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# Add user without root access
RUN adduser --disabled-password --gecos '' appuser && \
    mkdir -p /app/data /app/artifacts && \
    chown -R appuser:appuser /app

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser artifacts/ ./artifacts/

USER appuser

ENTRYPOINT ["python", "-m", "src.predict"]
CMD ["--stub"]