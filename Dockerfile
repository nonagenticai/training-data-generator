ARG BASE_REGISTRY=docker.io
FROM ${BASE_REGISTRY}/library/python:3.14-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.10.10 /uv /uvx /usr/local/bin/

WORKDIR /app

COPY pyproject.toml uv.lock* ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


FROM ${BASE_REGISTRY}/library/python:3.14-slim AS runtime

RUN groupadd --gid 1000 app && \
    useradd --uid 1000 --gid app --shell /bin/bash --create-home app

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY src/ src/
# Alembic migration assets — the `migrate` init-container runs
# `alembic upgrade head` from these (schema source of truth). env.py adds
# /app/src to sys.path so `from models...` resolves under PYTHONPATH=/app/src.
COPY alembic.ini alembic.ini
COPY alembic alembic

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app/src

RUN mkdir -p /home/app/.training-data && chown -R app:app /home/app/.training-data

USER app

EXPOSE 8152

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8152", "--workers", "2"]
