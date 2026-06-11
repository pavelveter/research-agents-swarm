# Generation Step 1: Resolve dependencies inside isolated builder
FROM ghcr.io/astral-sh/uv:python3.13-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Кэшируем только файлы зависимостей для сборки слоев
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen --no-install-project --no-dev

# Generation Step 2: Runtime image
FROM python:3.13-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Копируем виртуальное окружение из builder стадии
COPY --from=builder /app/.venv /app/.venv
# Копируем исходники приложения
COPY src/ /app/src/
COPY pyproject.toml uv.lock /app/

# Проверяем работоспособность виртуального окружения
RUN python -c "import sys; print(sys.path)"

ENTRYPOINT ["python", "-m", "research_swarm.main"]