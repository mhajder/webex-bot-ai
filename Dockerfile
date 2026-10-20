FROM python:3.14-alpine3.24 AS builder

RUN apk add --no-cache build-base libffi-dev

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_DOWNLOADS=0

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --all-extras --no-install-project --no-dev --no-editable

COPY pyproject.toml uv.lock LICENSE README.md ./
COPY src/ ./src/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --all-extras --no-dev --no-editable


FROM python:3.14-alpine3.24
LABEL org.opencontainers.image.title="Webex Bot AI" \
    org.opencontainers.image.description="AI-powered Webex Bot" \
    org.opencontainers.image.url="https://github.com/mhajder/webex-bot-ai" \
    org.opencontainers.image.source="https://github.com/mhajder/webex-bot-ai" \
    org.opencontainers.image.vendor="Mateusz Hajder" \
    org.opencontainers.image.licenses="MIT"
ENV PYTHONUNBUFFERED=1

RUN apk add --no-cache ca-certificates \
    && addgroup -g 10001 appuser \
    && adduser -D -u 10001 -G appuser appuser \
    && mkdir -p /app/data \
    && chown 10001:10001 /app/data

COPY --from=builder /app/.venv /app/.venv

WORKDIR /app

USER 10001:10001

ENV PATH="/app/.venv/bin:$PATH"
ENV CONVERSATION_DB_PATH="/app/data/conversations.db"
VOLUME ["/app/data"]

ENTRYPOINT ["webex-bot-ai"]
