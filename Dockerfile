FROM python:3.13-alpine3.23 AS builder

RUN pip install --root-user-action=ignore --no-cache-dir --upgrade pip \
    && pip install --root-user-action=ignore --no-cache-dir uv

ENV UV_LINK_MODE=copy

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --all-extras --no-install-project --no-dev

COPY pyproject.toml uv.lock LICENSE README.md ./
COPY src/ ./src/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --all-extras --no-dev


FROM python:3.13-alpine3.23
LABEL org.opencontainers.image.title="Webex Bot AI" \
    org.opencontainers.image.description="AI-powered Webex Bot" \
    org.opencontainers.image.url="https://github.com/mhajder/webex-bot-ai" \
    org.opencontainers.image.source="https://github.com/mhajder/webex-bot-ai" \
    org.opencontainers.image.vendor="Mateusz Hajder" \
    org.opencontainers.image.licenses="MIT"
ENV PYTHONUNBUFFERED=1

RUN apk add --no-cache ca-certificates \
    && addgroup -g 1000 appuser \
    && adduser -D -u 1000 -G appuser appuser

COPY --from=builder --chown=appuser:appuser /app /app

WORKDIR /app

USER appuser

ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["webex-bot-ai"]
