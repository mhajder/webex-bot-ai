FROM ghcr.io/astral-sh/uv:0.13.8@sha256:5cc1805e2426484abca4ad46ea783bab281fdb225fd82d8ce6d66a81254bac85 AS uv

FROM python:3.14.10-alpine3.24@sha256:5862d971d1f0ae26b258cefa4e893556b1d3aba81bd2ff30888704ad378bab67 AS builder

COPY --from=uv /uv /uvx /bin/
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


FROM python:3.14.10-alpine3.24@sha256:5862d971d1f0ae26b258cefa4e893556b1d3aba81bd2ff30888704ad378bab67
LABEL org.opencontainers.image.title="Webex Bot AI" \
      org.opencontainers.image.description="AI-powered Webex Bot" \
      org.opencontainers.image.url="https://github.com/mhajder/webex-bot-ai" \
      org.opencontainers.image.documentation="https://github.com/mhajder/webex-bot-ai#readme" \
      org.opencontainers.image.authors="Mateusz Hajder" \
      org.opencontainers.image.source="https://github.com/mhajder/webex-bot-ai" \
      org.opencontainers.image.vendor="Mateusz Hajder" \
      org.opencontainers.image.licenses="MIT"
ENV PYTHONUNBUFFERED=1

RUN addgroup -g 10001 appuser \
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
