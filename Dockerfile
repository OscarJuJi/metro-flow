# Serving image: it copies the artifacts, it does not create them.
#
# Training happens in the monthly GitHub Actions job, which commits the four
# small files under artifacts/. Building the image therefore needs no network
# access, no LightGBM training run and no credentials -- and the image that
# ships is byte-for-byte the model that was evaluated.

FROM python:3.11-slim AS build

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir ".[api]"

FROM python:3.11-slim AS runtime

# LightGBM links against libgomp even when it is only ever asked to predict.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# The package is installed into the venv rather than run from a source tree, so
# PROJECT_ROOT resolves inside site-packages, not to /app. The artifact location
# is therefore stated explicitly instead of inferred: without this the service
# starts happily and then answers 503 to every request.
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PORT=7860 \
    METRO_PULSE_ARTIFACTS_DIR=/app/artifacts \
    METRO_PULSE_DATA_DIR=/app/data

WORKDIR /app
COPY --from=build /opt/venv /opt/venv
COPY artifacts ./artifacts

RUN useradd --create-home --uid 1000 app && chown -R app:app /app
USER app

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

CMD ["sh", "-c", "uvicorn metro_pulse.api.app:app --host 0.0.0.0 --port ${PORT}"]
