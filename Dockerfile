# Production Dockerfile for AGY bridge service
# Antigravity/Google provider OpenAI-compatible shim
# Uses python:3.11-slim with healthcheck and restart policies

FROM python:3.11-slim

# Metadata
LABEL description="AGY bridge service - OpenAI-compatible Antigravity/Google provider shim"
LABEL maintainer="agy-bridge"

# Install minimal runtime dependencies (no build tools in production)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy application
COPY bridge.py .
COPY app_lane.py .

# The app lane is stdlib-only (http.client + urllib); NO google-antigravity / SDK
# in the container (Phase 3 hard gate: zero API keys in the bridge path).
RUN python -c "import json, http.client, urllib.request; print('stdlib ok')"
# Environment variables
ENV BIND_HOST=0.0.0.0 \
    AGY_BRIDGE_PORT=8790 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Health check: probe /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import http.client; c = http.client.HTTPConnection('127.0.0.1', 8790); c.request('GET', '/health'); r = c.getresponse(); exit(0 if r.status == 200 else 1)" || exit 1

# Run the bridge service
CMD ["python", "bridge.py"]
