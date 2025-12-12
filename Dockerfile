# Multi-stage Dockerfile for T.A.R.S. Observability Framework (Phase 14.6)
# Produces a production-ready container with all CLI tools installed

# Stage 1: Builder
FROM python:3.11-slim as builder

LABEL maintainer="Veleron Dev Studios <engineering@veleron.dev>"
LABEL description="T.A.R.S. Phase 14.6 - Post-GA 7-Day Stabilization & Retrospective Framework"
LABEL version="1.0.2-pre"

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy dependency files first (for Docker layer caching)
COPY requirements-dev.txt .
COPY pyproject.toml .
COPY README.md .

# Copy source code
COPY tars_observability/ tars_observability/
COPY observability/ observability/
COPY scripts/ scripts/
COPY tests/ tests/

# Install Python dependencies and build package
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements-dev.txt && \
    pip install --no-cache-dir -e .

# Stage 2: Runtime
FROM python:3.11-slim

LABEL maintainer="Veleron Dev Studios <engineering@veleron.dev>"
LABEL description="T.A.R.S. Phase 14.6 - Post-GA 7-Day Stabilization & Retrospective Framework"
LABEL version="1.0.2-pre"

# Create non-root user for security
RUN groupadd -r tars && useradd -r -g tars -u 1000 tars

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --from=builder /build/tars_observability /app/tars_observability
COPY --from=builder /build/observability /app/observability
COPY --from=builder /build/scripts /app/scripts
COPY --from=builder /build/pyproject.toml /app/

# Create directories for data persistence
RUN mkdir -p /data/ga_kpis /data/stability /data/anomalies /data/regression /data/health /data/output && \
    chown -R tars:tars /data /app

# Switch to non-root user
USER tars

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/usr/local/bin:${PATH}" \
    TARS_DATA_DIR=/data \
    TARS_OUTPUT_DIR=/data/output

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from tars_observability import get_version_string; print(get_version_string())" || exit 1

# Default command: run retrospective generator in auto mode
ENTRYPOINT ["tars-retro"]
CMD ["--auto", "--output-dir", "/data/output"]

# Expose volume for data persistence
VOLUME ["/data"]

# Usage examples (as labels):
LABEL usage.ga-kpi="docker run -v $(pwd)/data:/data tars-observability tars-ga-kpi --prometheus-url http://prometheus:9090"
LABEL usage.stability="docker run -v $(pwd)/data:/data tars-observability tars-stability-monitor --day-number 1"
LABEL usage.anomaly="docker run -v $(pwd)/data:/data tars-observability tars-anomaly-detector"
LABEL usage.health="docker run -v $(pwd)/data:/data tars-observability tars-health-report --day-number 1"
LABEL usage.regression="docker run -v $(pwd)/data:/data tars-observability tars-regression-analyzer"
LABEL usage.retro="docker run -v $(pwd)/data:/data tars-observability tars-retro --auto"
