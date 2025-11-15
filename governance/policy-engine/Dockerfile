FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY evaluator.py .

# Copy policies
COPY ../policies /app/policies

# Create non-root user
RUN useradd -m -u 1000 tars && \
    chown -R tars:tars /app

USER tars

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8082/health')"

# Run application
CMD ["python", "evaluator.py"]
