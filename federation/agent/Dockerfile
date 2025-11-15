FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 tars && \
    chown -R tars:tars /app

USER tars

# Expose Prometheus metrics port
EXPOSE 9091

# Run application
CMD ["python", "sidecar.py"]
