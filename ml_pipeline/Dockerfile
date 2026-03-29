FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy pipeline code
COPY pipeline.py .
COPY models/ ./models/ 2>/dev/null || true

# Create models directory
RUN mkdir -p /app/models

# Default command - train a model
CMD ["python", "pipeline.py"]
