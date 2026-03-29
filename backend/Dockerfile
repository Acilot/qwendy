FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Create directories for models and HDFS storage
RUN mkdir -p /workspace/models /workspace/hdfs_storage /workspace/pipelines

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
