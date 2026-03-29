FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Hadoop integration code
COPY hdfs_client.py .

# Create HDFS simulation directory
RUN mkdir -p /tmp/hdfs/mlops/models

# Default command - show HDFS status
CMD ["python", "hdfs_client.py"]
