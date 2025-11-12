FROM python:3.11-slim

WORKDIR /app

# Install system deps for images
RUN apt-get update && apt-get install -y build-essential libjpeg-dev zlib1g-dev && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt



