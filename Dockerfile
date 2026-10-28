# Base Dockerfile for Auto Classifier Benchmarking System (acbs)
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies first for caching
COPY pyproject.toml README.md ./
RUN pip install --upgrade pip setuptools wheel && \
    pip install -e ".[dev,postgres]"

# Copy source and application code
COPY . .

EXPOSE 8000 8501

CMD ["uvicorn", "acbs.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
