# Isolated Linux Dockerfile for Auto-sklearn / Auto-sklearn 2
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    swig \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /worker

RUN pip install --upgrade pip wheel setuptools && \
    pip install "numpy<2.0.0" pandas pyarrow scikit-learn==0.24.2 "auto-sklearn==0.15.0"

COPY src/acbs/automl/contracts.py /worker/acbs/automl/contracts.py
COPY src/acbs/automl/workers/autosklearn_worker.py /worker/acbs/automl/workers/autosklearn_worker.py

CMD ["python", "-m", "acbs.automl.workers.autosklearn_worker"]
