# Isolated Dockerfile for H2O AutoML (requires Java runtime)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /worker

RUN pip install --upgrade pip wheel && \
    pip install "numpy<2.0.0" pandas pyarrow h2o==3.46.0.1

COPY src/acbs/automl/contracts.py /worker/acbs/automl/contracts.py
COPY src/acbs/automl/workers/h2o_worker.py /worker/acbs/automl/workers/h2o_worker.py

CMD ["python", "-m", "acbs.automl.workers.h2o_worker"]
