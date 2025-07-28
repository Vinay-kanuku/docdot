# Use official Python slim image
FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc g++ libffi-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY document_parser.py feature_extractor.py semantic_analyzer.py ensemble_scorer.py main_system.py .

RUN mkdir -p /app/input /app/output

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["python", "main_system.py"]