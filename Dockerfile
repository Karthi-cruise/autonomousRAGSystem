FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_XET=1 \
    LIGHTWEIGHT_DEPLOY=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY . .

RUN python scripts/bootstrap_sample_db.py
RUN python -m src.main --ingest

EXPOSE 7860

CMD ["sh", "-c", "uvicorn src.api:create_app --factory --host 0.0.0.0 --port ${PORT:-7860}"]
