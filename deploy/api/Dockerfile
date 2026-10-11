# Public /predict API for the learner risk model, built for a Hugging Face
# Docker Space (port 7860). Build context is the repository root, both for
# docker compose (service "api") and for the Space, where
# scripts/deploy_hf_space.py uploads the same layout (deploy/api, src, models).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/app/models EXPLANATION_STORE=/tmp/explanations/predictions.jsonl

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 1000 api && mkdir -p /tmp/explanations && chown api /tmp/explanations
WORKDIR /app
COPY --chown=api deploy/api/requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY --chown=api params.yaml ./
COPY --chown=api src ./src
COPY --chown=api models ./models
USER api
EXPOSE 7860
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"
CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "7860"]
