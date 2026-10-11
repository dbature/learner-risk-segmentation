# Public /predict API for the learner risk model. Build context is the
# repository root, for docker compose (service "api"), for CI and for the
# hosted copy on Render (Dockerfile path deploy/api/Dockerfile). The port comes
# from $PORT when the host sets one (Render sets 10000) and is 7860 otherwise.
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
HEALTHCHECK CMD python -c "import os, urllib.request; urllib.request.urlopen('http://localhost:%s/health' % os.environ.get('PORT', '7860'))"
CMD ["sh", "-c", "exec uvicorn src.serving.api:app --host 0.0.0.0 --port ${PORT:-7860}"]
