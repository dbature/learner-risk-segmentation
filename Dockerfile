# Learner risk segmentation: data pipeline image.
# Python 3.11 matches environment.yml and CI, so a result in the container is
# the same result as on a laptop or in GitHub Actions.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PREFECT_HOME=/tmp/prefect \
    PREFECT_SERVER_ANALYTICS_ENABLED=false \
    PREFECT_SERVER_EPHEMERAL_STARTUP_TIMEOUT_SECONDS=120

WORKDIR /app

# Dependencies first, so editing pipeline code does not reinstall them.
COPY requirements-pipeline.txt .
RUN pip install -r requirements-pipeline.txt

COPY params.yaml pytest.ini ./
COPY src ./src
COPY flows ./flows
COPY gx ./gx
COPY tests ./tests

# Run as an unprivileged user. Learner data is mounted at run time, never
# baked into the image, and the salt arrives as an environment variable.
RUN useradd --create-home --uid 10001 pipeline \
    && mkdir -p data logs reports \
    && chown -R pipeline /app
USER pipeline

VOLUME ["/app/data", "/app/logs", "/app/reports"]

# Default: run the Prefect flow. Override to run the tests:
#   docker run --rm learner-risk-pipeline pytest -q
CMD ["python", "-m", "flows.oulad_pipeline"]
