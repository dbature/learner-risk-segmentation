# Stakeholder dashboard (Module 5) for docker compose. The public copy runs on
# Streamlit Community Cloud from the same files.
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY dashboard/requirements.txt dashboard/requirements.txt
RUN pip install -r dashboard/requirements.txt
COPY --chown=app params.yaml ./
COPY --chown=app .streamlit ./.streamlit
COPY --chown=app src ./src
COPY --chown=app models ./models
COPY --chown=app dashboard ./dashboard
USER app
EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
