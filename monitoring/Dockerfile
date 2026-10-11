# Monitoring dashboard for docker compose. The public copy runs on Streamlit
# Community Cloud from the same files.
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY monitoring/requirements.txt monitoring/requirements.txt
RUN pip install -r monitoring/requirements.txt
COPY --chown=app .streamlit ./.streamlit
COPY --chown=app monitoring ./monitoring
USER app
EXPOSE 8502
CMD ["streamlit", "run", "monitoring/app.py", "--server.address=0.0.0.0", "--server.port=8502", "--server.headless=true"]
