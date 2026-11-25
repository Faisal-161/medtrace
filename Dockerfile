FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY medtrace ./medtrace
COPY eval ./eval
RUN pip install --no-cache-dir ".[api,ui]"

COPY data ./data
COPY ui ./ui

# Build the demo index at image build time so the container starts ready.
RUN python -m medtrace.cli ingest data/sample

EXPOSE 8000
CMD ["uvicorn", "medtrace.api:app", "--host", "0.0.0.0", "--port", "8000"]
