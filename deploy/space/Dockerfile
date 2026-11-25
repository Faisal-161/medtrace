# Hugging Face Space image: runs the Streamlit demo on port 7860.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY --chown=user pyproject.toml README.md ./
COPY --chown=user medtrace ./medtrace
COPY --chown=user eval ./eval
RUN pip install --no-cache-dir ".[ui]"

COPY --chown=user data ./data
COPY --chown=user ui ./ui

USER user
ENV HOME=/home/user
RUN python -m medtrace.cli ingest data/sample

EXPOSE 7860
CMD ["streamlit", "run", "ui/app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
