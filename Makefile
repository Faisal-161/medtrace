.PHONY: install ingest api ui test eval docker

install:
	pip install -e ".[api,ui,dev]"

ingest:
	python -m medtrace.cli ingest data/sample

api:
	uvicorn medtrace.api:app --reload

ui:
	streamlit run ui/app.py

test:
	pytest -q

eval:
	python -m medtrace.cli eval

docker:
	docker compose up --build
