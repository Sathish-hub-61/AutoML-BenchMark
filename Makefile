.PHONY: install test lint api ui
install:
	python -m pip install -e ".[dev]"
test:
	pytest
lint:
	ruff check src tests app
api:
	uvicorn acbs.api.main:app --reload --port 8000
ui:
	streamlit run app/streamlit_app.py
up:
	docker compose up --build
