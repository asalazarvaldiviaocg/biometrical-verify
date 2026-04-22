.PHONY: help up down logs build seed migrate test lint fmt frontend frontend-build clean

help:
	@echo "Targets:"
	@echo "  up            Start full stack (api + worker + db + redis + minio)"
	@echo "  down          Stop stack"
	@echo "  logs          Tail api + worker logs"
	@echo "  build         Build images"
	@echo "  seed          Download anti-spoof model + run DB migrations"
	@echo "  migrate       Run alembic upgrade head"
	@echo "  test          Run pytest with coverage"
	@echo "  lint          Run ruff + mypy"
	@echo "  fmt           Auto-format with ruff"
	@echo "  frontend      Install + run vite dev server"
	@echo "  frontend-build  Build production frontend bundle"
	@echo "  clean         Remove caches + build artifacts"

env:
	@test -f .env || cp .env.example .env

up: env
	docker compose up -d --build
	@echo "API:    http://localhost:8000/docs"
	@echo "MinIO:  http://localhost:9001 (biominio / biominio-secret)"

down:
	docker compose down

logs:
	docker compose logs -f api worker

build: env
	docker compose build

seed: env
	python scripts/bootstrap.py
	docker compose run --rm api alembic upgrade head

migrate:
	docker compose run --rm api alembic upgrade head

test:
	pytest -q --cov=app --cov-report=term-missing

lint:
	ruff check app tests
	mypy app

fmt:
	ruff format app tests
	ruff check --fix app tests

frontend:
	cd frontend && npm install && npm run dev

frontend-build:
	cd frontend && npm install && npm run build

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist build
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
