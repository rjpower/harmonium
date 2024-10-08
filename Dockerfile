FROM python:3.11-slim
WORKDIR /app
RUN pip install uv
COPY pyproject.toml .
RUN uv venv && uv sync --no-install-project
COPY . .
RUN uv sync
EXPOSE 8000

CMD [".venv/bin/uvicorn", "harmonium.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
