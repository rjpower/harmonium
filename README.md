# Harmonium

Can we make commenting on the internet better? What if we assume people have a
real need to communicate but struggle to do so well? Can we use LLMs to help us communicate our internal needs in a more constructive way?

## Installation

```
pip install uv
uv venv
source .venv/bin/activate
uv sync

uvicorn insight.app:app --reload
```

## Deployment

```
docker-compose up --build
```