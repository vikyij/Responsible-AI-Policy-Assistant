# Responsible AI Policy Assistant Backend

FastAPI service for the Responsible AI Policy Assistant RAG pipeline.

## Setup

Create a virtual environment with Python 3.11, then install dependencies:

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your Pinecone values:

```bash
cp .env.example .env
```

Your Pinecone index must use dimension `384` for `BAAI/bge-small-en-v1.5`.

## Local Model

Start Ollama and make sure the configured model is available:

```bash
ollama pull qwen2.5:7b
```

## Run API

```bash
uvicorn main:app --reload --port 8000
```

The frontend should point to:

```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

## Endpoints

- `GET /health`
- `POST /documents/upload`
- `POST /chat`
- `POST /checklist`
- `POST /gap-analysis`
