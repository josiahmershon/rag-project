# Scoop RAG Project

A comprehensive internal AI system with Retrieval-Augmented Generation (RAG) capabilities, built with FastAPI, PostgreSQL with pgvector, and BGE embeddings.

To start backend:
from backend dir:
uvicorn main:app --host 0.0.0.0 --port 8001

To start frontend:
from frontend dir:
chainlit run app.py --host 0.0.0.0 --port 8002 -w

Docker notes:
- Edit code in this repo.
- Rebuild backend container: `cd /fastpool/containers/compose/rag-project/app && docker compose build rag-backend`.
- Rebuild frontend container if needed: same command but `rag-frontend`.
- Restart services: `docker compose up -d rag-backend rag-frontend`.
- Logs: `docker compose logs -f rag-backend` or `... rag-frontend`.
