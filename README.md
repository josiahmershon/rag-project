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


Next steps (notes to self):
- Sales pilot parsing overhaul:
  - Build a parser that reads promo tables row-by-row and emits one sentence per product entry.
  - Capture key fields alongside the chunk (product code, product name, tier, date window, source file, security group).
  - Write a quick test script against sample promos to verify each chunk looks atomic.
  - Once happy, re-embed the sales corpus and spot-check retrieval results.
- User group filtering:
  - Map AD distribution lists to simple RAG groups (finance, sales, engineering, legal, etc.).
  - Thread group claims through the frontend (Chainlit) once LDAP sign-in exists.
  - Apply list-based filters in SQL before similarity search so the LLM sees less noise.
- General polish:
  - Add chunk IDs + timestamps to the vector table for debugging.
  - Keep an eye on ingestion performance; consider batching inserts and async uploads later.
  - Log retrieval hits/misses so I can tune chunking rules.
