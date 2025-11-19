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

## Ingestion Overview

- Document upload UI (Live)  
  - Accepts PDFs, DOCX, Markdown, or text files.  
  - Runs the existing chunk/embedding pipeline.  
  - Use for manual knowledge drops and non-Oracle sources.

- Oracle promo feed (Implemented)  
  - CLI tool `ingest_oracle_feed.py` is ready.
  - Parses JSONL/CSV exports from Oracle.
  - Handles metadata extraction and vector upserts.

### Oracle Feed Contract (requested)

- Stable `chunk_id` (promo ID or hash of key columns).  
- Sentence text (plain UTF-8).  
- Key fields: product code/name, tier, start/end dates, region, security/security_level.  
- Source table/view name, snapshot timestamp or run ID.  
- Optional `last_updated` to support incremental loads.

## Roadmap & Future Ideas

### Immediate Priorities (Stability)
- **Fix Blocking I/O**: The current `upload_document` endpoint blocks the server. Need to switch to `asyncpg` or run DB calls in a thread pool.
- **Authentication**: Add proper user authentication (OAuth2/OIDC) instead of relying on client-provided groups.
- **Secrets Management**: Move all credentials to environment variables and remove defaults from code.

### Medium Term (Enhancements)
- **Hybrid Search**: Implement keyword search (BM25) alongside vector search for better precision on specific terms.
- **Reranking**: Add a cross-encoder reranking step to improve result relevance.
- **Chat History**: Persist user conversations in the database.

### Completed / In Progress
- [x] Sales pilot ingestion (CLI implemented)
- [x] User group filtering (SQL implementation active)
- [ ] General polish (Logging needs improvement)
