# Scoop RAG Project

A comprehensive internal AI system with Retrieval-Augmented Generation (RAG) capabilities, built with FastAPI, PostgreSQL with pgvector, and BGE embeddings.

To start backend:
1. Copy `backend/env.example` to `backend/.env` and fill in credentials.
2. From `backend/`:
   ```
   uvicorn main:app --host 0.0.0.0 --port 8001
   ```

To start frontend:
1. Copy `frontend/env.example` to `frontend/.env` (optional, but recommended).
2. From `frontend/`:
   ```
   chainlit run app.py --host 0.0.0.0 --port 8002 -w
   ```

Docker notes:
- Edit code in this repo.
- Copy `containers/compose/rag-project/app/env/backend.env` and `frontend.env` with the correct credentials before running compose (`DB_PASSWORD` in particular).
- Rebuild backend container: `cd /fastpool/containers/compose/rag-project/app && docker compose build rag-backend`.
- Rebuild frontend container if needed: same command but `rag-frontend`.
- Restart services: `docker compose up -d rag-backend rag-frontend`.
- Logs: `docker compose logs -f rag-backend` or `... rag-frontend`.

## Chat Experience

- Per-session history is retained automatically (up to 40 turns) and included with every backend call.
- Attachments (PDF, DOCX, TXT, MD) are inlined per message and never persisted to the vector store.
- Attachments referenced in an answer are surfaced in the sources list; other citations are limited to documents explicitly mentioned in the reply.
- Chainlit notifies users when history is trimmed or when attachments are registered for the session.

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
- **Authentication**: Add proper user authentication (OAuth2/OIDC) instead of relying on client-provided groups.
- **Secrets Management**: Move all credentials to environment variables and remove defaults from code.

### Medium Term (Enhancements)
- **Hybrid Search**: Implement keyword search (BM25) alongside vector search for better precision on specific terms.
- **Reranking**: Add a cross-encoder reranking step to improve result relevance.
- **Persisted Chat History**: Store conversations for users across sessions (requires auth).
- **Logging polish**: Standardize log formats/levels and surface key events (uploads, chats, errors).

### Completed / In Progress
- [x] Per-session chat history with trimming & attachment surfacing
- [x] Frontend accepts transient attachments (Chainlit UI)
- [x] Backend ingests attachments per turn without persisting to DB
- [x] Source citations limited to top relevant documents
- [x] Sales pilot ingestion (CLI implemented)
- [x] User group filtering (SQL implementation active)
- [ ] Async upload pipeline (replace blocking DB writes)
- [ ] Persisted chat history + auth UX
