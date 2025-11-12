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

- Document upload UI (already live)  
  - Accepts PDFs, DOCX, Markdown, or text files.  
  - Runs the existing chunk/embedding pipeline.  
  - Use for manual knowledge drops and non-Oracle sources.

- Oracle promo feed (in progress)  
  - Oracle team will return pre-summarized rows: one sentence per promo plus metadata.  
  - We ingest via a lightweight script/CLI (JSON Lines or CSV).  
  - Script handles validation, embeddings, upserts into `vector_index`.

### Oracle Feed Contract (requested)

- Stable `chunk_id` (promo ID or hash of key columns).  
- Sentence text (plain UTF-8).  
- Key fields: product code/name, tier, start/end dates, region, security/security_level.  
- Source table/view name, snapshot timestamp or run ID.  
- Optional `last_updated` to support incremental loads.

## Next steps (notes to self):
- Sales pilot ingestion:
  - Build the CLI that reads the Oracle sentence feed and writes chunks/metadata.
  - Store the raw feed alongside each run (checksum + metadata table for audits).
  - Re-embed the sales corpus and spot-check retrieval results once feed arrives.
- User group filtering:
  - Map AD distribution lists to simple RAG groups (finance, sales, engineering, legal, etc.).
  - Thread group claims through the frontend (Chainlit) once LDAP sign-in exists.
  - Apply list-based filters in SQL before similarity search so the LLM sees less noise.
- General polish:
  - Add chunk IDs + timestamps to the vector table for debugging.
  - Keep an eye on ingestion performance; consider batching inserts and async uploads later.
  - Log retrieval hits/misses so I can tune chunking rules.
