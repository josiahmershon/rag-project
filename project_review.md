# Scoop RAG Project Review

## Executive Summary
The Scoop RAG project provides a solid foundation for an internal AI assistant. The core RAG pipeline (Ingestion -> Embedding -> Vector Storage -> Retrieval -> Generation) is functional. The system supports multiple file types and includes a specialized ingestion pipeline for Oracle promo feeds. However, there are critical performance and security issues that need addressing before broader deployment.

## Glaring Issues

### 1. Critical Performance Bottleneck (Blocking I/O)
- **Issue**: The `upload_document` endpoint in `backend/main.py` is defined as `async def` but performs synchronous database operations using `psycopg2`.
- **Impact**: This blocks the entire FastAPI event loop. During a file upload or database insertion, the server cannot handle *any* other requests (including health checks or queries).
- **Fix**: Use `run_in_threadpool` for blocking calls or switch to an asynchronous database driver like `asyncpg`.

### 2. Security & Authentication
- **Issue**: There is no real authentication. The system relies on the client to self-report `user_groups`.
- **Risk**: Any user can send a request claiming to be in the "executives" group and access restricted documents.
- **Fix**: Implement proper authentication (e.g., OAuth2/OIDC) and derive groups from the authenticated user's session/token.

### 3. Hardcoded Credentials & Configuration
- **Issue**: `backend/settings.py` and `backend/main.py` contain default/dummy credentials (e.g., `dummy-key` for vLLM) and rely on `psycopg2` connection parameters that might be hardcoded if `.env` is missing.
- **Fix**: Enforce environment variable usage for all secrets. Fail startup if critical secrets are missing.

### 4. Input Sanitization
- **Issue**: `sanitize_model_output` uses regex to strip HTML tags. This is generally fragile and can be bypassed.
- **Fix**: Use a dedicated HTML sanitization library (like `bleach`) to safely strip dangerous tags while preserving formatting.

## Implementation Status (vs README)

| Feature | Status | Notes |
| :--- | :--- | :--- |
| **Document Upload UI** | ✅ Live | Supports PDF, DOCX, MD, TXT. |
| **Oracle Promo Feed** | ✅ Implemented | CLI (`ingest_oracle_feed.py`) and Parser (`oracle_feed.py`) are complete. |
| **User Group Filtering** | ✅ Implemented | SQL-level filtering using Postgres arrays (`&&` operator) is working. |
| **Sales Pilot Ingestion** | ✅ Ready | The CLI is built and ready for testing with real data. |
| **General Polish** | ⚠️ Partial | Timestamps exist, but structured logging for retrieval tuning is minimal. |

## Future Ideas & Recommendations

### Short Term (Stability & Security)
1.  **Switch to Async DB**: Migrate to `asyncpg` or `SQLAlchemy` (async) to fix the blocking I/O issue.
2.  **Add Authentication**: Integrate a simple auth layer (even Basic Auth temporarily) to protect the API.
3.  **Structured Logging**: Implement structured JSON logging (e.g., using `structlog`) to track retrieval quality (query vs. retrieved chunks).

### Medium Term (Features)
4.  **Hybrid Search**: Combine vector similarity with keyword search (BM25) using Postgres `tsvector` for better precision on specific terms (e.g., product codes).
5.  **Reranking Step**: Add a cross-encoder reranking step after retrieval to improve context relevance before sending to the LLM.
6.  **Chat History**: Persist chat sessions in the database so users can access past conversations across devices.

### Long Term (Advanced)
7.  **Evaluation Pipeline**: Integrate a framework like Ragas or TruLens to automatically evaluate RAG performance (faithfulness, answer relevance).
8.  **Admin Dashboard**: Create a UI for managing uploaded documents (delete/update) and viewing usage stats.
