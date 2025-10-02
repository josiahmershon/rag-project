# Scoop RAG Project

A comprehensive internal AI system with Retrieval-Augmented Generation (RAG) capabilities, built with FastAPI, PostgreSQL with pgvector, and BGE embeddings.

## Architecture

- **Backend**: FastAPI API with permission-based document retrieval
- **Database**: PostgreSQL with pgvector extension for vector similarity search
- **Embeddings**: BGE (sentence-transformers/all-MiniLM-L6-v2) model for text-to-vector conversion
- **Future Integration**: Chainlit UI, LangChain orchestration

## Features

- Permission-based document access using user groups
- Vector similarity search with BGE embeddings
- PostgreSQL connection pooling
- RESTful API with FastAPI
- LLM integration
- Chainlit UI (planned)

## API Endpoints

### Health Check
```
GET /health
```

### Query Documents
```
POST /query
Content-Type: application/json

{
  "query": "What is our revenue growth?",
  "user_groups": ["finance", "executives"]
}
```

## Sample Data

The system comes with sample documents covering:
- Financial reports
- Product updates
- Security audits
- HR surveys
- Facilities updates
- ML model analysis

Each document is tagged with user groups for permission-based access.

## Next Steps

2. Add Chainlit for conversational UI
3. Implement LangChain for advanced RAG orchestration
4. Add document upload and processing capabilities
