# Scoop RAG Project

A comprehensive internal AI system with Retrieval-Augmented Generation (RAG) capabilities, built with FastAPI, PostgreSQL with pgvector, and BGE embeddings.

## Features

- **Intelligent Document Search**: Vector similarity search using pgvector
- **Permission-Based Access**: Group-based document access control
- **Multiple Document Formats**: Support for TXT, MD, PDF, DOCX
- **LLM Integration**: Powered by vLLM with LangChain
- **Web Interface**: Document upload and management UI
- **Chat Interface**: Chainlit-based conversational UI

## Tech Stack

- **Backend**: FastAPI, Python 3.11
- **Database**: PostgreSQL 16 with pgvector extension
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **LLM**: vLLM with Qwen3-32B-AWQ
- **Frontend**: Chainlit
- **Deployment**: Docker & Docker Compose

## Quick Start with Docker (Recommended)

**See [DOCKER_SETUP.md](DOCKER_SETUP.md) for detailed instructions.**

```bash
# 1. Clone and navigate to project
cd rag-project

# 2. Create .env file
cp backend/env.example .env
# Edit .env and set DB_PASSWORD

# 3. Start all services
docker-compose up -d

# 4. Load sample data
docker-compose exec backend python ingest.py

# 5. Access the application
# Backend: http://localhost:8001
# Frontend: http://localhost:8002
```

## Manual Setup (Without Docker)

### Prerequisites
- Python 3.11+
- PostgreSQL 16 with pgvector extension
- vLLM server running (for LLM inference)

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp env.example .env
# Edit .env with your database credentials

# Create database schema
psql -U postgres -d postgres -f schema.sql

# Load sample documents
python ingest.py

# Start backend server
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

### Frontend Setup

```bash
cd frontend

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file (optional)
echo "BACKEND_API_URL=http://localhost:8001" > .env
echo "DEFAULT_USER_GROUPS=executives,engineering" >> .env

# Start Chainlit
chainlit run app.py --host 0.0.0.0 --port 8002 -w
```

## API Endpoints

### Query Endpoints
- `POST /query` - Standard RAG query (similarity threshold: 0.3)
- `POST /query-precise` - Strict filtering (threshold: 0.5)
- `POST /query-lc` - LangChain-powered query

### Document Management
- `GET /upload` - Upload form UI
- `POST /upload` - Upload and ingest documents
- `GET /database` - View all documents
- `GET /docs/preview/{filename}` - Preview document chunks
- `DELETE /docs/delete/{filename}` - Delete document

### System
- `GET /health` - Backend health check
- `GET /health/vllm` - vLLM connection check

## Configuration

Environment variables (`.env` file):

```bash
# Database
DB_HOST=localhost
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=your_password_here

# vLLM
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_MODEL=Qwen/Qwen3-32B-AWQ

# Optional: Override defaults
MAX_FILE_SIZE=10485760  # 10MB
MAX_QUERY_LENGTH=1000
DEFAULT_CHUNK_SIZE=512
DEFAULT_CHUNK_OVERLAP=50
DEFAULT_SIMILARITY_THRESHOLD=0.3
```

## Project Structure

```
rag-project/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── settings.py          # Configuration
│   ├── utils.py             # Utility functions
│   ├── ingest.py            # Data ingestion script
│   ├── schema.sql           # Database schema
│   ├── Dockerfile           # Backend container
│   └── requirements.txt
├── frontend/
│   ├── app.py               # Chainlit UI
│   ├── Dockerfile           # Frontend container
│   └── requirements.txt
├── docker-compose.yml       # Orchestration
├── DOCKER_SETUP.md         # Docker guide
└── README.md               # This file
```

## Security Features

- Input validation for queries and uploads
- File size limits
- XSS protection with HTML escaping
- Group-based access control
- CORS configuration
- Output sanitization

## Development

### Running Tests

```bash
cd backend
python test_retrieval.py
```

### Database Management

```bash
# Connect to database
docker-compose exec postgres psql -U postgres

# View documents
SELECT source_path, COUNT(*) FROM vector_index GROUP BY source_path;

# Check vector index
SELECT COUNT(*) FROM vector_index;
```

### Rebuilding Containers

```bash
docker-compose build --no-cache
docker-compose up -d
```

## Troubleshooting

See [DOCKER_SETUP.md](DOCKER_SETUP.md) for detailed troubleshooting.

**Common Issues:**

1. **Database connection failed**: Check PostgreSQL is running
2. **vLLM unavailable**: Ensure vLLM server is accessible
3. **Port conflicts**: Change ports in docker-compose.yml
4. **Out of memory**: Reduce model size or increase Docker memory limit

## Contributing

1. Create feature branch
2. Make changes
3. Test thoroughly
4. Submit pull request

## License

Internal use only.

## Support

For issues or questions, contact the development team.
