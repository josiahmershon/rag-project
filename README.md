# Scoop RAG Project

A comprehensive internal AI system with Retrieval-Augmented Generation (RAG) capabilities, built with FastAPI, PostgreSQL with pgvector, and BGE embeddings.

To start backend:
from backend dir:
uvicorn main:app --host 0.0.0.0 --port 8001

To start frontend:
from frontend dir:
chainlit run app.py --host 0.0.0.0 --port 8002 -w
