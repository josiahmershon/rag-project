# main.py
import psycopg2.pool
from contextlib import contextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# imports settings from settings.py
from settings import settings

# define the structure of the incoming request JSON
class QueryRequest(BaseModel):
    query: str
    user_groups: List[str]

# define the structure for a single source document
class DocumentSource(BaseModel):
    text: str
    source: str

# define the structure of the final JSON response
class QueryResponse(BaseModel):
    response: str
    sources: List[DocumentSource]

# create the FastAPI application
app = FastAPI(
    title="AI Assistant API",
    description="API for the AI assistant.",
    version="0.1.0",
)

# create the connection pool when the application starts up
# this pool will manage all database connections
db_pool = psycopg2.pool.SimpleConnectionPool(
    minconn=1,
    maxconn=20,
    host=settings.db_host,
    database=settings.db_name,
    user=settings.db_user,
    password=settings.db_password
)

@contextmanager
def get_db_connection():
    """
    A helper function to get a connection from the pool and ensure
    it's returned, even if an error occurs.
    """
    conn = db_pool.getconn()
    try:
        yield conn
    finally:
        db_pool.putconn(conn)


def get_relevant_chunks(query_embedding: List[float], user_groups: List[str]) -> List[dict]:
    """
    Connects to the database via the pool and retrieves chunks.
    """
    try:
        with get_db_connection() as conn:  # borrows a connection from the pool
            cursor = conn.cursor()
            
            print(f"Searching for user with groups: {user_groups}")
            
            # use the query_embedding and user_groups as parameters for perm-filtered vector search
            
            cursor.close()
            # for rn it will just return dummy data 
            return [{"text": "Chunk 1 text...", "source": "/path/to/doc1"}]

    except Exception as e:
        print(f"Database error: {e}")
        return []


@app.get("/health", tags=["System"])
def health_check():
    """Check if the API is running."""
    return {"status": "ok"}


@app.post("/query", tags=["RAG"], response_model=QueryResponse)
def process_query(request: QueryRequest):
    """
    Takes a user query and user groups, retrieves relevant context,
    and returns an answer.
    """
    # embed the user's query
    dummy_embedding = [] 

    # retrieve relevant chunks from the database
    chunks = get_relevant_chunks(dummy_embedding, request.user_groups)
    
    # pass the query and chunks to the LLM
    llm_response = f"This is the LLM's future answer to '{request.query}'"
    
    # return the final response
    return {"response": llm_response, "sources": chunks}