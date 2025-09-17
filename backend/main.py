from fastapi import FastAPI

# fastapi app object
app = FastAPI(
    title="AI Assistant API",
    description="API for the AI assistant.",
    version="0.1.0",
)

@app.get("/health", tags=["System"])
def health_check():
    """Check if the API is running."""
    return {"status": "ok"}

@app.post("/query", tags=["RAG"])
def process_query(query: str):
    """Placeholder for the main RAG query logic."""
    # langchain chain will go here
    return {"response": f"You asked: {query}", "sources": []}