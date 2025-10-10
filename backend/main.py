# MAIN.PY
import psycopg2.pool
from contextlib import contextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
import time
import uuid
from openai import OpenAI

# LangChain imports
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# imports settings from settings.py
from settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# define the structure of the incoming request json
class QueryRequest(BaseModel):
    query: str
    user_groups: List[str]

# define the structure for a single source document
class DocumentSource(BaseModel):
    text: str
    source: str

# define the structure of the final json response
class QueryResponse(BaseModel):
    response: str
    sources: List[DocumentSource]

# load embedding model for 1536-dimensional embeddings
logger.info("Loading embedding model...")
# using model that generates 1536 dimensions (openai text-embedding-ada-002 compatible)
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# we'll pad the embeddings to 1536 dimensions to match the database schema
logger.info("Embedding model loaded successfully")

# initialize vLLM client
logger.info("Initializing vLLM client...")
vllm_client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy-key"  # vLLM doesn't require real API key
)
logger.info("vLLM client initialized successfully")

# initialize LangChain ChatOpenAI client (same vLLM endpoint)
logger.info("Initializing LangChain ChatOpenAI client...")
langchain_llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy-key",
    model=settings.vllm_model,
    temperature=0.7,
    max_tokens=1000
)
logger.info("LangChain ChatOpenAI client initialized successfully")

def sanitize_model_output(text: str) -> str:
    """Remove chain-of-thought blocks and basic unsafe HTML tags from model output.

    - Strips <think> ... </think> blocks entirely
    - Removes script/style/iframe/object/embed tags if present
    - Trims excessive surrounding whitespace
    """
    try:
        import re

        if not isinstance(text, str):
            return ""

        # Remove <think>...</think> blocks (non-greedy, dotall, case-insensitive)
        text = re.sub(r"<\s*think\b[\s\S]*?>[\s\S]*?<\s*/\s*think\s*>", "", text, flags=re.IGNORECASE)
        # Also remove lone <think> or </think> tags if any remain
        text = re.sub(r"<\/?\s*think\s*>", "", text, flags=re.IGNORECASE)

        # Drop dangerous tags entirely (keep inner text out to be safe)
        for tag in ["script", "style", "iframe", "object", "embed"]:
            pattern = rf"<\s*{tag}\b[\s\S]*?>[\s\S]*?<\s*/\s*{tag}\s*>"
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # Basic cleanup of stray HTML tags that might affect rendering
        # Keep markdown; only remove remaining HTML tags conservatively
        text = re.sub(r"</?\w+[^>]*>", "", text)

        return text.strip()
    except Exception:
        # Fail-closed: return original text rather than crashing request
        return (text or "").strip()

def get_relevant_chunks_langchain(query: str, user_groups: List[str], limit: int = 3, similarity_threshold: float = 0.7) -> List[Document]:
    """
    LangChain-compatible retriever that returns Document objects.
    Uses the same SQL logic as get_relevant_chunks but returns LangChain Documents.
    """
    try:
        # Generate embedding for the query
        query_embedding = embedding_model.encode(query).tolist()
        query_embedding = pad_embedding_to_1536(query_embedding)
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            logger.info(f"LangChain: Searching for user with groups: {user_groups}")
            
            # Convert embedding to postgresql array format
            embedding_array = "[" + ",".join(map(str, query_embedding)) + "]"
            
            # SQL query with similarity threshold
            query_sql = """
            SELECT chunk_text, source_path, 
                   1 - (embedding <=> %s::vector) as similarity
            FROM vector_index 
            WHERE allowed_groups && %s::text[]
            AND (1 - (embedding <=> %s::vector)) >= %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """
            
            cursor.execute(query_sql, (embedding_array, user_groups, embedding_array, similarity_threshold, embedding_array, limit))
            results = cursor.fetchall()
            
            # Convert to LangChain Documents
            documents = []
            for row in results:
                doc = Document(
                    page_content=row[0],
                    metadata={
                        "source": row[1],
                        "similarity": float(row[2])
                    }
                )
                documents.append(doc)
            
            logger.info(f"LangChain: Retrieved {len(documents)} relevant documents")
            cursor.close()
            return documents

    except Exception as e:
        logger.error(f"LangChain database error: {e}")
        return []

# LangChain prompt template
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant that answers questions based on provided context from internal company documents. 

Guidelines:
- Answer based ONLY on the provided context
- If the context doesn't contain enough information, say so
- Be concise but informative
- Cite sources when relevant
- If asked about something not in the context, politely explain that you don't have that information
- Do not include any internal reasoning or chain-of-thought. Provide only the final answer and brief citations."""),
    ("user", """Context from company documents:
{context}

Question: {question}

Please provide a helpful answer based on the context above.""")
])

def pad_embedding_to_1536(embedding: List[float]) -> List[float]:
    """Pad embedding to 1536 dimensions to match database schema."""
    current_dim = len(embedding)
    if current_dim == 1536:
        return embedding
    elif current_dim < 1536:
        # pad with zeros
        padding = [0.0] * (1536 - current_dim)
        return embedding + padding
    else:
        # truncate if somehow larger
        return embedding[:1536]

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
    a helper function to get a connection from the pool and ensure
    it's returned, even if an error occurs.
    """
    conn = db_pool.getconn()
    try:
        yield conn
    finally:
        db_pool.putconn(conn)


def get_relevant_chunks(query_embedding: List[float], user_groups: List[str], limit: int = 3, similarity_threshold: float = 0.7) -> List[dict]:
    """
    connects to the database via the pool and retrieves chunks using vector similarity search.
    """
    try:
        with get_db_connection() as conn:  # borrows a connection from the pool
            cursor = conn.cursor()
            
            logger.info(f"Searching for user with groups: {user_groups}")
            
            # convert embedding to postgresql array format
            embedding_array = "[" + ",".join(map(str, query_embedding)) + "]"
            
            # create user groups filter for permission-based search
            user_groups_str = "','".join(user_groups)
            
            # first, get all matching documents with similarity scores for debugging
            debug_query = """
            SELECT chunk_text, source_path, 
                   1 - (embedding <=> %s::vector) as similarity
            FROM vector_index 
            WHERE allowed_groups && %s::text[]
            ORDER BY embedding <=> %s::vector
            LIMIT 10
            """
            
            cursor.execute(debug_query, (embedding_array, user_groups, embedding_array))
            debug_results = cursor.fetchall()
            
            logger.info(f"Debug - All matching documents and their similarity scores:")
            for row in debug_results:
                logger.info(f"  - {row[1]}: {row[2]:.4f}")
            
            # now apply similarity threshold
            query = """
            SELECT chunk_text, source_path, 
                   1 - (embedding <=> %s::vector) as similarity
            FROM vector_index 
            WHERE allowed_groups && %s::text[]
            AND (1 - (embedding <=> %s::vector)) >= %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """
            
            cursor.execute(query, (embedding_array, user_groups, embedding_array, similarity_threshold, embedding_array, limit))
            results = cursor.fetchall()
            
            # convert results to chunks
            chunks = []
            for row in results:
                chunks.append({
                    "text": row[0],
                    "source": row[1],
                    "similarity": float(row[2])
                })
            
            logger.info(f"Retrieved {len(chunks)} relevant chunks (similarity >= {similarity_threshold})")
            cursor.close()
            return chunks

    except Exception as e:
        logger.error(f"Database error: {e}")
        return []


def generate_response_with_vllm(query: str, context_chunks: List[dict]) -> str:
    """
    Generate a response using vLLM based on the query and retrieved context chunks.
    """
    try:
        # prepare context from retrieved chunks
        if not context_chunks:
            context = "No relevant documents found."
        else:
            context = "\n\n".join([f"Source: {chunk['source']}\nContent: {chunk['text']}" for chunk in context_chunks])
        
        # create system prompt for RAG
        system_prompt = """You are a helpful AI assistant that answers questions based on provided context from internal company documents. 
        
        Guidelines:
        - Answer based ONLY on the provided context
        - If the context doesn't contain enough information, say so
        - Be concise but informative
        - Cite sources when relevant
        - If asked about something not in the context, politely explain that you don't have that information"""
        
        # prepare the user message with context
        user_message = f"""Context from company documents:
{context}

Question: {query}

Please provide a helpful answer based on the context above."""
        
        # make request to vLLM
        request_id = str(uuid.uuid4())[:8]
        logger.info(f"[{request_id}] Making vLLM request for query: {query[:50]}...")
        start_time = time.time()
        response = vllm_client.chat.completions.create(
            model=settings.vllm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=1000,
            top_p=0.9
        )
        response_time = time.time() - start_time
        logger.info(f"[{request_id}] vLLM response generated in {response_time:.2f}s")
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Error generating response with vLLM: {e}")
        return f"I apologize, but I encountered an error while generating a response: {str(e)}"


@app.get("/health", tags=["System"])
def health_check():
    """checks if the API is running"""
    return {"status": "ok"}

@app.get("/health/vllm", tags=["System"])
def vllm_health_check():
    """checks if vLLM is accessible"""
    try:
        # simple test request to vLLM
        response = vllm_client.chat.completions.create(
            model=settings.vllm_model,
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1
        )
        return {"status": "ok", "vllm": "accessible"}
    except Exception as e:
        return {"status": "error", "vllm": "unavailable", "error": str(e)}


@app.post("/query", tags=["RAG"], response_model=QueryResponse)
def process_query(request: QueryRequest):
    """
    takes user query and user groups, retrieves relevant context,
    and returns an answer.
    """
    try:
        # embed the user's query using embedding model
        logger.info(f"Processing query: {request.query}")
        query_embedding = embedding_model.encode(request.query).tolist()
        query_embedding = pad_embedding_to_1536(query_embedding)
        logger.info(f"Generated embedding with {len(query_embedding)} dimensions")

        # retrieve relevant chunks from the database with moderate precision
        chunks = get_relevant_chunks(query_embedding, request.user_groups, limit=3, similarity_threshold=0.3)
        
        # convert chunks to DocumentSource format
        sources = [DocumentSource(text=chunk["text"], source=chunk["source"]) for chunk in chunks]
        
        # generate response using vLLM
        llm_response = generate_response_with_vllm(request.query, chunks)
        llm_response = sanitize_model_output(llm_response)
        
        logger.info(f"Returning response with {len(sources)} sources")
        return {"response": llm_response, "sources": sources}
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return {"response": f"Error processing query: {str(e)}", "sources": []}


@app.post("/query-precise", tags=["RAG"], response_model=QueryResponse)
def process_query_precise(request: QueryRequest):
    """
    takes user query and user groups, retrieves only highly relevant context,
    and returns an answer with stricter similarity filtering.
    """
    try:
        # embed the user's query using embedding model
        logger.info(f"Processing precise query: {request.query}")
        query_embedding = embedding_model.encode(request.query).tolist()
        query_embedding = pad_embedding_to_1536(query_embedding)
        logger.info(f"Generated embedding with {len(query_embedding)} dimensions")

        # retrieve only the most relevant chunks with moderate similarity thresholds    
        chunks = get_relevant_chunks(query_embedding, request.user_groups, limit=2, similarity_threshold=0.5)
        
        # convert chunks to DocumentSource format
        sources = [DocumentSource(text=chunk["text"], source=chunk["source"]) for chunk in chunks]
        
        # generate response using vLLM
        llm_response = generate_response_with_vllm(request.query, chunks)
        llm_response = sanitize_model_output(llm_response)
        
        logger.info(f"Returning precise response with {len(sources)} sources")
        return {"response": llm_response, "sources": sources}
        
    except Exception as e:
        logger.error(f"Error processing precise query: {e}")
        return {"response": f"Error processing query: {str(e)}", "sources": []}


@app.post("/query-lc", tags=["RAG"], response_model=QueryResponse)
def process_query_langchain(request: QueryRequest):
    """
    LangChain-powered query processing using LCEL (LangChain Expression Language).
    Uses the same retriever logic but with LangChain's chain composition.
    """
    try:
        logger.info(f"Processing LangChain query: {request.query}")
        
        # Retrieve documents using LangChain-compatible retriever
        documents = get_relevant_chunks_langchain(request.query, request.user_groups, limit=3, similarity_threshold=0.3)
        
        if not documents:
            return {"response": "No relevant documents found for your query.", "sources": []}
        
        # Format context from documents
        context = "\n\n".join([
            f"Source: {doc.metadata['source']}\nContent: {doc.page_content}" 
            for doc in documents
        ])
        
        # Create the LangChain chain
        chain = (
            RunnableParallel({
                "context": lambda x: context,
                "question": RunnablePassthrough()
            })
            | RAG_PROMPT
            | langchain_llm
            | StrOutputParser()
        )
        
        # Invoke the chain
        request_id = str(uuid.uuid4())[:8]
        logger.info(f"[{request_id}] Invoking LangChain chain...")
        start_time = time.time()
        
        response_text = chain.invoke({"question": request.query})
        
        response_time = time.time() - start_time
        logger.info(f"[{request_id}] LangChain response generated in {response_time:.2f}s")
        
        # Sanitize the response
        response_text = sanitize_model_output(response_text)
        
        # Convert documents to DocumentSource format
        sources = [DocumentSource(text=doc.page_content, source=doc.metadata["source"]) for doc in documents]
        
        logger.info(f"Returning LangChain response with {len(sources)} sources")
        return {"response": response_text, "sources": sources}
        
    except Exception as e:
        logger.error(f"Error processing LangChain query: {e}")
        return {"response": f"Error processing query: {str(e)}", "sources": []}