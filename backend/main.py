# MAIN.PY
import psycopg2.pool
from contextlib import contextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Union
from sentence_transformers import SentenceTransformer
import logging
import time
import uuid
from openai import OpenAI
import tempfile
import os
from pathlib import Path
import re

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

# Globals initialized during FastAPI startup
embedding_model = None
vllm_client = None
langchain_llm = None

def validate_query_request(request: QueryRequest) -> None:
    """Validate query request parameters."""
    if not request.query or len(request.query.strip()) < 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Query must be at least 3 characters"
        )
    if len(request.query) > 1000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Query too long (max 1000 characters)"
        )
    if not request.user_groups:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="At least one user group required"
        )

def validate_upload_file(file: UploadFile) -> None:
    """Validate uploaded file."""
    if file.size and file.size > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, 
            detail="File too large (max 10MB)"
        )

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

# Database connection pool (initialized on startup)
db_pool = None

@app.on_event("startup")
def on_startup() -> None:
    """Initialize heavy resources: embeddings model, DB pool, LLM clients."""
    global embedding_model, db_pool, vllm_client, langchain_llm

    # Initialize embedding model
    logger.info("Loading embedding model...")
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    logger.info("Embedding model loaded successfully")

    # Initialize DB connection pool
    logger.info("Initializing database connection pool...")
    global db_pool
    db_pool = psycopg2.pool.SimpleConnectionPool(
    minconn=1,
    maxconn=20,
    host=settings.db_host,
    database=settings.db_name,
    user=settings.db_user,
    password=settings.db_password
)
    logger.info("Database connection pool initialized successfully")

    # Initialize vLLM client
    logger.info("Initializing vLLM client...")
    vllm_client = OpenAI(
        base_url=settings.vllm_base_url,
        api_key="dummy-key"  # vLLM doesn't require real API key
    )
    logger.info("vLLM client initialized successfully")

    # Initialize LangChain ChatOpenAI client (same vLLM endpoint)
    logger.info("Initializing LangChain ChatOpenAI client...")
    langchain_llm = ChatOpenAI(
        base_url=settings.vllm_base_url,
        api_key="dummy-key",
        model=settings.vllm_model,
        temperature=0.7,
        max_tokens=1000
    )
    logger.info("LangChain ChatOpenAI client initialized successfully")

@app.on_event("shutdown")
def on_shutdown() -> None:
    """Tear down heavy resources cleanly."""
    global db_pool
    try:
        if db_pool is not None:
            db_pool.closeall()
            logger.info("Database connection pool closed")
    except Exception as e:
        logger.warning(f"Error closing DB pool: {e}")

@contextmanager
def get_db_connection():
    """
    a helper function to get a connection from the pool and ensure
    it's returned, even if an error occurs.
    """
    if db_pool is None:
        raise RuntimeError("Database connection pool is not initialized")
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
            # (groups are passed as array param; no string concat needed)
            
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


def process_query_common(request: QueryRequest, limit: int = 3, similarity_threshold: float = 0.3) -> QueryResponse:
    """
    Common query processing logic used by all query endpoints.
    """
    try:
        # Validate request
        validate_query_request(request)
        
        # embed the user's query using embedding model
        logger.info(f"Processing query: {request.query}")
        query_embedding = embedding_model.encode(request.query).tolist()
        query_embedding = pad_embedding_to_1536(query_embedding)
        logger.info(f"Generated embedding with {len(query_embedding)} dimensions")

        # retrieve relevant chunks from the database
        chunks = get_relevant_chunks(query_embedding, request.user_groups, limit=limit, similarity_threshold=similarity_threshold)
        
        # convert chunks to DocumentSource format
        sources = [DocumentSource(text=chunk["text"], source=chunk["source"]) for chunk in chunks]
        
        # generate response using vLLM
        llm_response = generate_response_with_vllm(request.query, chunks)
        llm_response = sanitize_model_output(llm_response)
        
        logger.info(f"Returning response with {len(sources)} sources")
        return {"response": llm_response, "sources": sources}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/query", tags=["RAG"], response_model=QueryResponse)
def process_query(request: QueryRequest):
    """
    takes user query and user groups, retrieves relevant context,
    and returns an answer.
    """
    return process_query_common(request, limit=3, similarity_threshold=0.3)


@app.post("/query-precise", tags=["RAG"], response_model=QueryResponse)
def process_query_precise(request: QueryRequest):
    """
    takes user query and user groups, retrieves only highly relevant context,
    and returns an answer with stricter similarity filtering.
    """
    return process_query_common(request, limit=2, similarity_threshold=0.5)


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
            # Fallback to a generic assistant answer without context
            logger.info("No documents found – falling back to general assistant response")
            try:
                raw = langchain_llm.invoke(
                    "You are a helpful AI assistant.\n\nQuestion: "
                    f"{request.query}\n\nAnswer concisely."
                )
                # Extract text if ChatMessage
                if hasattr(raw, "content"):
                    raw = raw.content
                text = str(raw).strip()
                cleaned = sanitize_model_output(text)
                if not cleaned:
                    cleaned = text  # use raw if sanitizer stripped too much
                return {"response": cleaned, "sources": []}
            except Exception as e:
                logger.error(f"Fallback failed: {e}")
                return {"response": "I'm sorry, I couldn't answer that.", "sources": []}
        
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
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


def parse_document_content(file_path: Path) -> str:
    """Parse document content based on file extension."""
    ext = file_path.suffix.lower()
    
    if ext == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    elif ext == '.md':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    elif ext == '.pdf':
        try:
            import PyPDF2
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except ImportError:
            raise HTTPException(status_code=400, detail="PDF parsing requires PyPDF2. Install with: pip install PyPDF2")
    elif ext == '.docx':
        try:
            from docx import Document as DocxDocument
            doc = DocxDocument(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except ImportError:
            raise HTTPException(status_code=400, detail="DOCX parsing requires python-docx. Install with: pip install python-docx")
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """Simple text chunking."""
    words = text.split()
    chunks = []
    
    if len(words) <= chunk_size:
        return [text]
    
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text = ' '.join(chunk_words)
        
        if len(chunk_text) >= 100:  # Minimum chunk size
            chunks.append(chunk_text)
        
        start += chunk_size - overlap
        if start >= len(words):
            break
    
    return chunks


@app.post("/upload", tags=["Upload"])
async def upload_document(
    file: UploadFile = File(...),
    department: str = Form(...),
    allowed_groups: str = Form(...),
    chunk_size: int = Form(512),
    chunk_overlap: int = Form(50)
):
    """
    Upload and ingest a document with metadata.
    """
    try:
        # Validate file
        validate_upload_file(file)
        
        # Validate file type
        allowed_extensions = ['.txt', '.md', '.pdf', '.docx']
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Parse allowed groups
        groups = [g.strip() for g in allowed_groups.split(',') if g.strip()]
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = Path(tmp_file.name)
        
        try:
            # Parse document content
            logger.info(f"Parsing document: {file.filename}")
            doc_content = parse_document_content(tmp_path)
            
            if not doc_content.strip():
                raise HTTPException(status_code=400, detail="Document appears to be empty")
            
            # Chunk the document
            chunks = chunk_text(doc_content, chunk_size, chunk_overlap)
            logger.info(f"Created {len(chunks)} chunks from {file.filename}")
            
            # Process each chunk
            inserted_chunks = 0
            for i, chunk_content in enumerate(chunks):
                try:
                    # Generate embedding
                    embedding = embedding_model.encode(chunk_content).tolist()
                    embedding = pad_embedding_to_1536(embedding)
                    
                    # Insert into database
                    with get_db_connection() as conn:
                        cursor = conn.cursor()
                        
                        cursor.execute("""
                            INSERT INTO vector_index (
                                doc_id, source_path, department, security_level,
                                allowed_groups, last_updated_by, last_updated_at, 
                                chunk_text, embedding
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, NOW(), %s, %s)
                        """, (
                            str(uuid.uuid4()),
                            file.filename,
                            department,
                            "Internal",  # Default security_level
                            groups,
                            "web_upload",
                            chunk_content,
                            embedding
                        ))
                        
                        conn.commit()
                        cursor.close()
                        inserted_chunks += 1
                        
                except Exception as e:
                    logger.error(f"Failed to insert chunk {i}: {e}")
                    continue
            
            return {
                "message": f"Successfully uploaded and processed {file.filename}",
                "filename": file.filename,
                "chunks_created": len(chunks),
                "chunks_inserted": inserted_chunks,
                "department": department,
                "allowed_groups": groups
            }
            
        finally:
            # Clean up temporary file
            if tmp_path.exists():
                os.unlink(tmp_path)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/database", response_class=HTMLResponse, tags=["Database"])
def view_documents():
    """View all documents in the database."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get document summary
            cursor.execute("""
                SELECT 
                    source_path,
                    department,
                    allowed_groups,
                    COUNT(*) as chunk_count,
                    MAX(last_updated_at) as last_updated
                FROM vector_index 
                GROUP BY source_path, department, allowed_groups
                ORDER BY last_updated DESC
            """)
            
            documents = cursor.fetchall()
            
            # Get total stats
            cursor.execute("SELECT COUNT(*) FROM vector_index")
            total_chunks = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT source_path) FROM vector_index")
            total_documents = cursor.fetchone()[0]
            
            cursor.close()
        
        # Build HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Document Database - RAG System</title>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
                .stats {{ display: flex; gap: 20px; margin-bottom: 20px; }}
                .stat-box {{ background-color: #e9ecef; padding: 15px; border-radius: 8px; text-align: center; min-width: 120px; }}
                .stat-number {{ font-size: 24px; font-weight: bold; color: #007bff; }}
                .stat-label {{ font-size: 14px; color: #6c757d; }}
                .documents-table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                .documents-table th, .documents-table td {{ border: 1px solid #dee2e6; padding: 12px; text-align: left; }}
                .documents-table th {{ background-color: #f8f9fa; font-weight: bold; }}
                .documents-table tr:nth-child(even) {{ background-color: #f8f9fa; }}
                .btn {{ padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; text-decoration: none; display: inline-block; margin: 2px; }}
                .btn-primary {{ background-color: #007bff; color: white; }}
                .btn-danger {{ background-color: #dc3545; color: white; }}
                .btn-secondary {{ background-color: #6c757d; color: white; }}
                .btn:hover {{ opacity: 0.8; }}
                .groups {{ font-size: 12px; color: #6c757d; }}
                .actions {{ white-space: nowrap; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Document Database</h1>
                <p>View and manage all documents in the RAG system</p>
            </div>
            
            <div class="stats">
                <div class="stat-box">
                    <div class="stat-number">{total_documents}</div>
                    <div class="stat-label">Documents</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">{total_chunks}</div>
                    <div class="stat-label">Chunks</div>
                </div>
            </div>
            
            <div style="margin-bottom: 20px;">
                <a href="/upload" class="btn btn-primary">📄 Upload New Document</a>
                <a href="/database" class="btn btn-secondary">🔄 Refresh</a>
            </div>
            
            <table class="documents-table">
                <thead>
                    <tr>
                        <th>Document</th>
                        <th>Department</th>
                        <th>Allowed Groups</th>
                        <th>Chunks</th>
                        <th>Last Updated</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for doc in documents:
            source_path, department, allowed_groups, chunk_count, last_updated = doc
            
            # Safely format the data
            safe_source_path = str(source_path).replace("'", "\\'").replace('"', '\\"')
            safe_department = str(department) if department else "Unknown"
            
            # Handle allowed_groups safely
            if allowed_groups and isinstance(allowed_groups, list):
                groups_str = ', '.join(str(g) for g in allowed_groups)
            else:
                groups_str = "None"
            
            # Format date safely
            if last_updated:
                try:
                    date_str = last_updated.strftime('%Y-%m-%d %H:%M')
                except:
                    date_str = str(last_updated)
            else:
                date_str = "Unknown"
            
            html += f"""
                    <tr>
                        <td><strong>{safe_source_path}</strong></td>
                        <td>{safe_department}</td>
                        <td class="groups">{groups_str}</td>
                        <td>{chunk_count}</td>
                        <td>{date_str}</td>
                        <td class="actions">
                            <button class="btn btn-primary" onclick="previewDocument('{safe_source_path}')">👁️ Preview</button>
                            <button class="btn btn-danger" onclick="deleteDocument('{safe_source_path}')">🗑️ Delete</button>
                        </td>
                    </tr>
            """
        
        html += """
                </tbody>
            </table>
            
            <script>
                function previewDocument(filename) {
                    fetch(`/docs/preview/${encodeURIComponent(filename)}`)
                        .then(response => response.json())
                        .then(data => {
                            if (data.error) {
                                alert('Error: ' + data.error);
                            } else {
                                let content = 'Document: ' + data.filename + '\\n\\n';
                                content += 'Department: ' + data.department + '\\n';
                                content += 'Allowed Groups: ' + data.allowed_groups.join(', ') + '\\n';
                                content += 'Chunks: ' + data.chunks.length + '\\n\\n';
                                content += 'Content Preview:\\n';
                                content += data.chunks.slice(0, 2).map((chunk, i) => 
                                    `Chunk ${i+1}: ${chunk.substring(0, 200)}...`
                                ).join('\\n\\n');
                                alert(content);
                            }
                        })
                        .catch(error => {
                            alert('Error loading preview: ' + error.message);
                        });
                }
                
                function deleteDocument(filename) {
                    if (confirm('Are you sure you want to delete all chunks for "' + filename + '"? This cannot be undone.')) {
                        fetch(`/docs/delete/${encodeURIComponent(filename)}`, { method: 'DELETE' })
                            .then(response => response.json())
                            .then(data => {
                                if (data.success) {
                                    alert('Document deleted successfully!');
                                    location.reload();
                                } else {
                                    alert('Error: ' + data.error);
                                }
                            })
                            .catch(error => {
                                alert('Error deleting document: ' + error.message);
                            });
                    }
                }
            </script>
        </body>
        </html>
        """
        
        return html
        
    except Exception as e:
        logger.error(f"Error viewing documents: {e}")
        return f"<h1>Error</h1><p>Failed to load documents: {str(e)}</p>"


@app.get("/docs/preview/{filename}", tags=["Database"])
def preview_document(filename: str):
    """Preview a document's chunks."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT chunk_text, department, allowed_groups
                FROM vector_index 
                WHERE source_path = %s
                ORDER BY doc_id
            """, (filename,))
            
            results = cursor.fetchall()
            
            if not results:
                return {"error": "Document not found"}
            
            chunks = [row[0] for row in results]
            department = results[0][1]
            allowed_groups = results[0][2]
            
            cursor.close()
            
            return {
                "filename": filename,
                "department": department,
                "allowed_groups": allowed_groups,
                "chunks": chunks
            }
            
    except Exception as e:
        logger.error(f"Error previewing document: {e}")
        return {"error": str(e)}


@app.delete("/docs/delete/{filename}", tags=["Database"])
def delete_document(filename: str):
    """Delete all chunks for a document."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM vector_index WHERE source_path = %s", (filename,))
            deleted_count = cursor.rowcount
            
            conn.commit()
            cursor.close()
            
            return {
                "success": True,
                "message": f"Deleted {deleted_count} chunks for {filename}",
                "deleted_count": deleted_count
            }
            
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        return {"success": False, "error": str(e)}


@app.get("/upload", response_class=HTMLResponse, tags=["Upload"])
def upload_form():
    """Serve the upload form."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Document Upload - RAG System</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .form-group { margin-bottom: 20px; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            input, select, textarea { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; }
            button { background-color: #007bff; color: white; padding: 12px 24px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
            button:hover { background-color: #0056b3; }
            .result { margin-top: 20px; padding: 15px; border-radius: 4px; }
            .success { background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724; }
            .error { background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
        </style>
    </head>
    <body>
        <h1>Document Upload</h1>
        <p>Upload documents to add them to the RAG system. Select the appropriate metadata and access controls.</p>
        
        <form id="uploadForm" enctype="multipart/form-data">
            <div class="form-group">
                <label for="file">Choose Document:</label>
                <input type="file" id="file" name="file" accept=".txt,.md,.pdf,.docx" required>
                <small>Supported formats: TXT, MD, PDF, DOCX</small>
            </div>
            
            <div class="form-group">
                <label for="department">Department:</label>
                <select id="department" name="department" required>
                    <option value="">Select Department</option>
                    <option value="Finance">Finance</option>
                    <option value="Engineering">Engineering</option>
                    <option value="Human Resources">Human Resources</option>
                    <option value="Security">Security</option>
                    <option value="Product">Product</option>
                    <option value="Marketing">Marketing</option>
                    <option value="Sales">Sales</option>
                    <option value="Operations">Operations</option>
                    <option value="General">General</option>
                </select>
            </div>
            
            
            <div class="form-group">
                <label for="allowed_groups">Allowed Groups (comma-separated):</label>
                <input type="text" id="allowed_groups" name="allowed_groups" 
                       placeholder="e.g., finance,executives,management" required>
                <small>Examples: finance,executives | engineering,product | hr,employees,management</small>
            </div>
            
            <div class="form-group">
                <label for="chunk_size">Chunk Size (words):</label>
                <input type="number" id="chunk_size" name="chunk_size" value="512" min="100" max="2048">
                <small>Number of words per chunk (100-2048)</small>
            </div>
            
            <div class="form-group">
                <label for="chunk_overlap">Chunk Overlap (words):</label>
                <input type="number" id="chunk_overlap" name="chunk_overlap" value="50" min="0" max="200">
                <small>Overlap between chunks for context preservation</small>
            </div>
            
            <button type="submit">Upload & Ingest Document</button>
        </form>
        
        <div id="result"></div>
        
        <script>
            document.getElementById('uploadForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const formData = new FormData();
                formData.append('file', document.getElementById('file').files[0]);
                formData.append('department', document.getElementById('department').value);
                formData.append('allowed_groups', document.getElementById('allowed_groups').value);
                formData.append('chunk_size', document.getElementById('chunk_size').value);
                formData.append('chunk_overlap', document.getElementById('chunk_overlap').value);
                
                const resultDiv = document.getElementById('result');
                resultDiv.innerHTML = '<p>Uploading and processing document...</p>';
                
                try {
                    const response = await fetch('/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        resultDiv.innerHTML = `
                            <div class="result success">
                                <h3>Upload Successful!</h3>
                                <p><strong>File:</strong> ${data.filename}</p>
                                <p><strong>Chunks Created:</strong> ${data.chunks_created}</p>
                                <p><strong>Chunks Inserted:</strong> ${data.chunks_inserted}</p>
                                <p><strong>Department:</strong> ${data.department}</p>
                                <p><strong>Allowed Groups:</strong> ${data.allowed_groups.join(', ')}</p>
                            </div>
                        `;
                    } else {
                        resultDiv.innerHTML = `
                            <div class="result error">
                                <h3>Upload Failed</h3>
                                <p>${data.detail}</p>
                            </div>
                        `;
                    }
                } catch (error) {
                    resultDiv.innerHTML = `
                        <div class="result error">
                            <h3>Upload Error</h3>
                            <p>${error.message}</p>
                        </div>
                    `;
                }
            });
        </script>
    </body>
    </html>
    """