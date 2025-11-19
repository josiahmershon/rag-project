import psycopg2
import numpy as np
from sentence_transformers import SentenceTransformer
from backend.settings import settings

def verify():
    print("Verifying migration...")
    
    # 1. Check Table Schema
    conn = psycopg2.connect(
        host=settings.db_host,
        database=settings.db_name,
        user=settings.db_user,
        password=settings.db_password
    )
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT atttypmod 
        FROM pg_attribute 
        WHERE attrelid = 'vector_index'::regclass 
        AND attname = 'embedding';
    """)
    dim = cursor.fetchone()[0]
    print(f"Database Vector Dimension: {dim}")
    
    if dim != 768:
        print("FAIL: Dimension mismatch!")
        return

    # 2. Check Data Count
    cursor.execute("SELECT count(*) FROM vector_index")
    count = cursor.fetchone()[0]
    print(f"Total Documents: {count}")
    
    if count == 0:
        print("FAIL: No documents found!")
        return

    # 3. Test Vector Search
    model = SentenceTransformer(settings.embedding_model_name)
    query = "revenue growth"
    embedding = model.encode(query).tolist()
    
    # Simple cosine similarity search
    cursor.execute("""
        SELECT chunk_text, 1 - (embedding <=> %s::vector) as similarity
        FROM vector_index
        ORDER BY embedding <=> %s::vector
        LIMIT 1
    """, (str(embedding), str(embedding)))
    
    row = cursor.fetchone()
    if row:
        print(f"Top Match: {row[0][:50]}... (Similarity: {row[1]:.4f})")
        print("SUCCESS: Retrieval works!")
    else:
        print("FAIL: Retrieval returned no results.")

    cursor.close()
    conn.close()

if __name__ == "__main__":
    verify()
