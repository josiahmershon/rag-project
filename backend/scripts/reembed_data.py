#!/usr/bin/env python3
"""
Script to re-embed all documents in the vector_index table using the new model.
This handles the migration from 1536 dimensions (padded) to 768 dimensions (native).
"""

import logging
import psycopg2
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

from backend.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reembed_data():
    logger.info("Starting re-embedding process...")
    
    # Load new model
    logger.info(f"Loading model: {settings.embedding_model_name}")
    model = SentenceTransformer(settings.embedding_model_name)
    
    conn = psycopg2.connect(
        host=settings.db_host,
        database=settings.db_name,
        user=settings.db_user,
        password=settings.db_password
    )
    
    try:
        cursor = conn.cursor()
        
        # 1. Create a temporary table with the new schema
        logger.info("Creating temporary table 'vector_index_new'...")
        cursor.execute("DROP TABLE IF EXISTS vector_index_new;")
        cursor.execute("""
            CREATE TABLE vector_index_new (
                doc_id TEXT,
                source_path TEXT,
                department TEXT,
                security_level TEXT,
                allowed_groups TEXT[],
                last_updated_by TEXT,
                last_updated_at TIMESTAMP,
                chunk_text TEXT,
                embedding vector(768)
            );
        """)
        
        # 2. Fetch all existing data
        logger.info("Fetching existing documents...")
        cursor.execute("""
            SELECT doc_id, source_path, department, security_level, 
                   allowed_groups, last_updated_by, last_updated_at, chunk_text 
            FROM vector_index
        """)
        rows = cursor.fetchall()
        total_rows = len(rows)
        logger.info(f"Found {total_rows} documents to re-embed.")
        
        # 3. Re-embed and insert into new table
        for i, row in enumerate(rows):
            if i % 10 == 0:
                logger.info(f"Processing {i}/{total_rows}...")
            
            doc_id, source_path, department, security_level, allowed_groups, last_updated_by, last_updated_at, chunk_text = row
            
            # Generate new embedding
            embedding = model.encode(chunk_text).tolist()
            
            cursor.execute("""
                INSERT INTO vector_index_new (
                    doc_id, source_path, department, security_level, 
                    allowed_groups, last_updated_by, last_updated_at, 
                    chunk_text, embedding
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (doc_id, source_path, department, security_level, allowed_groups, last_updated_by, last_updated_at, chunk_text, embedding))
        
        # 4. Recreate indexes on new table
        logger.info("Recreating indexes...")
        cursor.execute("""
            CREATE INDEX vector_index_new_embedding_idx 
            ON vector_index_new USING ivfflat (embedding vector_cosine_ops) 
            WITH (lists = 100);
        """)
        cursor.execute("CREATE INDEX vector_index_new_allowed_groups_idx ON vector_index_new USING gin (allowed_groups);")
        cursor.execute("CREATE INDEX vector_index_new_department_idx ON vector_index_new (department);")
        cursor.execute("CREATE INDEX vector_index_new_security_level_idx ON vector_index_new (security_level);")
        
        # 5. Swap tables
        logger.info("Swapping tables...")
        cursor.execute("DROP TABLE vector_index;")
        cursor.execute("ALTER TABLE vector_index_new RENAME TO vector_index;")
        
        conn.commit()
        logger.info("Migration completed successfully!")
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Migration failed: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    reembed_data()
