#!/usr/bin/env python3
"""
data ingestion script for the RAG system.
loads sample documents, generates embeddings, and stores them in PostgreSQL.
"""

import psycopg2
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Dict
import json
from backend.settings import settings

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# load embedding model
logger.info("Loading embedding model...")
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
logger.info("Embedding model loaded successfully")

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

# sample documents for testing; updated to match vector_index table structure
SAMPLE_DOCUMENTS = [
    {
        "doc_id": "fin_001",
        "source_path": "financial_report_q3_2024.pdf",
        "department": "Finance",
        "security_level": "Internal",
        "allowed_groups": ["finance", "executives", "management"],
        "last_updated_by": "system",
        "chunk_text": "The company's quarterly revenue increased by 15% compared to the previous quarter. This growth was primarily driven by increased sales in the enterprise segment and successful product launches."
    },
    {
        "doc_id": "prod_001",
        "source_path": "product_update_ai_chatbot.md",
        "department": "Engineering",
        "security_level": "Internal",
        "allowed_groups": ["engineering", "product", "customer_success"],
        "last_updated_by": "system",
        "chunk_text": "Our new AI-powered customer service chatbot has reduced response times by 40% and improved customer satisfaction scores. The system uses natural language processing to understand customer queries and provide accurate responses."
    },
    {
        "doc_id": "sec_001",
        "source_path": "security_audit_report_2024.pdf",
        "department": "Security",
        "security_level": "Confidential",
        "allowed_groups": ["security", "engineering", "executives"],
        "last_updated_by": "system",
        "chunk_text": "The security audit revealed three critical vulnerabilities that need immediate attention. We recommend implementing multi-factor authentication and updating all systems to the latest security patches within 30 days."
    },
    {
        "doc_id": "hr_001",
        "source_path": "hr_employee_survey_2024.pdf",
        "department": "Human Resources",
        "security_level": "Internal",
        "allowed_groups": ["hr", "management", "executives"],
        "last_updated_by": "system",
        "chunk_text": "Employee satisfaction survey results show 85% of staff are happy with their current role. The main areas for improvement include work-life balance and career development opportunities."
    },
    {
        "doc_id": "fac_001",
        "source_path": "facilities_construction_update.pdf",
        "department": "Facilities",
        "security_level": "Internal",
        "allowed_groups": ["facilities", "management", "executives"],
        "last_updated_by": "system",
        "chunk_text": "The new office building construction is 60% complete and on track for completion by Q2 2025. The facility will include modern amenities, flexible workspaces, and sustainable design features."
    },
    {
        "doc_id": "ml_001",
        "source_path": "ml_churn_prediction_analysis.pdf",
        "department": "Data Science",
        "security_level": "Internal",
        "allowed_groups": ["data_science", "engineering", "product"],
        "last_updated_by": "system",
        "chunk_text": "Our machine learning model achieved 94% accuracy in predicting customer churn. The model uses features like usage patterns, support ticket frequency, and payment history to identify at-risk customers."
    }
]

def setup_existing_table():
    """Set up indexes for the existing vector_index table."""
    try:
        conn = psycopg2.connect(
            host=settings.db_host,
            database=settings.db_name,
            user=settings.db_user,
            password=settings.db_password
        )
        cursor = conn.cursor()
        
        # enable pgvector extension
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # create index for vector similarity search on existing table
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS vector_index_embedding_idx 
            ON vector_index USING ivfflat (embedding vector_cosine_ops) 
            WITH (lists = 100);
        """)
        
        # create index for allowed_groups filtering
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS vector_index_allowed_groups_idx 
            ON vector_index USING gin (allowed_groups);
        """)
        
        # create index for department filtering
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS vector_index_department_idx 
            ON vector_index (department);
        """)
        
        # create index for security_level filtering
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS vector_index_security_level_idx 
            ON vector_index (security_level);
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        logger.info("Database indexes created successfully for existing vector_index table")
        
    except Exception as e:
        logger.error(f"Error setting up table indexes: {e}")
        raise

def ingest_documents(documents: List[Dict]):
    """Ingest documents with embeddings into the existing vector_index table."""
    try:
        conn = psycopg2.connect(
            host=settings.db_host,
            database=settings.db_name,
            user=settings.db_user,
            password=settings.db_password
        )
        cursor = conn.cursor()
        
        # clear existing test data (optional - comment out if you want to keep existing data)
        cursor.execute("DELETE FROM vector_index WHERE last_updated_by = 'system';")
        logger.info("Cleared existing test data")
        
        for i, doc in enumerate(documents):
            logger.info(f"Processing document {i+1}/{len(documents)}: {doc['source_path']}")
            
            # generate embedding and pad to 1536 dimensions
            embedding = embedding_model.encode(doc['chunk_text']).tolist()
            embedding = pad_embedding_to_1536(embedding)
            
            # insert into existing vector_index table
            cursor.execute("""
                INSERT INTO vector_index (
                    doc_id, source_path, department, security_level, 
                    allowed_groups, last_updated_by, last_updated_at, 
                    chunk_text, embedding
                )
                VALUES (%s, %s, %s, %s, %s, %s, NOW(), %s, %s)
            """, (
                doc['doc_id'],
                doc['source_path'],
                doc['department'],
                doc['security_level'],
                doc['allowed_groups'],
                doc['last_updated_by'],
                doc['chunk_text'],
                embedding
            ))
        
        conn.commit()
        cursor.close()
        conn.close()
        logger.info(f"Successfully ingested {len(documents)} documents")
        
    except Exception as e:
        logger.error(f"Error ingesting documents: {e}")
        raise

def verify_ingestion():
    """Verify that documents were ingested correctly."""
    try:
        conn = psycopg2.connect(
            host=settings.db_host,
            database=settings.db_name,
            user=settings.db_user,
            password=settings.db_password
        )
        cursor = conn.cursor()
        
        # count total documents
        cursor.execute("SELECT COUNT(*) FROM vector_index;")
        count = cursor.fetchone()[0]
        logger.info(f"Total documents in database: {count}")
        
        # count test documents
        cursor.execute("SELECT COUNT(*) FROM vector_index WHERE last_updated_by = 'system';")
        test_count = cursor.fetchone()[0]
        logger.info(f"Test documents added: {test_count}")
        
        # show sample documents
        cursor.execute("""
            SELECT doc_id, source_path, department, security_level, 
                   allowed_groups
            FROM vector_index 
            WHERE last_updated_by = 'system'
            LIMIT 3;
        """)
        samples = cursor.fetchall()
        
        logger.info("Sample documents:")
        for sample in samples:
            logger.info(f"  - {sample[0]} ({sample[2]}) - {sample[1]}")
            logger.info(f"    Security: {sample[3]}, Groups: {sample[4]}")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error verifying ingestion: {e}")
        raise

if __name__ == "__main__":
    logger.info("Starting data ingestion process...")
    
    try:
        # set up indexes for existing table
        setup_existing_table()
        
        # ingest sample documents
        ingest_documents(SAMPLE_DOCUMENTS)
        
        # verify ingestion
        verify_ingestion()
        
        logger.info("Data ingestion completed successfully!")
        
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        exit(1)
