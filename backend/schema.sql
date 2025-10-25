-- RAG System Database Schema
-- PostgreSQL with pgvector extension

-- Enable pgvector extension for vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Main vector index table for document chunks
CREATE TABLE IF NOT EXISTS vector_index (
    -- Primary key
    id SERIAL PRIMARY KEY,

    -- Document identifiers
    doc_id VARCHAR(255) NOT NULL,
    source_path TEXT NOT NULL,

    -- Metadata
    department VARCHAR(100),
    security_level VARCHAR(50) DEFAULT 'Internal',
    allowed_groups TEXT[] NOT NULL,

    -- Audit fields
    last_updated_by VARCHAR(100),
    last_updated_at TIMESTAMP DEFAULT NOW(),

    -- Content and embedding
    chunk_text TEXT NOT NULL,
    embedding vector(1536) NOT NULL,

    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_vector_index_embedding
ON vector_index USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_vector_index_allowed_groups
ON vector_index USING gin (allowed_groups);

CREATE INDEX IF NOT EXISTS idx_vector_index_department
ON vector_index (department);

CREATE INDEX IF NOT EXISTS idx_vector_index_security_level
ON vector_index (security_level);

CREATE INDEX IF NOT EXISTS idx_vector_index_source_path
ON vector_index (source_path);

CREATE INDEX IF NOT EXISTS idx_vector_index_last_updated
ON vector_index (last_updated_at DESC);

-- Create a view for document summary statistics
CREATE OR REPLACE VIEW document_summary AS
SELECT
    source_path,
    department,
    security_level,
    allowed_groups,
    COUNT(*) as chunk_count,
    MAX(last_updated_at) as last_updated,
    MIN(created_at) as created_at
FROM vector_index
GROUP BY source_path, department, security_level, allowed_groups;

-- Add comments for documentation
COMMENT ON TABLE vector_index IS 'Stores document chunks with their embeddings for RAG retrieval';
COMMENT ON COLUMN vector_index.doc_id IS 'Unique identifier for the document chunk';
COMMENT ON COLUMN vector_index.source_path IS 'Original file path or name of the source document';
COMMENT ON COLUMN vector_index.department IS 'Department that owns the document';
COMMENT ON COLUMN vector_index.security_level IS 'Security classification (Internal, Confidential, Public, etc.)';
COMMENT ON COLUMN vector_index.allowed_groups IS 'Array of user groups with access permission';
COMMENT ON COLUMN vector_index.chunk_text IS 'Text content of the document chunk';
COMMENT ON COLUMN vector_index.embedding IS '1536-dimensional vector embedding of the chunk';
