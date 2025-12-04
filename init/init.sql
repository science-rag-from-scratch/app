CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    arxiv_id TEXT,
    chunk_id INT,
    content TEXT,
    embedding vector(1024),
    metadata JSONB
);

CREATE TABLE documents_meta (
    id SERIAL PRIMARY KEY,
    arxiv_id TEXT,
    paper_name TEXT,
    pub_year INTEGER,
    main_category TEXT,
    subcategory TEXT
);