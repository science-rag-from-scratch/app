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

CREATE TABLE chat_history (
    id SERIAL PRIMARY KEY,
    user_id TEXT,
    doc_id TEXT,
    user_message TEXT,
    rephrased_message TEXT,
    assistant_message TEXT,
    timestamp TIMESTAMP,
    sources_ids JSONB,
    chunks TEXT
);

CREATE TABLE IF NOT EXISTS steps (
    "id" UUID PRIMARY KEY,
    "name" TEXT NOT NULL,
    "type" TEXT NOT NULL,
    "threadId" UUID NOT NULL,
    "parentId" UUID,
    "streaming" BOOLEAN NOT NULL,
    "waitForAnswer" BOOLEAN,
    "isError" BOOLEAN,
    "metadata" JSONB,
    "tags" TEXT[],
    "input" TEXT,
    "output" TEXT,
    "createdAt" TEXT,
    "command" TEXT,
    "start" TEXT,
    "end" TEXT,
    "generation" JSONB,
    "showInput" TEXT,
    "language" TEXT,
    "indent" INT,
    "defaultOpen" BOOLEAN
);