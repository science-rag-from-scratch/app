# Science RAG: Retrieval-Augmented Generation for Academic Research

A Retrieval-Augmented Generation (RAG) system that ingests, processes, and indexes academic papers from arXiv to enable intelligent semantic search and question-answering over scientific literature.

## 🎯 Project Overview

Science RAG combines:

- **Data Collection**: Automated downloading of research papers from arXiv
- **Document Processing**: PDF extraction, text chunking, and preprocessing
- **Vector Embeddings**: Semantic embeddings for similarity-based retrieval
- **Database Storage**: PostgreSQL with vector similarity search
- **LLM Integration**: LLM-powered question answering with retrieved context

This system enables researchers to query scientific papers using natural language and receive context-aware answers grounded in the source material.

## ✨ Key Features

- **Scalable Data Pipeline**: Batch processing with async PDF downloads
- **Intelligent Chunking**: Document-aware text segmentation for better retrieval
- **Vector Database**: PostgreSQL-backed vector storage for semantic search
- **Distributed Processing**: Docker-based containerization for easy deployment
- **Metadata Tracking**: Comprehensive paper metadata and source attribution

## 📋 Prerequisites

- Python 3.10+
- [UV](https://docs.astral.sh/uv/) package manager
- PostgreSQL with pgvector extension
- Docker & Docker Compose (optional)
- 16GB+ RAM (for model inference)

## 🚀 Quick Start

### 1. Environment Setup

```bash
uv sync
```

```bash
./setup_env.sh
```

### 2. Data Pipeline

Run the data pipeline in stages:

```bash
# Stage 1: Download arXiv metadata and PDFs
python scripts/download_arxiv_pdfs.py

# Stage 2: Process PDFs and create chunks with embeddings
python scripts/pdf_to_chunks.py


## 📂 Project Structure

```text
app/
├── src/
│   ├── db/
│   │   ├── scheme.py          # Database models (Pydantic)
│   │   └── __init__.py
│   ├── processing/
│   │   ├── download_arxiv.py  # arXiv PDF downloading
│   │   ├── pdf.py             # PDF text extraction & chunking
│   │   └── __init__.py
│   └── __init__.py
├── main.py                     # Application entry point
├── scripts/
│   ├── download_arxiv_pdfs.py # Batch PDF downloader
│   └── pdf_to_chunks.py       # PDF processing pipeline
├── data/
│   └── large/
│       ├── arxiv-metadata-cs-uniform.json
│       ├── arxiv_pdfs/        # Downloaded PDF files
│       └── arxiv_metadata_with_pdfs.parquet
├── pyproject.toml
├── docker-compose.yaml
├── Dockerfile
└── README.md
```

## 🔄 Data Pipeline

### Stage 1: Download Papers

- **Input**: arXiv metadata JSON (papers in Computer Science categories)
- **Process**: Stratified sampling by subcategory, async PDF downloads
- **Output**: Raw PDF files + metadata with PDF paths

```bash
python scripts/download_arxiv_pdfs.py
```

### Stage 2: Process PDFs to Chunks

- **Input**: PDFs from Stage 1
- **Process**: Extract text, intelligently chunk documents, generate embeddings
- **Output**: Document chunks stored in PostgreSQL with embeddings

```bash
python scripts/pdf_to_chunks.py
```

### Stage 3: Enable Query Interface

- **Input**: Database with indexed chunks
- **Process**: LLM retrieval & context-aware responses
- **Output**: Web interface for semantic search & Q&A

```bash
chainlit run app/main.py
```

## 🛠️ Core Components

### PDFProcessor

Handles PDF processing:

- Text extraction from PDF documents
- Intelligent text chunking (maintains semantic boundaries)
- Metadata extraction and preservation

### ArXiv Downloader

Batch downloads papers from arXiv:

- Concurrent async downloads for efficiency
- Validation and error handling
- Resumable downloads with duplicate detection

### Vector Database

PostgreSQL with pgvector:

- Stores document chunks with embeddings
- Enables semantic similarity search
- Maintains source attribution for results

## 🐳 Docker Deployment

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop services
docker-compose down
```

## 📈 Performance Considerations

- **Batch Processing**: PDFs downloaded in batches of 200 with 1s delays to respect arXiv rate limits
- **Async Operations**: Concurrent downloads improve throughput
- **Streaming Chunks**: Large PDFs processed sequentially to manage memory
- **Vector Indexing**: PostgreSQL pgvector enables efficient semantic search

## 📄 License

See LICENSE file for details.

## ❤️ Acknowledgements

Built with ❤️ for open science and reproducible research by [AI Talent Hub](https://ai.itmo.ru/) students.
