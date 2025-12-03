import os
import time
import json
from pathlib import Path

import dotenv
import psycopg2
import asyncio
import pandas as pd
from loguru import logger
from app.src.processing import PDFProcessor

dotenv.load_dotenv()

DATA_PATH = Path(os.environ['DATA_PATH'])
LARGE_DATA_PATH = DATA_PATH / "large"
ARXIV_META_PATH = LARGE_DATA_PATH / "arxiv_metadata_with_pdfs.parquet"
PDF_OUT_PATH = LARGE_DATA_PATH / "arxiv_pdfs"


conn = psycopg2.connect(
    host=os.environ['POSTGRES_HOST'],
    dbname=os.environ['POSTGRES_DB'],
    user=os.environ['POSTGRES_USER'],
    password=os.environ['POSTGRES_PASSWORD'],
)

cur = conn.cursor()

def save_chunk(doc_id, chunk_id, text, metadata = {}):
    cur.execute(
        """
        INSERT INTO CTO (doc_id, chunk_id, content, embedding, metadata) VALUES (%s, %s, %s, %s, %s)
        """,
        (doc_id, chunk_id, text, emb.tolist(), json.dumps(metadata))
    )
    conn.commit()


async def main():
    processor = PDFProcessor()
    df = pd.read_parquet(ARXIV_META_PATH)
    df['text'] = df['pdf_path'].apply(lambda path: processor.pdf_to_text(path))
    chunked_texts = []
    for arxiv_id, text in df[['arxiv_id', 'text']].itertuples(index=False):
        chunks = processor.chunk_text(text)
        for idx, chunk in enumerate(chunks):
            chunked_texts.append({
                'arxiv_id': arxiv_id,
                'chunk_id': idx,
                'text': chunk,
                "embedding": processor.embed_text(chunk).tolist(),
            })


if __name__ == "__main__":
    asyncio.run(main())
