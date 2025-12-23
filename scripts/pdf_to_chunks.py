import os
import time
import json
from pathlib import Path

import dotenv
import psycopg2
import asyncio
import pandas as pd
from tqdm import tqdm
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


def save_paper_meta(
    arxiv_id: str,
    paper_name: str,
    pub_year: int,
    main_category: str,
    subcategory: str,
):
    cur.execute(
        """
        INSERT INTO documents_meta (
            arxiv_id,
            paper_name,
            pub_year,
            main_category,
            subcategory
        ) VALUES (%s, %s, %s, %s, %s)
        """,
        (arxiv_id, paper_name, pub_year, main_category, subcategory)
    )
    conn.commit()


def save_chunk(
    arxiv_id: str,
    chunk_id: int,
    text: str,
    embeddings: list[float],
    metadata = {}
):
    cur.execute(
        """
        INSERT INTO documents (
            arxiv_id,
            chunk_id,
            content,
            embedding,
            metadata
        ) VALUES (%s, %s, %s, %s, %s)
        """,
        (arxiv_id, chunk_id, text, embeddings, json.dumps(metadata))
    )
    conn.commit()


async def main():
    processor = PDFProcessor()
    df = pd.read_parquet(ARXIV_META_PATH)

    valid_files = [os.path.basename(i) for i in os.listdir(PDF_OUT_PATH)]
    valid_arxiv_ids = [os.path.splitext(i)[0] for i in valid_files]

    df = df[df['arxiv_id'].isin(valid_arxiv_ids)]
    success = []
    for idx, row in tqdm(df.iterrows(), desc="all papers"):
        try:
            pdf_path = PDF_OUT_PATH / os.path.basename(row['pdf_path'])
            text = processor.pdf_to_text(pdf_path)
            logger.info("parsed pdf")

            save_paper_meta(
                arxiv_id=row['arxiv_id'],
                paper_name=row['paper_name'],
                pub_year=row['year'],
                main_category=row['main_category'],
                subcategory=row['subcategory'],
            )
            logger.info("saved meta")

            chunks = processor.chunk_text(text)

            logger.info("chunked doc")

            for idx, chunk in tqdm(enumerate(chunks), desc="chunks"):
                save_chunk(
                    arxiv_id=row['arxiv_id'],
                    chunk_id=idx,
                    text=chunk,
                    embeddings=processor.embed_text(chunk).tolist(),
                )
            logger.info("saved chunks")
            success.append(row['arxiv_id'])
        except Exception as e:
            logger.error(f"Cannot make process document: {e}")
            continue

    try:
        with open("success_docs.json", "w") as f:
            json.dump(success, f)
    except Exception as e:
        logger.error(f"Cannot save success docs: {e}")
        print(success)



if __name__ == "__main__":
    asyncio.run(main())
