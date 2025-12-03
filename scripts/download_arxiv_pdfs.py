import os
import time
import json
from pathlib import Path

import dotenv
import asyncio
import pandas as pd
from loguru import logger
from app.src.processing import download_arxiv_pdf_from_doi

dotenv.load_dotenv()

DATA_PATH = Path(os.environ['DATA_PATH'])
LARGE_DATA_PATH = DATA_PATH / "large"
ARXIV_META_PATH = LARGE_DATA_PATH / "arxiv-metadata-cs-uniform.json"
PDF_OUT_PATH = LARGE_DATA_PATH / "arxiv_pdfs"


for path in [
    DATA_PATH,
    LARGE_DATA_PATH,
    PDF_OUT_PATH,
]:
    os.makedirs(path, exist_ok=True)


def load_arxiv_df(group_size: int = 1000) -> pd.DataFrame:
    with open(ARXIV_META_PATH, "r", encoding="utf-8") as f:
        arxiv_entries = [json.loads(line) for line in f.readlines()]

    arxiv_df = pd.DataFrame(arxiv_entries)
    arxiv_df.drop_duplicates(subset=['arxiv_id'], inplace=True)

    sampled_df = arxiv_df.groupby('subcategory', group_keys=False).apply(
        lambda x: x.sample(min(len(x), group_size), random_state=42)
    )

    print(f"Loaded {sampled_df.shape[0]} papers from arXiv metadata.")
    print(sampled_df.head())

    return sampled_df


def papers_batch_iterator(arxiv_df: pd.DataFrame, batch_size: int = 200):
    num_papers = arxiv_df.shape[0]
    paper_ids = arxiv_df['arxiv_id'].unique().tolist()
    for start_idx in range(0, num_papers, batch_size):
        end_idx = min(start_idx + batch_size, num_papers)
        yield [idx for idx in paper_ids[start_idx:end_idx]]


async def download_papers_pdfs(batch: list[str]) -> None:
    tasks = []
    for item in batch:
        out_path = PDF_OUT_PATH / f"{item}.pdf"
        if os.path.exists(out_path):
            # logger.info(f"PDF for {item} already exists, skipping download.")
            continue
        task = download_arxiv_pdf_from_doi(item, str(out_path))
        tasks.append(task)
    await asyncio.gather(*tasks)


async def main():
    arxiv_df = load_arxiv_df()    

    for batch in papers_batch_iterator(arxiv_df):
        await download_papers_pdfs(batch)
        time.sleep(1)  # To avoid overwhelming arXiv servers
    arxiv_df['pdf_path'] = arxiv_df['arxiv_id'].apply(lambda x: str(PDF_OUT_PATH / f"{x}.pdf"))

    arxiv_df = arxiv_df[arxiv_df['pdf_path'].apply(lambda x: os.path.exists(x))]

    arxiv_df.to_parquet(LARGE_DATA_PATH / "arxiv_metadata_with_pdfs.parquet", index=False)

    logger.info("Saved arXiv metadata with PDF paths.")
    logger.info("Sample size: {}".format(arxiv_df.shape[0]))


if __name__ == "__main__":
    asyncio.run(main())