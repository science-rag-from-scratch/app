import os
import time
import json
from pathlib import Path

import dotenv
import asyncio
import pandas as pd
from loguru import logger
from app.src.processing import download_arxiv_pdf_from_doi, is_existing_pdf_valid

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


async def download_papers_pdfs(batch: list[str]) -> dict[str, bool]:
    """
    Скачивает PDF файлы для батча статей.
    
    Returns:
        Словарь с результатами: {arxiv_id: success}
    """
    tasks = []
    task_to_id = {}
    
    for item in batch:
        out_path = PDF_OUT_PATH / f"{item}.pdf"
        
        # Проверяем существующий файл на валидность
        if os.path.exists(out_path):
            if is_existing_pdf_valid(out_path):
                logger.debug(f"PDF for {item} already exists and is valid, skipping download.")
                continue
            else:
                logger.warning(f"Existing PDF for {item} is invalid, will re-download.")
                # Удаляем невалидный файл
                try:
                    os.remove(out_path)
                except Exception as e:
                    logger.error(f"Failed to remove invalid PDF {out_path}: {e}")
        
        task = download_arxiv_pdf_from_doi(item, str(out_path))
        tasks.append(task)
        task_to_id[task] = item
    
    # Выполняем все задачи параллельно
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Формируем словарь результатов
    download_results = {}
    for task, result in zip(tasks, results):
        arxiv_id = task_to_id[task]
        if isinstance(result, Exception):
            logger.error(f"Exception downloading {arxiv_id}: {result}")
            download_results[arxiv_id] = False
        else:
            download_results[arxiv_id] = result
            if not result:
                logger.warning(f"Failed to download PDF for {arxiv_id}")
    
    return download_results


async def main():
    arxiv_df = load_arxiv_df()    

    total_downloaded = 0
    total_failed = 0
    
    for batch in papers_batch_iterator(arxiv_df):
        results = await download_papers_pdfs(batch)
        
        # Подсчитываем статистику
        batch_success = sum(1 for success in results.values() if success)
        batch_failed = len(results) - batch_success
        total_downloaded += batch_success
        total_failed += batch_failed
        
        logger.info(f"Batch completed: {batch_success} succeeded, {batch_failed} failed")
        time.sleep(1)  # To avoid overwhelming arXiv servers
    
    # Фильтруем только валидные PDF файлы
    arxiv_df['pdf_path'] = arxiv_df['arxiv_id'].apply(lambda x: str(PDF_OUT_PATH / f"{x}.pdf"))
    
    # Проверяем не только существование, но и валидность файлов
    def is_pdf_valid_and_exists(pdf_path: str) -> bool:
        return os.path.exists(pdf_path) and is_existing_pdf_valid(pdf_path)
    
    arxiv_df = arxiv_df[arxiv_df['pdf_path'].apply(is_pdf_valid_and_exists)]

    arxiv_df.to_parquet(LARGE_DATA_PATH / "arxiv_metadata_with_pdfs.parquet", index=False)

    logger.info("Saved arXiv metadata with PDF paths.")
    logger.info(f"Total downloaded: {total_downloaded}, Total failed: {total_failed}")
    logger.info(f"Final dataset size with valid PDFs: {arxiv_df.shape[0]}")


if __name__ == "__main__":
    asyncio.run(main())