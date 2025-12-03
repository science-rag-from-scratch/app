import httpx
from loguru import logger


CONTENT_LOWER_BOUND = 1000  # Minimum size in bytes to consider a valid PDF


async def download_arxiv_pdf_from_doi(arxiv_id: str, out_path: str) -> bool:
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    async with httpx.AsyncClient() as client:
        r = await client.get(pdf_url, follow_redirects=True)

        if len(r.content) < CONTENT_LOWER_BOUND:
            logger.error(f"Failed to download PDF from {pdf_url}, text: {r.text}")
            return False

        with open(out_path, "wb") as f:
            f.write(r.content)
        return True
