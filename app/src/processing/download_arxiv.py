import httpx
from loguru import logger
from pathlib import Path

# Minimum size in bytes to consider a valid PDF in bytes (2KB)
CONTENT_LOWER_BOUND = 2 * 1024

async def is_arxiv_pdf_url_valid(pdf_url: str) -> bool:
    """
    Проверяет, доступен ли PDF с данного URL, не является ли он заглушкой arXiv (404 и пр.).

    Возвращает True если это валидный PDF, False иначе.
    """
    try:
        async with httpx.AsyncClient() as client:
            r = await client.head(pdf_url, follow_redirects=True, timeout=10)
            # Проверяем Content-Type и размер
            if r.status_code != 200:
                return False
            content_type = r.headers.get("Content-Type", "")
            if "pdf" not in content_type.lower():
                return False
            if "Content-Length" in r.headers:
                try:
                    size = int(r.headers["Content-Length"])
                    if size < CONTENT_LOWER_BOUND:
                        return False
                except Exception:
                    # Malformed header, fallback to GET strategy
                    return False
            # Если Content-Length неизвестен — проверим body при GET
            return True
    except Exception as e:
        logger.error(f"HEAD request failed for {pdf_url}: {e}")
        return False

def is_existing_pdf_valid(pdf_path: str | Path) -> bool:
    """
    Проверяет, является ли существующий файл валидным PDF.
    Возвращает True если файл существует и является валидным PDF, False иначе.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        return False
    
    try:
        # Проверяем размер файла
        file_size = pdf_path.stat().st_size
        if file_size < CONTENT_LOWER_BOUND:
            logger.warning(f"PDF file too small: {pdf_path} ({file_size} bytes)")
            return False
        
        # Проверяем первые байты файла
        with open(pdf_path, "rb") as f:
            first_bytes = f.read(4)
            if not first_bytes.startswith(b"%PDF"):
                logger.warning(f"File does not start with %PDF: {pdf_path}")
                return False
        
        return True
    except Exception as e:
        logger.error(f"Error checking existing PDF {pdf_path}: {e}")
        return False

async def download_arxiv_pdf_fallback(arxiv_id: str, out_path: str) -> bool:
    fallback_pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    """
    Скачивает валидный arXiv PDF по "фолбэк" URL, аналогично robust_download_arxiv_pdf.
    Возвращает True если успех, иначе False.
    """
    # Фолбэк URL: он может не содержать расширение ".pdf"
    # Проверим валидность URL предварительно
    is_pdf_valid = await is_arxiv_pdf_url_valid(fallback_pdf_url)
    if not is_pdf_valid:
        logger.error(f"arXiv PDF fallback URL invalid or unavailable: {fallback_pdf_url}")
        return False

    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(fallback_pdf_url, follow_redirects=True, timeout=50)
            content_type = r.headers.get("Content-Type", "")
            if r.status_code != 200 or "pdf" not in content_type.lower():
                logger.error(f"Bad response/code for {fallback_pdf_url}: Status {r.status_code}, Content-Type {content_type}")
                return False

            if len(r.content) < CONTENT_LOWER_BOUND:
                logger.error(f"Failed to download PDF from {fallback_pdf_url}, file too small ({len(r.content)} bytes), text: {r.text[:512]}")
                return False

            # Быстрая проверка — первый байт должен быть %PDF
            if not r.content.startswith(b"%PDF"):
                logger.error(f"Downloaded file does not start with %PDF, probably not a valid PDF: {fallback_pdf_url}")
                return False

            # Сохраняем во временный файл, затем атомарно переименовываем
            out_path_obj = Path(out_path)
            temp_path = out_path_obj.with_suffix(out_path_obj.suffix + '.tmp')
            try:
                with open(temp_path, "wb") as f:
                    f.write(r.content)
                temp_path.replace(out_path_obj)
            except Exception as e:
                logger.error(f"Error writing PDF to {out_path}: {e}")
                if temp_path.exists():
                    temp_path.unlink(missing_ok=True)
                return False

        # Валидируем скачанный файл
        if not is_existing_pdf_valid(out_path_obj):
            logger.error(f"Downloaded PDF is not valid: {out_path_obj}")
            return False

        logger.info(f"Successfully downloaded fallback arXiv PDF: {arxiv_id} -> {out_path}")
        return True
    except Exception as e:
        logger.error(f"Exception during fallback download of {arxiv_id}: {e}")
        return False


async def robust_download_arxiv_pdf(arxiv_id: str, out_path: str) -> bool:
    """
    Корректно скачивает валидный arXiv PDF, возвращает True если успех, иначе False.
    
    Args:
        arxiv_id: arXiv ID статьи (например, "1234.5678" или "1234.5678v1")
        out_path: Путь для сохранения PDF файла
    
    Returns:
        True если PDF успешно скачан и валиден, False иначе
    """
    # Нормализуем arxiv_id (убираем версию если есть, т.к. URL работает без версии)
    arxiv_id_clean = arxiv_id.split('v')[0] if 'v' in arxiv_id else arxiv_id
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id_clean}.pdf"
    
    # Сначала проверим HEAD (чтобы не скачивать "404 PDF")
    is_pdf_valid = await is_arxiv_pdf_url_valid(pdf_url)
    if not is_pdf_valid:
        logger.error(f"arXiv PDF URL invalid or unavailable: {pdf_url}")
        return await download_arxiv_pdf_fallback(arxiv_id, out_path)
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(pdf_url, follow_redirects=True, timeout=50)
            content_type = r.headers.get("Content-Type", "")
            if r.status_code != 200 or "pdf" not in content_type.lower():
                logger.error(f"Bad response/code for {pdf_url}: Status {r.status_code}, Content-Type {content_type}")
                return await download_arxiv_pdf_fallback(arxiv_id, out_path)

            if len(r.content) < CONTENT_LOWER_BOUND:
                logger.error(f"Failed to download PDF from {pdf_url}, file too small ({len(r.content)} bytes), text: {r.text[:512]}")
                return await download_arxiv_pdf_fallback(arxiv_id, out_path)

            # Быстрая доп. проверка — первый байт должен быть %PDF
            if not r.content.startswith(b"%PDF"):
                logger.error(f"Downloaded file does not start with %PDF, probably not a valid PDF: {pdf_url}")
                return await download_arxiv_pdf_fallback(arxiv_id, out_path)

            # Записываем файл во временное место, затем переименовываем атомарно
            out_path_obj = Path(out_path)
            temp_path = out_path_obj.with_suffix(out_path_obj.suffix + '.tmp')
            
            try:
                with open(temp_path, "wb") as f:
                    f.write(r.content)
                
                # Проверяем скачанный файл перед финальным сохранением
                if not is_existing_pdf_valid(temp_path):
                    temp_path.unlink(missing_ok=True)
                    logger.error(f"Downloaded file is not a valid PDF: {pdf_url}")
                    return await download_arxiv_pdf_fallback(arxiv_id, out_path)
                
                # Атомарно заменяем существующий файл (если есть)
                temp_path.replace(out_path_obj)
                logger.info(f"Successfully downloaded PDF: {arxiv_id} -> {out_path}")
                return True
            except Exception as e:
                temp_path.unlink(missing_ok=True)
                raise e
                
    except Exception as e:
        logger.error(f"Exception downloading {pdf_url}: {e}")
        return await download_arxiv_pdf_fallback(arxiv_id, out_path)

# Monkey-patch the existing function name to use robust download
download_arxiv_pdf_from_doi = robust_download_arxiv_pdf
