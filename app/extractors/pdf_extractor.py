"""PDF text extraction.

Tries the fast, accurate path first (pdfplumber, works for text-based PDFs).
Falls back to a vision-capable LLM (same provider/model as the main scoring
call) for scanned/image-only PDFs, transcribing each rasterized page.
"""

import base64
import io
import logging
import time
from dataclasses import dataclass

import pdfplumber
from openai import APIError, OpenAI

from app.config import settings

logger = logging.getLogger(__name__)

PAGE_TRANSCRIPTION_RETRY_DELAY_SECONDS = 3

_client = OpenAI(
    base_url=settings.llm_base_url,
    api_key=settings.llm_api_key,
    timeout=settings.llm_timeout,
    max_retries=settings.llm_max_retries,
)

TRANSCRIPTION_PROMPT = (
    "Transcribe all text visible in this resume page image, exactly as written, "
    "preserving reading order (top to bottom, left to right for multi-column "
    "layouts). Output only the transcribed text - no summary, no commentary, no "
    "markdown formatting. If the page has no readable text, output nothing."
)


@dataclass
class ExtractionResult:
    text: str
    method: str  # "text" or "vision"


def _extract_with_pdfplumber(pdf_path: str) -> tuple[str, int]:
    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        page_count = len(pdf.pages)
        for page in pdf.pages:
            pages_text.append(page.extract_text() or "")
    return "\n".join(pages_text), page_count


def _page_to_data_url(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{encoded}"


def _transcribe_page(image) -> str:
    try:
        response = _client.chat.completions.create(
            model=settings.llm_model,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": TRANSCRIPTION_PROMPT},
                        {"type": "image_url", "image_url": {"url": _page_to_data_url(image)}},
                    ],
                }
            ],
        )
    except APIError as e:
        raise RuntimeError(f"Vision transcription failed ({settings.llm_model}): {e}") from e
    return response.choices[0].message.content or ""


def _transcribe_page_with_retry(image, page_num: int) -> str:
    try:
        return _transcribe_page(image)
    except RuntimeError:
        logger.warning(
            "Page %d transcription failed, retrying once in %ds",
            page_num,
            PAGE_TRANSCRIPTION_RETRY_DELAY_SECONDS,
            exc_info=True,
        )
        time.sleep(PAGE_TRANSCRIPTION_RETRY_DELAY_SECONDS)
        return _transcribe_page(image)


def _extract_with_vision_llm(pdf_path: str) -> str:
    from pdf2image import convert_from_path

    images = convert_from_path(pdf_path, dpi=300)
    pages_text = []
    for page_num, image in enumerate(images, start=1):
        try:
            pages_text.append(_transcribe_page_with_retry(image, page_num))
        except RuntimeError:
            logger.warning("Page %d transcription failed twice, giving up on this page", page_num, exc_info=True)
            pages_text.append(f"[page {page_num}: transcription failed]")
    return "\n".join(pages_text)


def _looks_like_scanned(text: str, page_count: int) -> bool:
    if page_count == 0:
        return False
    avg_chars_per_page = len(text.strip()) / page_count
    return avg_chars_per_page < settings.ocr_fallback_char_threshold


def extract_text(pdf_path: str) -> ExtractionResult:
    text, page_count = _extract_with_pdfplumber(pdf_path)

    if _looks_like_scanned(text, page_count):
        logger.info("PDF looks image-based/scanned, falling back to vision-LLM transcription")
        vision_text = _extract_with_vision_llm(pdf_path)
        if len(vision_text.strip()) > len(text.strip()):
            return ExtractionResult(text=vision_text, method="vision")

    return ExtractionResult(text=text, method="text")
