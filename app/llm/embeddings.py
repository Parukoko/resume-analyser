import logging
import math
from functools import lru_cache

from openai import OpenAI

from app.config import settings
from app.llm.prompt import CATEGORY_REFERENCE_TEXTS

logger = logging.getLogger(__name__)

_client = OpenAI(
    base_url=settings.llm_base_url,
    api_key=settings.llm_api_key,
    timeout=settings.llm_timeout,
    max_retries=settings.llm_max_retries,
)


def _embed(text: str) -> list[float]:
    response = _client.embeddings.create(model=settings.embedding_model, input=text)
    return response.data[0].embedding


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"Embedding dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


@lru_cache(maxsize=None)
def _category_reference_embedding(category: str) -> tuple[float, ...]:
    return tuple(_embed(CATEGORY_REFERENCE_TEXTS[category]))


def compute_semantic_signals(resume_text: str) -> dict[str, float]:
    resume_embedding = _embed(resume_text)

    signals = {}
    for category in CATEGORY_REFERENCE_TEXTS:
        reference_embedding = list(_category_reference_embedding(category))
        similarity = _cosine_similarity(resume_embedding, reference_embedding)
        signals[category] = round(max(0.0, min(1.0, similarity)) * 100, 1)
    return signals
