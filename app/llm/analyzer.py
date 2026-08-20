import json
import logging
import re

from openai import APIError, OpenAI
from pydantic import ValidationError

from app.config import settings
from app.llm.embeddings import compute_semantic_signals
from app.llm.prompt import CATEGORY_MAX_POINTS, SYSTEM_PROMPT, build_user_prompt
from app.schemas import CATEGORY_KEYS, AnalysisResult

logger = logging.getLogger(__name__)

_client = OpenAI(
    base_url=settings.llm_base_url,
    api_key=settings.llm_api_key,
    timeout=settings.llm_timeout,
    max_retries=settings.llm_max_retries,
)

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


class LLMAnalysisError(Exception):
    pass


def _extract_json(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    match = _JSON_BLOCK_RE.search(raw)
    if not match:
        raise LLMAnalysisError(f"No JSON object found in LLM response: {raw[:500]!r}")

    return json.loads(match.group(0))


def _safe_compute_semantic_signals(resume_text: str) -> dict[str, float]:
    try:
        return compute_semantic_signals(resume_text)
    except Exception:
        logger.warning(
            "Embedding backend unavailable, skipping semantic-similarity signals "
            "(check LLM_API_KEY / LLM_BASE_URL, and that embedding model %r is "
            "available there)",
            settings.embedding_model,
            exc_info=True,
        )
        return {}


def _blend_and_finalize(result: AnalysisResult, semantic_signals: dict[str, float]) -> AnalysisResult:
    total = 0
    for category in CATEGORY_KEYS:
        cat = getattr(result.breakdown, category)
        llm_score = max(0, min(cat.max, cat.score))
        cat.llm_score = llm_score

        similarity = semantic_signals.get(category)
        if similarity is not None:
            cat.semantic_similarity = similarity
            semantic_scaled = (similarity / 100) * cat.max
            blended = round(
                (1 - settings.semantic_weight) * llm_score + settings.semantic_weight * semantic_scaled
            )
            cat.score = max(0, min(cat.max, blended))
        else:
            cat.score = llm_score

        total += cat.score

    result.overall_score = total
    result.max_score = sum(CATEGORY_MAX_POINTS.values())
    return result


def analyze_resume(resume_text: str) -> AnalysisResult:
    semantic_signals = _safe_compute_semantic_signals(resume_text)

    try:
        response = _client.chat.completions.create(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(resume_text, semantic_signals)},
            ],
        )
    except APIError as e:
        raise LLMAnalysisError(f"LLM call failed ({settings.llm_model} at {settings.llm_base_url}): {e}") from e

    content = response.choices[0].message.content
    if not content:
        raise LLMAnalysisError("LLM returned an empty response")

    try:
        data = _extract_json(content)
    except json.JSONDecodeError as e:
        raise LLMAnalysisError(f"Failed to parse LLM response as JSON: {e}") from e

    try:
        result = AnalysisResult.model_validate(data)
    except ValidationError as e:
        raise LLMAnalysisError(f"LLM response did not match expected schema: {e}") from e

    return _blend_and_finalize(result, semantic_signals)
