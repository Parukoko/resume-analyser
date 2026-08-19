import json
from unittest.mock import patch

import pytest

from app.llm import analyzer

FAKE_RESULT = {
    "job_title": "AI & Data Solution Intern",
    "overall_score": 999,
    "max_score": 100,
    "breakdown": {
        "education": {"score": 18, "max": 20, "reasoning": "x"},
        "experience": {"score": 15, "max": 25, "reasoning": "x"},
        "skills": {"score": 22, "max": 25, "reasoning": "x"},
        "tools_and_technologies": {"score": 13, "max": 15, "reasoning": "x"},
        "knowledge_and_domain": {"score": 10, "max": 15, "reasoning": "x"},
    },
    "strengths": ["x"],
    "gaps": ["x"],
    "recommendation": "x",
    "raw_resume_summary": "x",
}


def test_extract_json_handles_plain_json():
    data = analyzer._extract_json(json.dumps(FAKE_RESULT))
    assert data["job_title"] == FAKE_RESULT["job_title"]


def test_extract_json_strips_think_blocks_and_code_fences():
    raw = "<think>reasoning...</think>\n```json\n" + json.dumps(FAKE_RESULT) + "\n```"
    data = analyzer._extract_json(raw)
    assert data["overall_score"] == 999


def test_extract_json_raises_on_garbage():
    with pytest.raises(analyzer.LLMAnalysisError):
        analyzer._extract_json("not json at all")


def test_blend_and_finalize_recomputes_overall_score_ignoring_llm_total():
    from app.schemas import AnalysisResult

    result = AnalysisResult.model_validate(FAKE_RESULT)
    signals = {
        "education": 80.0,
        "experience": 60.0,
        "skills": 88.0,
        "tools_and_technologies": 70.0,
        "knowledge_and_domain": 55.0,
    }
    finalized = analyzer._blend_and_finalize(result, signals)

    category_total = sum(
        getattr(finalized.breakdown, key).score
        for key in ["education", "experience", "skills", "tools_and_technologies", "knowledge_and_domain"]
    )
    assert finalized.overall_score == category_total
    assert finalized.overall_score != 999
    assert finalized.breakdown.education.llm_score == 18
    assert finalized.breakdown.education.semantic_similarity == 80.0


def test_blend_and_finalize_without_signals_leaves_score_unblended():
    from app.schemas import AnalysisResult

    result = AnalysisResult.model_validate(FAKE_RESULT)
    finalized = analyzer._blend_and_finalize(result, {})

    assert finalized.breakdown.education.score == finalized.breakdown.education.llm_score == 18
    assert finalized.breakdown.education.semantic_similarity is None


def test_analyze_resume_end_to_end_with_mocked_llm():
    with patch("app.llm.analyzer._client") as mock_client, patch(
        "app.llm.analyzer.compute_semantic_signals", return_value={}
    ):
        mock_resp = mock_client.chat.completions.create.return_value
        mock_resp.choices = [type("C", (), {"message": type("M", (), {"content": json.dumps(FAKE_RESULT)})()})]
        result = analyzer.analyze_resume("some resume text")

    assert result.overall_score == 78
