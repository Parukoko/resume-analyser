from typing import List, Optional

from pydantic import BaseModel, Field

CATEGORY_KEYS = (
    "education",
    "experience",
    "skills",
    "tools_and_technologies",
    "knowledge_and_domain",
)


class CategoryScore(BaseModel):
    score: int = Field(
        ...,
        ge=0,
        description="Final score after blending llm_score with the semantic_similarity signal.",
    )
    max: int = Field(..., gt=0)
    reasoning: str
    llm_score: Optional[int] = Field(
        default=None,
        description="Raw score as returned by the LLM, before blending with semantic_similarity.",
    )
    semantic_similarity: Optional[float] = Field(
        default=None,
        description=(
            "Embedding cosine similarity (0-100) between the resume and this category's "
            "reference description, computed independently of the LLM. None if the "
            "embedding backend was unavailable (in which case score == llm_score, unblended)."
        ),
    )


class ScoreBreakdown(BaseModel):
    education: CategoryScore
    experience: CategoryScore
    skills: CategoryScore
    tools_and_technologies: CategoryScore
    knowledge_and_domain: CategoryScore


class AnalysisResult(BaseModel):
    job_title: str
    # Not range-validated here: this is the LLM's raw self-reported total, which
    # app/llm/analyzer.py::_blend_and_finalize immediately recomputes from the
    # (validated) category scores. Enforcing ge=0/le=100 on the untrusted raw
    # value would 502 on a simple LLM arithmetic slip instead of being corrected.
    overall_score: int
    max_score: int = 100
    breakdown: ScoreBreakdown
    strengths: List[str]
    gaps: List[str]
    recommendation: str
    raw_resume_summary: str


class AnalyzeResponse(BaseModel):
    resume_id: str
    filename: str
    extraction_method: str  # "text" or "vision"
    extracted_text: str
    result: AnalysisResult
