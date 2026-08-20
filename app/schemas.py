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
    score: int = Field(..., ge=0, description="Score after blending llm_score with semantic_similarity.")
    max: int = Field(..., gt=0)
    reasoning: str
    llm_score: Optional[int] = Field(default=None, description="Raw score from the LLM, before blending.")
    semantic_similarity: Optional[float] = Field(
        default=None, description="Embedding cosine similarity (0-100), null if unavailable."
    )


class ScoreBreakdown(BaseModel):
    education: CategoryScore
    experience: CategoryScore
    skills: CategoryScore
    tools_and_technologies: CategoryScore
    knowledge_and_domain: CategoryScore


class AnalysisResult(BaseModel):
    job_title: str
    # recomputed in analyzer._blend_and_finalize, not trusted from the LLM
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
