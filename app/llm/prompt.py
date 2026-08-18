JOB_TITLE = "AI & Data Solution Intern"

# Per-category "ideal candidate" description. Doubles as the reference text
# embedded for the semantic-similarity signal in app/llm/embeddings.py, so the
# LLM rubric and the embedding signal always describe the same criteria.
CATEGORY_LABELS = {
    "education": "Education",
    "experience": "Experience",
    "skills": "Skills",
    "tools_and_technologies": "Tools & Technologies",
    "knowledge_and_domain": "Knowledge & Domain",
}

CATEGORY_REFERENCE_TEXTS = {
    "education": (
        "Currently pursuing or recently completed a degree in Computer Science, "
        "Data Science, AI, Statistics, Engineering, or a related field."
    ),
    "experience": (
        "Prior internships, academic projects, hackathons, or personal projects "
        "involving data analysis, machine learning, or software development."
    ),
    "skills": (
        "Programming (especially Python), data manipulation, problem solving, "
        "statistics/ML fundamentals, ability to communicate technical findings clearly."
    ),
    "tools_and_technologies": (
        "Python data/ML stack (pandas, numpy, scikit-learn, PyTorch/TensorFlow), SQL, "
        "cloud platforms (GCP/AWS/Azure), Git, LLM/GenAI tooling, BI or visualization tools."
    ),
    "knowledge_and_domain": (
        "Understanding of AI/ML concepts, data pipelines, prompt engineering / LLM "
        "applications, and general business/data-solution context."
    ),
}

CATEGORY_MAX_POINTS = {
    "education": 20,
    "experience": 25,
    "skills": 25,
    "tools_and_technologies": 15,
    "knowledge_and_domain": 15,
}

JOB_DESCRIPTION = (
    f"Position: {JOB_TITLE}\n\n"
    "We are looking for an intern to support the design and delivery of AI and data "
    "solutions. Responsibilities and areas we evaluate candidates on:\n\n"
    + "\n".join(
        f"- {CATEGORY_LABELS[key]}: {text}" for key, text in CATEGORY_REFERENCE_TEXTS.items()
    )
)

SCORING_RUBRIC = f"""
Score the candidate against the job description above across exactly these five
categories, using the max points shown:

{chr(10).join(f"- {key} (max {CATEGORY_MAX_POINTS[key]})" for key in CATEGORY_REFERENCE_TEXTS)}

(The five max values sum to 100 = max_score.)

For each category give an integer score from 0 up to that category's max, and a
short (1-3 sentence) reasoning grounded in specific evidence quoted or paraphrased
from the resume. If the resume provides no evidence for a category, score it low
and say so explicitly rather than guessing.

overall_score must equal the sum of the five category scores.
"""

SYSTEM_PROMPT = f"""You are a strict, evidence-based technical recruiter assistant.
You evaluate resumes against a specific job description and return your assessment
as a single JSON object only — no markdown fences, no commentary before or after it.

{JOB_DESCRIPTION}

{SCORING_RUBRIC}

You may also be given independently computed semantic-similarity signals between
the resume and each category's ideal-candidate description. Treat these only as a
soft supporting signal — your score and reasoning must still be grounded in
specific evidence from the resume text itself, not the signal alone.

Respond with a single JSON object matching exactly this shape:

{{
  "job_title": "{JOB_TITLE}",
  "overall_score": <int, 0-100, sum of the five category scores>,
  "max_score": 100,
  "breakdown": {{
    "education": {{"score": <int>, "max": 20, "reasoning": "<string>"}},
    "experience": {{"score": <int>, "max": 25, "reasoning": "<string>"}},
    "skills": {{"score": <int>, "max": 25, "reasoning": "<string>"}},
    "tools_and_technologies": {{"score": <int>, "max": 15, "reasoning": "<string>"}},
    "knowledge_and_domain": {{"score": <int>, "max": 15, "reasoning": "<string>"}}
  }},
  "strengths": ["<short bullet>", "..."],
  "gaps": ["<short bullet>", "..."],
  "recommendation": "<1-2 sentence hiring recommendation>",
  "raw_resume_summary": "<2-4 sentence neutral summary of the candidate>"
}}

Return ONLY that JSON object, with no surrounding text.
"""


def build_user_prompt(resume_text: str, semantic_signals: dict[str, float] | None = None) -> str:
    signals_section = ""
    if semantic_signals:
        signal_lines = "\n".join(
            f"- {CATEGORY_LABELS[key]}: {value}/100" for key, value in semantic_signals.items()
        )
        signals_section = (
            "\n\nIndependently computed semantic-similarity signals (embedding cosine "
            "similarity, 0-100, between the resume and each category's ideal-candidate "
            "description):\n" + signal_lines
        )

    return (
        f"Here is the candidate's resume, extracted from a PDF:\n\n---\n{resume_text}\n---"
        f"{signals_section}\n\nEvaluate this resume and return the JSON object as instructed."
    )
