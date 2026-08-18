import html
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
PUBLIC_API_BASE_URL = os.getenv("PUBLIC_API_BASE_URL", API_BASE_URL)
API_REQUEST_TIMEOUT = int(os.getenv("API_REQUEST_TIMEOUT", "180"))
FEEDBACK_LOG = Path(os.getenv("FEEDBACK_LOG_PATH", "data/feedback_log.jsonl"))
FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)

CATEGORY_LABELS = {
    "education": "Education",
    "experience": "Experience",
    "skills": "Skills",
    "tools_and_technologies": "Tools & Technologies",
    "knowledge_and_domain": "Knowledge & Domain",
}

st.set_page_config(page_title="Resume Analyzer", page_icon="📋", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=Prompt:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', 'Prompt', sans-serif;
    }
    code, .mono {
        font-family: 'IBM Plex Mono', monospace !important;
    }

    #MainMenu, footer, header {visibility: hidden;}

    .block-container {
        max-width: 880px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    h1 {
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    h2, h3 {
        font-weight: 600;
        letter-spacing: -0.01em;
    }

    .app-subtitle {
        color: var(--text-color-secondary, #6b7280);
        font-size: 0.95rem;
        margin-top: -0.5rem;
        margin-bottom: 1.5rem;
    }

    .category-card {
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.9rem;
    }
    .category-card h4 {
        margin: 0 0 0.4rem 0;
        font-size: 1.05rem;
        font-weight: 600;
    }
    .category-card p {
        margin: 0 0 0.7rem 0;
        font-size: 0.92rem;
        line-height: 1.5;
        color: var(--text-color-secondary, #6b7280);
    }

    .score-badge {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 0.4rem;
    }
    .score-badge.high { background: rgba(34, 197, 94, 0.16); color: #16a34a; }
    .score-badge.mid  { background: rgba(234, 179, 8, 0.18); color: #b45309; }
    .score-badge.low  { background: rgba(239, 68, 68, 0.14); color: #dc2626; }
    .score-badge.muted { background: rgba(128, 128, 128, 0.14); color: #6b7280; }

    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def score_tier(score: int, max_score: int) -> str:
    ratio = score / max_score if max_score else 0
    if ratio >= 0.8:
        return "high"
    if ratio >= 0.5:
        return "mid"
    return "low"


def badge(text: str, tier: str) -> str:
    return f'<span class="score-badge {tier}">{text}</span>'


st.title("Resume Analyzer")
st.markdown('<div class="app-subtitle">Human review for AI-scored resumes</div>', unsafe_allow_html=True)

with st.sidebar:
    st.caption(f"Connected to: {API_BASE_URL}")

if "result" not in st.session_state:
    st.session_state.result = None
    st.session_state.filename = None
    st.session_state.run_id = 0

with st.container(border=True):
    uploaded = st.file_uploader("Upload resume (PDF)", type=["pdf"], label_visibility="collapsed")
    analyze_clicked = st.button("Analyze", type="primary", disabled=not uploaded)

if uploaded and analyze_clicked:
    with st.spinner("Extracting text and scoring…"):
        try:
            resp = requests.post(
                f"{API_BASE_URL}/analyze-resume",
                files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                timeout=API_REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            st.session_state.result = resp.json()
            st.session_state.filename = uploaded.name
            st.session_state.run_id += 1
        except requests.RequestException as e:
            detail = None
            if e.response is not None:
                try:
                    detail = e.response.json().get("detail")
                except ValueError:
                    detail = e.response.text
            st.error(f"Analysis failed: {detail or e}")
            st.session_state.result = None

if st.session_state.result:
    data = st.session_state.result
    result = data["result"]

    resume_url = f"{PUBLIC_API_BASE_URL}/resumes/{data['resume_id']}"
    st.caption(f"{data['filename']} · extracted via {data['extraction_method']} · [view saved PDF]({resume_url})")

    is_vision = data["extraction_method"] == "vision"
    with st.expander("View raw extracted text", expanded=is_vision):
        if is_vision:
            st.warning(
                "This PDF had no selectable text and was transcribed from page images by "
                "the LLM instead of extracted directly. Transcription can miss or misread "
                "text — check it against the saved PDF above before trusting the scores below.",
                icon="⚠️",
            )
        st.text_area(
            "Extracted text",
            value=data["extracted_text"],
            height=300,
            disabled=True,
            label_visibility="collapsed",
        )

    col1, col2 = st.columns(2)
    col1.metric("Overall score", f"{result['overall_score']} / {result['max_score']}")

    st.subheader("Category breakdown")

    overrides = {}
    for key, label in CATEGORY_LABELS.items():
        cat = result["breakdown"][key]
        sim = cat["semantic_similarity"]
        tier = score_tier(cat["score"], cat["max"])

        with st.container():
            st.markdown(
                f"""
                <div class="category-card">
                    <h4>{html.escape(label)}</h4>
                    <p>{html.escape(cat["reasoning"])}</p>
                    {badge(f"Blended {cat['score']}/{cat['max']}", tier)}
                    {badge(f"LLM {cat['llm_score']}", "muted")}
                    {badge(f"Similarity {sim if sim is not None else 'n/a'}", "muted")}
                </div>
                """,
                unsafe_allow_html=True,
            )
            overrides[key] = st.number_input(
                f"Human override — {label}",
                min_value=0,
                max_value=cat["max"],
                value=cat["score"],
                key=f"override_{st.session_state.run_id}_{key}",
            )

    human_total = sum(overrides.values())
    col2.metric("Human-adjusted score", f"{human_total} / {result['max_score']}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Strengths")
        for s in result["strengths"]:
            st.markdown(f"- {s}")
    with col_b:
        st.subheader("Gaps")
        for g in result["gaps"]:
            st.markdown(f"- {g}")

    st.subheader("Recommendation")
    st.info(result["recommendation"])

    st.subheader("Resume summary")
    st.write(result["raw_resume_summary"])

    st.subheader("Review decision")
    decision = st.radio("Decision", ["approve", "needs changes", "reject"], horizontal=True, label_visibility="collapsed")
    notes = st.text_area("Reviewer notes (optional)")

    if st.button("Submit review", type="primary"):
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "resume_id": data["resume_id"],
            "filename": st.session_state.filename,
            "model_result": result,
            "human_overrides": overrides,
            "human_overall_score": human_total,
            "decision": decision,
            "notes": notes,
        }
        with open(FEEDBACK_LOG, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        st.success(f"Saved review to {FEEDBACK_LOG}")
