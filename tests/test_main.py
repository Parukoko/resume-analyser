import json
import uuid
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import app.config as config_module
from app.extractors.pdf_extractor import ExtractionResult
from app.main import app

FAKE_RESULT = {
    "job_title": "AI & Data Solution Intern",
    "overall_score": 78,
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


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mocked_pipeline():
    with patch(
        "app.main.extract_text",
        return_value=ExtractionResult(text="Some extracted resume text " * 5, method="text"),
    ), patch("app.llm.analyzer._client") as mock_client, patch(
        "app.llm.analyzer.compute_semantic_signals", return_value={}
    ):
        mock_resp = mock_client.chat.completions.create.return_value
        mock_resp.choices = [type("C", (), {"message": type("M", (), {"content": json.dumps(FAKE_RESULT)})()})]
        yield


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_analyze_rejects_non_pdf(client):
    resp = client.post("/analyze-resume", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert resp.status_code == 400


def test_analyze_rejects_oversized_upload(client, mocked_pipeline, monkeypatch):
    monkeypatch.setattr(config_module.settings, "max_upload_mb", 0)
    resp = client.post("/analyze-resume", files={"file": ("r.pdf", b"x" * 2000, "application/pdf")})
    assert resp.status_code == 400


def test_analyze_resume_persists_upload_and_returns_uuid(client, mocked_pipeline, isolated_upload_dir):
    pdf_bytes = b"%PDF-1.4 fake pdf content"
    resp = client.post("/analyze-resume", files={"file": ("resume.pdf", pdf_bytes, "application/pdf")})

    assert resp.status_code == 200
    data = resp.json()
    uuid.UUID(data["resume_id"])
    assert (isolated_upload_dir / f"{data['resume_id']}.pdf").read_bytes() == pdf_bytes
    assert data["extracted_text"]
    assert data["result"]["overall_score"] == 78


def test_get_resume_roundtrip(client, mocked_pipeline):
    pdf_bytes = b"%PDF-1.4 fake pdf content"
    resp = client.post("/analyze-resume", files={"file": ("resume.pdf", pdf_bytes, "application/pdf")})
    resume_id = resp.json()["resume_id"]

    dl = client.get(f"/resumes/{resume_id}")
    assert dl.status_code == 200
    assert dl.content == pdf_bytes
    assert dl.headers["content-type"] == "application/pdf"


@pytest.mark.parametrize("bad_id", ["not-a-uuid", "...", "12345", "a..b", "DROP TABLE resumes"])
def test_get_resume_rejects_invalid_id(client, bad_id):
    resp = client.get(f"/resumes/{bad_id}")
    assert resp.status_code == 400


def test_get_resume_404_for_unknown_uuid(client):
    resp = client.get(f"/resumes/{uuid.uuid4()}")
    assert resp.status_code == 404


def test_auth_disabled_by_default(client, mocked_pipeline):
    resp = client.post("/analyze-resume", files={"file": ("r.pdf", b"x", "application/pdf")})
    assert resp.status_code == 200


def test_auth_rejects_missing_token_when_enabled(client, mocked_pipeline, monkeypatch):
    monkeypatch.setattr(config_module.settings, "api_auth_token", "secret123")
    resp = client.post("/analyze-resume", files={"file": ("r.pdf", b"x", "application/pdf")})
    assert resp.status_code == 401


def test_auth_rejects_wrong_token_when_enabled(client, mocked_pipeline, monkeypatch):
    monkeypatch.setattr(config_module.settings, "api_auth_token", "secret123")
    resp = client.post(
        "/analyze-resume",
        files={"file": ("r.pdf", b"x", "application/pdf")},
        headers={"Authorization": "Bearer wrong"},
    )
    assert resp.status_code == 401


def test_auth_accepts_correct_token_when_enabled(client, mocked_pipeline, monkeypatch):
    monkeypatch.setattr(config_module.settings, "api_auth_token", "secret123")
    resp = client.post(
        "/analyze-resume",
        files={"file": ("r.pdf", b"x", "application/pdf")},
        headers={"Authorization": "Bearer secret123"},
    )
    assert resp.status_code == 200


def test_auth_also_protects_resume_download(client, mocked_pipeline, monkeypatch):
    resp = client.post("/analyze-resume", files={"file": ("r.pdf", b"x", "application/pdf")})
    resume_id = resp.json()["resume_id"]

    monkeypatch.setattr(config_module.settings, "api_auth_token", "secret123")
    dl = client.get(f"/resumes/{resume_id}")
    assert dl.status_code == 401
