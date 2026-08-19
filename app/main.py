import logging
import secrets
import uuid
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import settings
from app.extractors.pdf_extractor import extract_text
from app.llm.analyzer import LLMAnalysisError, analyze_resume
from app.llm.prompt import JOB_TITLE
from app.schemas import AnalyzeResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path(settings.upload_dir)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Resume Analyzer",
    description=f"Scores a PDF resume against the '{JOB_TITLE}' role using an LLM.",
    version="1.0.0",
)

_bearer_scheme = HTTPBearer(auto_error=False)


async def require_auth(credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme)) -> None:
    if not settings.api_auth_token:
        return
    if credentials is None or not secrets.compare_digest(credentials.credentials, settings.api_auth_token):
        raise HTTPException(status_code=401, detail="Invalid or missing API token")


@app.get("/health")
def health():
    return {"status": "ok", "job_title": JOB_TITLE, "llm_model": settings.llm_model}


@app.post("/analyze-resume", response_model=AnalyzeResponse, dependencies=[Depends(require_auth)])
async def analyze_resume_endpoint(file: UploadFile = File(...)):
    filename = file.filename or "upload.pdf"
    if file.content_type != "application/pdf" and not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    max_bytes = settings.max_upload_mb * 1024 * 1024
    chunks = bytearray()
    while chunk := await file.read(1024 * 1024):
        chunks.extend(chunk)
        if len(chunks) > max_bytes:
            raise HTTPException(status_code=400, detail=f"File exceeds {settings.max_upload_mb}MB limit")
    contents = bytes(chunks)

    resume_id = str(uuid.uuid4())
    stored_path = UPLOAD_DIR / f"{resume_id}.pdf"
    stored_path.write_bytes(contents)

    try:
        extraction = await run_in_threadpool(extract_text, str(stored_path))
    except Exception as e:
        logger.exception("PDF extraction failed")
        raise HTTPException(status_code=422, detail=f"Failed to extract text from PDF: {e}") from e

    if not extraction.text.strip():
        raise HTTPException(status_code=422, detail="No extractable text found in PDF")

    try:
        result = await run_in_threadpool(analyze_resume, extraction.text)
    except LLMAnalysisError as e:
        logger.exception("LLM analysis failed")
        raise HTTPException(status_code=502, detail=str(e)) from e

    return AnalyzeResponse(
        resume_id=resume_id,
        filename=filename,
        extraction_method=extraction.method,
        extracted_text=extraction.text,
        result=result,
    )


@app.get("/resumes/{resume_id}", dependencies=[Depends(require_auth)])
async def get_resume(resume_id: str):
    try:
        parsed_id = uuid.UUID(resume_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid resume_id")

    path = UPLOAD_DIR / f"{parsed_id}.pdf"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Resume not found")

    return FileResponse(path, media_type="application/pdf", filename=f"{parsed_id}.pdf")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    logger.exception("Unhandled error")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
