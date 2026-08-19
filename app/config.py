import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Defaults to Google's Gemini free tier via its OpenAI-compatible endpoint —
    # fast (cloud-hosted), no local GPU/CPU inference needed. Get a free API key
    # at https://aistudio.google.com/apikey. To use a local model instead (e.g.
    # Ollama), override these three: LLM_BASE_URL=http://localhost:11434/v1,
    # LLM_MODEL=qwen3:8b, LLM_API_KEY=ollama (see README "Alternative: local LLM").
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    llm_model: str = os.getenv("LLM_MODEL", "gemini-3.6-flash")
    # Non-empty placeholder if unset: the OpenAI client raises at construction
    # time (crashing the whole app on import) if api_key is an empty string, so
    # an unset real key must instead fail later, per-request, with a clear
    # authentication error from the LLM provider.
    llm_api_key: str = os.getenv("LLM_API_KEY", "unset-llm-api-key")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    # A cloud model normally responds in seconds; this default leaves generous
    # headroom for rate-limit backoff. If you switch to a local CPU-only model
    # (e.g. Ollama with no GPU passthrough), raise this a lot — a 1.5 tokens/sec
    # CPU run can take several hundred seconds for one resume. See README
    # "Troubleshooting".
    llm_timeout: float = float(os.getenv("LLM_TIMEOUT", "60"))
    # OpenAI SDK default is 2 retries per call, which multiplies the effective
    # wait on a slow/unreachable backend instead of failing predictably at
    # ~llm_timeout. Default to 0; bump it back up if you want retry-on-timeout.
    llm_max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "0"))
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
    # Weight given to the embedding semantic-similarity signal when blending it into
    # each category's final score; the LLM's own score keeps weight (1 - this).
    semantic_weight: float = float(os.getenv("SEMANTIC_WEIGHT", "0.2"))
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "10"))
    ocr_fallback_char_threshold: int = int(os.getenv("OCR_FALLBACK_CHAR_THRESHOLD", "20"))
    upload_dir: str = os.getenv("UPLOAD_DIR", "data/uploads")
    # Shared-secret bearer token required on /analyze-resume and /resumes/{id}.
    # Empty (default) disables auth entirely — fine for local-only use, but set
    # this before exposing the API beyond localhost.
    api_auth_token: str = os.getenv("API_AUTH_TOKEN", "")

settings = Settings()
