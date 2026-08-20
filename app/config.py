import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    llm_model: str = os.getenv("LLM_MODEL", "gemini-3.6-flash")
    llm_api_key: str = os.getenv("LLM_API_KEY", "unset-llm-api-key")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    llm_timeout: float = float(os.getenv("LLM_TIMEOUT", "60"))
    llm_max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "0"))
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
    semantic_weight: float = float(os.getenv("SEMANTIC_WEIGHT", "0.2"))
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "10"))
    ocr_fallback_char_threshold: int = int(os.getenv("OCR_FALLBACK_CHAR_THRESHOLD", "20"))
    upload_dir: str = os.getenv("UPLOAD_DIR", "data/uploads")
    api_auth_token: str = os.getenv("API_AUTH_TOKEN", "")

settings = Settings()
