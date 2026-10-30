"""Settings, read from environment variables (or a local .env file)."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    groq_api_key: str | None
    model: str  # writes the structured fit report
    # Browses the web (job posting + company). A different model has its own free-tier
    # rate limit, so the two steps don't compete for the same token budget.
    research_model: str
    request_timeout: float  # seconds per LLM request; web browsing makes these slower
    max_retries: int  # automatic retries on rate limits, timeouts and 5xx errors
    max_upload_mb: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            groq_api_key=os.getenv("GROQ_API_KEY") or None,
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            research_model=os.getenv("RESEARCH_MODEL", "openai/gpt-oss-20b"),
            request_timeout=float(os.getenv("REQUEST_TIMEOUT", "90")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "5")),
        )
