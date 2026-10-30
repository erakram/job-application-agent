"""FastAPI app: the /api/analyze endpoint, sample data, a health check, and the web page."""

import logging
import time
import uuid
from contextlib import asynccontextmanager

import groq
from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from app.agent import AnalysisError, JobPostingUnavailable, build_agent, is_url
from app.config import PROJECT_ROOT, Settings
from app.resume import ResumeError, clean, extract_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")

STATIC_DIR = PROJECT_ROOT / "app" / "static"
SAMPLE_DIR = PROJECT_ROOT / "sample_data"
MAX_JOB_CHARS = 20_000


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings.from_env()
    app.state.settings = settings
    # Without a key the UI still loads; /api/analyze explains what's missing.
    app.state.agent = build_agent(settings) if settings.groq_api_key else None
    if app.state.agent is None:
        logger.warning("GROQ_API_KEY is not set; /api/analyze will return 503.")
    yield


app = FastAPI(title="Job Application Agent", lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Give each request an id and log its status and latency. Resume text is never logged."""
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:8]
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    logger.info("request id=%s %s %s status=%d ms=%.0f", request_id, request.method,
                request.url.path, response.status_code, (time.perf_counter() - start) * 1000)
    return response


async def read_resume(file: UploadFile | None, text: str | None, max_mb: int) -> str:
    """Exactly one of an uploaded file or pasted text. Read in memory; nothing is stored."""
    if file is not None and file.filename:
        data = await file.read(max_mb * 1024 * 1024 + 1)
        if len(data) > max_mb * 1024 * 1024:
            raise HTTPException(413, f"The resume file is larger than {max_mb} MB.")
        return extract_text(file.filename, data)
    if text and text.strip():
        return clean(text)
    raise HTTPException(422, "Upload a resume file or paste your resume text.")


@app.post("/api/analyze")
async def analyze(
    request: Request,
    job: str = Form(..., description="Job posting URL, or the posting text."),
    resume: UploadFile | None = None,
    resume_text: str | None = Form(None, max_length=50_000),
    research_company: bool = Form(True),
) -> dict:
    agent = request.app.state.agent
    if agent is None:
        raise HTTPException(503, "The server has no GROQ_API_KEY configured.")

    job = job.strip()
    if not is_url(job) and not 100 <= len(job) <= MAX_JOB_CHARS:
        raise HTTPException(422, "Paste a job posting URL, or the full job description (at least a few lines).")
    try:
        resume_content = await read_resume(resume, resume_text, request.app.state.settings.max_upload_mb)
    except ResumeError as e:
        raise HTTPException(422, str(e))

    start = time.perf_counter()
    try:
        # The Groq client is synchronous, so run it in a worker thread.
        result = await run_in_threadpool(agent.analyze, resume_content, job, research_company)
    except JobPostingUnavailable as e:
        raise HTTPException(422, str(e))
    except groq.RateLimitError:
        raise HTTPException(429, "The AI service is busy. Please try again in a minute.")
    except groq.APITimeoutError:
        raise HTTPException(504, "The AI service took too long to respond. Please try again.")
    except groq.APIConnectionError:
        raise HTTPException(503, "Could not reach the AI service. Please try again.")
    except groq.AuthenticationError:
        logger.error("Groq rejected the API key.")
        raise HTTPException(503, "The server's AI credentials are invalid.")
    except AnalysisError as e:
        raise HTTPException(502, str(e))
    except groq.APIStatusError as e:
        logger.error("Groq error status=%s: %s", e.status_code, e.message)
        raise HTTPException(502, "The AI service returned an error. Please try again.")

    return {
        "report": result.report.model_dump(),
        "company_notes": result.company_notes,
        "sources": result.sources,
        "steps": result.steps,
        "latency_ms": round((time.perf_counter() - start) * 1000),
    }


@app.get("/api/sample")
def sample() -> dict:
    """A fictional resume and job posting, so visitors can try the agent without their own."""
    return {"resume": (SAMPLE_DIR / "sample_resume.md").read_text(encoding="utf-8"),
            "job": (SAMPLE_DIR / "sample_job.md").read_text(encoding="utf-8")}


@app.get("/api/health")
def health(request: Request) -> dict:
    settings = request.app.state.settings
    return {"status": "ok", "llm_configured": request.app.state.agent is not None,
            "model": settings.model, "research_model": settings.research_model}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
