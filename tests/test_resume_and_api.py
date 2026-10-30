import io

import groq
import httpx
import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from app.agent import AnalysisError, FitReport, JobFitResult, JobPostingUnavailable
from app.main import app
from app.resume import MAX_CHARS, ResumeError, extract_text
from tests.conftest import POSTING, REPORT

RESUME = "Alex Rivera. Software engineer with Python, FastAPI and SQL experience. " * 3


def blank_pdf() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_extract_text_from_text_files_and_truncates():
    assert extract_text("cv.md", RESUME.encode()).startswith("Alex Rivera")
    assert len(extract_text("cv.txt", ("x" * 100 + "\n").encode() * 500)) == MAX_CHARS


@pytest.mark.parametrize("name, data, message", [
    ("cv.docx", b"anything", "PDF, TXT or MD"),
    ("cv.pdf", b"not a pdf", "could not be read"),
    ("cv.pdf", blank_pdf(), "No text found"),
    ("cv.txt", b"too short", "No text found"),
])
def test_extract_text_rejects_bad_files(name, data, message):
    with pytest.raises(ResumeError, match=message):
        extract_text(name, data)


class StubAgent:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result, error, []

    def analyze(self, resume, job, research_company):
        self.calls.append((resume, job, research_company))
        if self.error:
            raise self.error
        return self.result


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def agent():
    stub = StubAgent(JobFitResult(report=FitReport.model_validate(REPORT), steps=["Compared"]))
    app.state.agent = stub
    return stub


def groq_error(cls, status):
    response = httpx.Response(status, request=httpx.Request("POST", "https://api.groq.com"))
    return cls("error", response=response, body=None)


def test_health_index_and_sample(client):
    assert client.get("/api/health").json()["llm_configured"] is False
    assert "Job Application Agent" in client.get("/").text
    sample = client.get("/api/sample").json()
    assert "Alex Rivera" in sample["resume"] and "Requirements" in sample["job"]


def test_analyze_without_api_key_returns_503(client):
    response = client.post("/api/analyze", data={"job": POSTING, "resume_text": RESUME})
    assert response.status_code == 503


def test_analyze_with_pasted_resume(client, agent):
    response = client.post("/api/analyze", data={"job": POSTING, "resume_text": RESUME, "research_company": "false"})
    assert response.status_code == 200
    data = response.json()
    assert data["report"]["match_score"] == 78 and data["steps"] == ["Compared"]
    assert agent.calls[0][2] is False


def test_analyze_with_uploaded_resume_and_job_url(client, agent):
    files = {"resume": ("cv.txt", RESUME.encode(), "text/plain")}
    response = client.post("/api/analyze", data={"job": "https://jobs.example.com/1"}, files=files)
    assert response.status_code == 200
    resume, job, research = agent.calls[0]
    assert resume.startswith("Alex Rivera") and job == "https://jobs.example.com/1" and research is True


@pytest.mark.parametrize("data, files, status", [
    ({"job": POSTING}, None, 422),  # no resume
    ({"job": "too short", "resume_text": RESUME}, None, 422),
    ({"job": POSTING}, {"resume": ("cv.docx", b"x", "application/octet-stream")}, 422),
    ({"job": POSTING}, {"resume": ("cv.txt", b"x" * (5 * 1024 * 1024 + 1), "text/plain")}, 413),
])
def test_bad_input_is_rejected(client, agent, data, files, status):
    response = client.post("/api/analyze", data=data, files=files)
    assert response.status_code == status
    assert isinstance(response.json()["detail"], str)
    assert agent.calls == []


@pytest.mark.parametrize("error, status", [
    (JobPostingUnavailable("Paste the job description text instead."), 422),
    (groq_error(groq.RateLimitError, 429), 429),
    (groq.APITimeoutError(request=httpx.Request("POST", "https://api.groq.com")), 504),
    (groq_error(groq.AuthenticationError, 401), 503),
    (AnalysisError("incomplete"), 502),
    (groq_error(groq.InternalServerError, 500), 502),
])
def test_errors_map_to_clear_http_statuses(client, error, status):
    app.state.agent = StubAgent(error=error)
    response = client.post("/api/analyze", data={"job": POSTING, "resume_text": RESUME})
    assert response.status_code == status
    assert isinstance(response.json()["detail"], str)
