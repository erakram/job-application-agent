# Job Application Agent

![tests](https://github.com/erakram/job-application-agent/actions/workflows/tests.yml/badge.svg)

Give it a resume and a job posting (a link or the text). An AI agent opens the posting, researches the company on the web, and returns an **honest match score**, strengths with evidence from the resume, gaps with ways to close them, resume tweaks, and a **tailored cover letter**.

Built with **Groq** (browser search on `gpt-oss-20b`, structured output on `gpt-oss-120b`) and **FastAPI**. Runs entirely on Groq's free tier.

<!-- Add a screenshot or demo GIF here: ![demo](docs/demo.gif) -->

## How it works

```
Resume (PDF/TXT/MD or text) ─► text extraction (in memory, never stored)

Job URL or text ─► Step 1: research agent (gpt-oss-20b + browser_search)
                     ├─ open the job URL and extract the posting
                     └─ search the web for the company
                   Groq's tool log is checked:
                     ├─ no page opened for a URL? ─► "paste the text instead"
                     └─ no real search? ─► company notes discarded (could be from memory)

Step 2: analysis (gpt-oss-120b, strict JSON schema)
   resume + posting + company notes ─► score, strengths, gaps, tweaks, cover letter
                   validated again with pydantic ─► UI
```

**Why two steps.** Groq's browser search can't be combined with structured output. So a small, fast model does the browsing, and a larger model writes the report under a strict JSON schema. The two models also have separate free-tier rate limits, so the steps don't throttle each other.

**Honesty rules.** Strengths must point to a specific line in the resume, or they count as gaps. The cover letter may not claim anything the resume doesn't say: no invented courses, projects or "currently learning X". This was tested on real postings, and the prompt was tightened after the model invented a side project.

**Grounded research.** Company notes are used only if the agent actually ran a web search. The UI lists the pages it read.

## Engineering details

- **Structured output** — a strict JSON schema generated from the pydantic model, so every level has `required` and `additionalProperties: false`. The result is validated again, and the score is clamped to 0–100.
- **Tool-log verification** — the job posting must come from a page the agent opened, and company facts from a search it ran.
- **Privacy** — resumes are read in memory, never saved, and never logged.
- **Safe uploads** — PDF/TXT/MD only, 5 MB limit, clear errors for scanned or broken PDFs, and long input truncated to stay within free-tier token limits.
- **Reliability** — timeouts, retries with backoff, and clear 422/429/503/504/502 errors, including "this link needs a login, paste the text".
- **Try it instantly** — a fictional sample resume and posting are one click away in the UI.
- **Tests** — 34 pytest tests for resume parsing, section parsing, schema strictness, the research and analysis flow (with a fake LLM, so no API key is needed), and the API.

## Run it locally

Requires Python 3.11+ and a free Groq API key from <https://console.groq.com/keys>.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then paste your key into .env
uvicorn app.main:app --reload
```

Open <http://localhost:8000> and click **Try a sample**. Run the tests with `pytest -q`.

## API

`POST /api/analyze` (multipart form)

| Field | Purpose |
|---|---|
| `job` | Job posting URL, or the posting text |
| `resume` | Resume file (PDF, TXT or MD), **or** |
| `resume_text` | Resume as text |
| `research_company` | `true` (default) or `false` |

Response: `report` (`role_title`, `company`, `match_score`, `summary`, `strengths`, `gaps`, `resume_tweaks`, `cover_letter`), `company_notes`, `sources`, `steps`, `latency_ms`.

`GET /api/sample` returns the sample resume and posting. `GET /api/health` returns status. Interactive API docs are at `/docs`.

## Deploy (free)

1. Push this folder to a GitHub repository.
2. On [Render](https://render.com), choose **New → Blueprint** and select the repo (it uses `render.yaml` and the `Dockerfile`).
3. Enter your `GROQ_API_KEY` when prompted.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | — | Required |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Writes the structured report |
| `RESEARCH_MODEL` | `openai/gpt-oss-20b` | Browses the web (needs browser search support) |
| `REQUEST_TIMEOUT` / `MAX_RETRIES` | `90` / `3` | LLM request timeout and retries |
| `MAX_UPLOAD_MB` | `5` | Resume upload size limit |

## Project structure

```
app/
  main.py      FastAPI app: analyze, sample, health, error handling
  agent.py     Research step, analysis step, schema, honesty checks
  resume.py    PDF/text extraction
  config.py    Settings from environment variables
  static/      Web UI (plain HTML/JS)
sample_data/   Fictional sample resume and job posting
tests/         pytest suite (no API key needed)
```
