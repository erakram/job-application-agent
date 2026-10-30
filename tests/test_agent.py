from datetime import date

import pytest

from app.agent import (FIT_REPORT_SCHEMA, AnalysisError, FitReport, JobApplicationAgent,
                       JobPostingUnavailable, browsing_activity, is_url, parse_sections)
from tests.conftest import POSTING, REPORT, FakeGroq, llm_message, open_tool, search_tool

URL = "https://jobs.example.com/123"
RESEARCH_REPLY = f"POSTING:\n{POSTING}\nCompany: Northwind Labs\n\nCOMPANY:\n- Builds AI support tools.\n- 200 staff."


def make_agent(responses):
    client = FakeGroq(responses)
    return client, JobApplicationAgent(client, "big-model", "small-model", today=date(2026, 10, 30))


def test_is_url():
    assert is_url(" https://a.com/job ") and not is_url("Python developer at https://a.com")


def test_parse_sections_ignores_company_lines_inside_the_posting():
    posting, company = parse_sections(RESEARCH_REPLY)
    assert "Company: Northwind Labs" in posting  # a lowercase "Company:" is part of the posting
    assert company.startswith("- Builds AI support tools.")


def test_parse_sections_handles_markdown_headings_and_groq_markers():
    posting, company = parse_sections("**POSTING:**\nThe job 【1†L2】\n**COMPANY:**\n- Facts")
    assert posting == "The job" and company == "- Facts"


def test_browsing_activity_counts_pages_and_searches():
    urls, searches = browsing_activity([search_tool("northwind"), open_tool(URL), open_tool(URL)])
    assert urls == [URL] and searches == 1


def test_schema_is_strict_at_every_level():
    assert FIT_REPORT_SCHEMA["additionalProperties"] is False
    assert set(FIT_REPORT_SCHEMA["required"]) == set(FitReport.model_fields)
    for definition in FIT_REPORT_SCHEMA["$defs"].values():
        assert definition["additionalProperties"] is False
        assert set(definition["required"]) == set(definition["properties"])


def test_score_is_clamped():
    assert FitReport.model_validate({**REPORT, "match_score": 140}).match_score == 100


def test_pasted_posting_without_research_makes_one_structured_call():
    client, agent = make_agent([llm_message(REPORT)])
    result = agent.analyze("My resume " * 20, POSTING, research_company=False)

    assert result.report.match_score == 78 and result.company_notes is None
    assert len(client.requests) == 1
    request = client.requests[0]
    assert request["model"] == "big-model"
    assert request["response_format"]["json_schema"]["strict"] is True
    assert POSTING.strip() in request["messages"][1]["content"]


def test_url_is_opened_and_company_researched_with_the_small_model():
    research = llm_message(RESEARCH_REPLY, [open_tool(URL), search_tool("Northwind Labs")])
    client, agent = make_agent([research, llm_message(REPORT)])
    result = agent.analyze("My resume " * 20, URL)

    browse, analysis = client.requests
    assert browse["model"] == "small-model" and browse["tools"] == [{"type": "browser_search"}]
    assert "Required: run at least one web search" in browse["messages"][0]["content"]
    assert "Company: Northwind Labs" in analysis["messages"][1]["content"]  # posting text came from the page
    assert "<company_research>" in analysis["messages"][1]["content"]
    assert result.sources == [URL] and result.company_notes.startswith("- Builds")
    assert result.steps[0] == f"Read the job posting at {URL}"


def test_company_notes_without_a_real_search_are_discarded():
    research = llm_message(RESEARCH_REPLY, [open_tool(URL)])  # opened the posting, never searched
    client, agent = make_agent([research, llm_message(REPORT)])
    result = agent.analyze("My resume " * 20, URL)

    assert result.company_notes is None
    assert "<company_research>" not in client.requests[1]["messages"][1]["content"]


@pytest.mark.parametrize("reply, tools", [
    ("POSTING:\nUNAVAILABLE\nCOMPANY:\nUNKNOWN", [open_tool(URL)]),
    ("POSTING:\nLogin to view\nCOMPANY:\nUNKNOWN", [open_tool(URL)]),  # too short to be a posting
    (RESEARCH_REPLY, []),  # claims a posting without opening any page
])
def test_unreadable_job_links_raise(reply, tools):
    _, agent = make_agent([llm_message(reply, tools)])
    with pytest.raises(JobPostingUnavailable):
        agent.analyze("My resume " * 20, URL)


def test_skipping_research_for_a_url_tells_the_model_not_to_search():
    research = llm_message(RESEARCH_REPLY.replace("- Builds AI support tools.\n- 200 staff.", "SKIPPED"),
                           [open_tool(URL)])
    client, agent = make_agent([research, llm_message(REPORT)])
    result = agent.analyze("My resume " * 20, URL, research_company=False)
    assert "write SKIPPED" in client.requests[0]["messages"][0]["content"]
    assert result.company_notes is None


@pytest.mark.parametrize("content", ["not json", '{"role_title": "x"}'])
def test_invalid_report_raises_analysis_error(content):
    _, agent = make_agent([llm_message(content)])
    with pytest.raises(AnalysisError):
        agent.analyze("My resume " * 20, POSTING, research_company=False)
