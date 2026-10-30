import json
from types import SimpleNamespace

REPORT = {
    "role_title": "Junior AI Engineer", "company": "Northwind Labs", "match_score": 78,
    "summary": "Good fit.", "strengths": [{"requirement": "Python", "evidence": "1.5 years at Brightline"}],
    "gaps": [{"requirement": "Docker", "suggestion": "Containerize a project"}],
    "resume_tweaks": ["Mention RAG evaluation"], "cover_letter": "Hi Northwind team, ...",
}

POSTING = "Junior AI Engineer at Northwind Labs. " + "Requirements: Python, FastAPI, RAG, Docker. " * 10


def open_tool(url: str) -> dict:
    return {"type": "browser.open", "name": "browser.open", "arguments": "{}",
            "output": f"L0: \nL1: URL: {url}\nL2: page text"}


def search_tool(query: str) -> dict:
    return {"type": "browser_search", "name": "browser.search", "arguments": f'{{"query": "{query}"}}',
            "output": "L0: \nL1: URL: https://exa.ai/search?q=x"}


def llm_message(content, executed_tools=None):
    """Shape of a Groq chat completion response, with just the fields the agent reads."""
    if isinstance(content, dict):
        content = json.dumps(content)
    message = SimpleNamespace(content=content, executed_tools=executed_tools or [], tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeGroq:
    """Stands in for the Groq client: returns scripted responses and records each request."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return self.responses.pop(0)
