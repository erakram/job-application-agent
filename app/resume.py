"""Turn an uploaded resume (PDF, TXT or MD) into plain text. Nothing is saved to disk."""

import io

from pypdf import PdfReader
from pypdf.errors import PdfReadError

SUPPORTED_TYPES = (".pdf", ".txt", ".md")
MAX_CHARS = 12_000  # about 3,000 tokens: plenty for a resume, and keeps requests within free-tier limits


class ResumeError(ValueError):
    """The resume could not be read; the message is safe to show to the user."""


def extract_text(filename: str, data: bytes) -> str:
    name = (filename or "").lower()
    if not name.endswith(SUPPORTED_TYPES):
        raise ResumeError("Please upload a PDF, TXT or MD file.")
    if name.endswith(".pdf"):
        try:
            reader = PdfReader(io.BytesIO(data))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except (PdfReadError, ValueError, KeyError):
            raise ResumeError("This PDF could not be read. Try exporting it again, or paste the text.")
    else:
        text = data.decode("utf-8", errors="replace")
    return clean(text)


def clean(text: str) -> str:
    text = "\n".join(line.rstrip() for line in (text or "").splitlines()).strip()
    if len(text) < 50:
        raise ResumeError("No text found in the resume. If it's a scanned PDF, paste the text instead.")
    return text[:MAX_CHARS]
