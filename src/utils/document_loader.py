"""Document loading for PDF, DOCX, Markdown."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

from src.utils.schema import DocumentMetadata


def load_text_file(path: Path) -> str:
    """Load plain text or markdown file."""
    with open(path) as f:
        return f.read()


def load_pdf(path: Path) -> str:
    """Load PDF and extract text."""
    from pypdf import PdfReader
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_docx(path: Path) -> str:
    """Load DOCX and extract text."""
    from docx import Document
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


def load_document(path: Path) -> tuple[str, DocumentMetadata]:
    """Load document and return (content, metadata)."""
    path = Path(path)
    name = path.name
    ext = path.suffix.lower()
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)

    if ext == ".pdf":
        content = load_pdf(path)
    elif ext in (".docx", ".doc"):
        content = load_docx(path)
    elif ext in (".md", ".txt", ".markdown"):
        content = load_text_file(path)
    else:
        try:
            content = load_text_file(path)
        except Exception:
            content = ""

    metadata = DocumentMetadata(
        source=name,
        timestamp=mtime,
        author="",
        publisher="",
        trust_score=0.5,
        version_id="",
    )
    return content, metadata


def load_documents_from_dir(
    dir_path: Path,
    extensions: tuple[str, ...] = (".pdf", ".docx", ".md", ".txt"),
) -> list[tuple[str, DocumentMetadata]]:
    """Load all supported documents from a directory."""
    results = []
    for path in Path(dir_path).rglob("*"):
        if path.suffix.lower() in extensions and path.is_file():
            try:
                content, meta = load_document(path)
                if content.strip():
                    results.append((content, meta))
            except Exception:
                pass
    return results
