"""Document parsing — PDF, DOCX, HTML, TXT/MD.

Returns unified (title, text, page_count) shape.
Streams large files to avoid memory spikes.
"""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def parse_document(file_path: str | Path, file_type: str | None = None) -> dict[str, Any]:
    """Parse a document file and return unified structure.

    Returns {"title": str, "text": str, "page_count": int, "success": bool, "error": str}
    """
    path = Path(file_path)
    ext = (file_type or path.suffix.lower()).lstrip(".")

    try:
        if ext in ("pdf",):
            return _parse_pdf(path)
        if ext in ("docx", "doc"):
            return _parse_docx(path)
        if ext in ("html", "htm"):
            return _parse_html(path)
        if ext in ("txt", "md", "markdown", "rst"):
            return _parse_text(path)
        return {"title": path.name, "text": "", "page_count": 0, "success": False,
                "error": f"Unsupported file type: {ext}"}
    except Exception as exc:
        logger.warning("Parse failed for %s: %s", path.name, exc)
        return {"title": path.name, "text": "", "page_count": 0, "success": False,
                "error": str(exc)}


def _parse_pdf(path: Path) -> dict[str, Any]:
    import pdfplumber

    text_parts: list[str] = []
    page_count = 0
    with pdfplumber.open(path) as pdf:
        page_count = len(pdf.pages)
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text.strip())

    full_text = "\n\n".join(text_parts)
    return {
        "title": path.stem,
        "text": full_text,
        "page_count": page_count,
        "success": bool(full_text.strip()),
        "error": "" if full_text.strip() else "No text extracted (possibly scanned PDF)",
    }


def _parse_docx(path: Path) -> dict[str, Any]:
    from docx import Document

    doc = Document(str(path))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    full_text = "\n\n".join(paragraphs)

    return {
        "title": path.stem,
        "text": full_text,
        "page_count": len(paragraphs) // 20 or 1,  # rough estimate
        "success": bool(full_text.strip()),
        "error": "",
    }


def _parse_html(path: Path) -> dict[str, Any]:
    from trafilatura import extract

    html_text = path.read_text(encoding="utf-8", errors="ignore")
    extracted = extract(html_text, output_format="markdown", with_metadata=True)
    text = extracted.strip() if extracted else ""

    # Try to extract title from first heading
    title = path.stem
    lines = text.splitlines()
    for line in lines[:10]:
        stripped = line.strip()
        if stripped.startswith("# "):
            title = stripped[2:].strip()
            break

    return {
        "title": title,
        "text": text,
        "page_count": 1,
        "success": bool(text),
        "error": "" if text else "No text extracted from HTML",
    }


def _parse_text(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="ignore")

    # For markdown, try to extract title from first H1
    title = path.stem
    if path.suffix.lower() in (".md", ".markdown"):
        for line in text.splitlines()[:20]:
            stripped = line.strip()
            if stripped.startswith("# "):
                title = stripped[2:].strip()
                break

    return {
        "title": title,
        "text": text,
        "page_count": max(1, text.count("\n") // 40),
        "success": bool(text.strip()),
        "error": "",
    }
