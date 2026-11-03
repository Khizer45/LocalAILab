"""Local document parsing (PDF, TXT/Markdown).

PDF text extraction uses **pypdf** (BSD-3-Clause) by default. PyMuPDF gives slightly better text layout but is
licensed under the GNU AGPL-3.0 (or a commercial licence from Artifex) – distributing or offering Local AI Lab with
it obliges you to follow the AGPL. It is therefore optional: install it yourself and set ``PDF_PARSER=pymupdf``
only if your use complies with its licence. See docs/LEGAL_AND_LICENSING.md.
"""

from __future__ import annotations

from pathlib import Path

from app.core.errors import ValidationFailed

ALLOWED_EXTENSIONS = {".pdf": "pdf", ".txt": "txt", ".md": "md", ".markdown": "md"}


def detect_type(filename: str, head: bytes) -> str:
    """Validate extension *and* content signature. Returns pdf | txt | md."""
    ext = Path(filename).suffix.lower()
    file_type = ALLOWED_EXTENSIONS.get(ext)
    if file_type is None:
        raise ValidationFailed(
            f"Unsupported file type '{ext or 'none'}'. Allowed: PDF, TXT, Markdown.", code="unsupported_file_type"
        )
    if file_type == "pdf":
        if not head.lstrip()[:5].startswith(b"%PDF-"):
            raise ValidationFailed("File has a .pdf extension but is not a valid PDF.", code="invalid_file_content")
    elif b"\x00" in head and head[:2] not in (b"\xff\xfe", b"\xfe\xff"):
        raise ValidationFailed("Text file appears to be binary.", code="invalid_file_content")
    return file_type


def decode_text(data: bytes) -> str:
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16", errors="replace")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def parse_document(path: Path, file_type: str, *, max_pages: int = 500,
                   pdf_parser: str = "pypdf") -> list[tuple[int | None, str]]:
    """Return a list of (page_number, text). page_number is None for text files."""
    if file_type == "pdf":
        if pdf_parser == "pymupdf":
            try:
                return _parse_pdf_pymupdf(path, max_pages)
            except ImportError:
                pass  # not installed – fall back to the permissively licensed parser
        return _parse_pdf(path, max_pages)
    text = decode_text(path.read_bytes())
    return [(None, text)]


def _parse_pdf(path: Path, max_pages: int) -> list[tuple[int | None, str]]:
    import logging

    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    logging.getLogger("pypdf").setLevel(logging.ERROR)  # malformed-but-readable PDFs are common
    try:
        reader = PdfReader(str(path), strict=False)
        if reader.is_encrypted:
            try:
                ok = reader.decrypt("")
            except Exception:  # noqa: BLE001 - unsupported encryption
                ok = 0
            if not ok:
                raise ValidationFailed("Encrypted / password-protected PDFs are not supported.", code="encrypted_pdf")
        n = len(reader.pages)
    except ValidationFailed:
        raise
    except (PdfReadError, ValueError, KeyError, TypeError, OSError) as exc:
        raise ValidationFailed(f"Could not open PDF: {type(exc).__name__}", code="malformed_document") from exc
    if n > max_pages:
        raise ValidationFailed(f"PDF has {n} pages; the limit is {max_pages}.", code="too_many_pages")
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:  # noqa: BLE001 - one broken page should not lose the document
            text = ""
        pages.append((i + 1, text))
    return pages


def _parse_pdf_pymupdf(path: Path, max_pages: int) -> list[tuple[int | None, str]]:
    import pymupdf  # optional, AGPL-3.0 – see the module docstring

    try:
        doc = pymupdf.open(path)
    except Exception as exc:  # noqa: BLE001
        raise ValidationFailed(f"Could not open PDF: {type(exc).__name__}", code="malformed_document") from exc
    try:
        if doc.needs_pass or doc.is_encrypted:
            raise ValidationFailed("Encrypted / password-protected PDFs are not supported.", code="encrypted_pdf")
        if doc.page_count > max_pages:
            raise ValidationFailed(f"PDF has {doc.page_count} pages; the limit is {max_pages}.", code="too_many_pages")
        pages = []
        for i, page in enumerate(doc):
            pages.append((i + 1, page.get_text("text") or ""))
        return pages
    finally:
        doc.close()
