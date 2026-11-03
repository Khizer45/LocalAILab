"""Page-aware text chunking.

Chunks never cross page boundaries, so every chunk can be cited with the page
it came from. Paragraphs are packed into chunks of roughly `chunk_size`
characters; long paragraphs are split on sentence boundaries and, as a last
resort, on whitespace. Consecutive chunks of the same page share `overlap`
characters of context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_WS = re.compile(r"[ \t\f\v]+")


@dataclass
class TextChunk:
    page: int | None
    text: str


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or ord(ch) >= 32)
    lines = [_WS.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # join hyphenated line breaks produced by PDF extraction: "infor-\nmation"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    return text.strip()


def _split_long(piece: str, size: int) -> list[str]:
    if len(piece) <= size:
        return [piece]
    out: list[str] = []
    buf = ""
    for sent in _SENTENCE_END.split(piece):
        if len(sent) > size:  # hard split very long "sentences" on whitespace
            words = sent.split(" ")
            for w in words:
                if len(buf) + len(w) + 1 > size and buf:
                    out.append(buf)
                    buf = ""
                buf = f"{buf} {w}".strip()
            continue
        if len(buf) + len(sent) + 1 > size and buf:
            out.append(buf)
            buf = ""
        buf = f"{buf} {sent}".strip()
    if buf:
        out.append(buf)
    return out


def _tail(text: str, overlap: int) -> str:
    if overlap <= 0 or len(text) <= overlap:
        return text if overlap > 0 else ""
    cut = text[-overlap:]
    space = cut.find(" ")
    return cut[space + 1:] if 0 <= space < len(cut) - 1 else cut


def chunk_pages(pages: list[tuple[int | None, str]], chunk_size: int = 900, overlap: int = 150) -> list[TextChunk]:
    chunk_size = max(200, chunk_size)
    overlap = max(0, min(overlap, chunk_size // 2))
    chunks: list[TextChunk] = []
    for page, raw in pages:
        text = normalize(raw)
        if not text:
            continue
        paragraphs = [p.replace("\n", " ").strip() for p in text.split("\n\n") if p.strip()]
        pieces: list[str] = []
        for p in paragraphs:
            pieces.extend(_split_long(p, chunk_size - overlap))
        buf = ""
        page_chunks: list[str] = []
        for piece in pieces:
            if buf and len(buf) + len(piece) + 2 > chunk_size:
                page_chunks.append(buf)
                buf = (_tail(buf, overlap) + " " + piece).strip() if overlap else piece
            else:
                buf = f"{buf}\n\n{piece}".strip() if buf else piece
        if buf:
            # merge a tiny trailing piece into the previous chunk of this page
            if page_chunks and len(buf) < chunk_size * 0.2 and len(page_chunks[-1]) + len(buf) <= chunk_size * 1.2:
                page_chunks[-1] = page_chunks[-1] + "\n\n" + buf
            else:
                page_chunks.append(buf)
        chunks.extend(TextChunk(page=page, text=c) for c in page_chunks)
    return chunks
