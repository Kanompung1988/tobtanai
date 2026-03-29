"""Split long text into overlapping chunks."""


def chunk_text(text: str, max_chars: int = 500, overlap: int = 50) -> list[str]:
    """Split text into chunks of at most max_chars with overlap."""
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        start += max_chars - overlap
    return chunks
