"""High-level retrieval interface used by the agent."""

from .embedder import Embedder
from .vector_store import Document, VectorStoreRegistry


def retrieve(
    query: str,
    branch_id: str,
    embedder: Embedder,
    store_registry: VectorStoreRegistry,
    top_k: int = 5,
) -> list[Document]:
    query_embedding = embedder.embed_query(query)
    store = store_registry.get(branch_id)
    return store.query(query_embedding, top_k=top_k)


def format_context(docs: list[Document]) -> str:
    """Format retrieved docs into a Thai-labelled context block."""
    if not docs:
        return ""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_file", "ไม่ระบุ")
        parts.append(f"[{i}] (จาก: {source})\n{doc.text}")
    return "\n\n---\n\n".join(parts)
