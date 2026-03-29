"""Per-branch ChromaDB vector store.

Each branch gets its own PersistentClient stored at:
  {chroma_persist_dir}/{branch_id}/

This physical separation guarantees zero cross-branch contamination.
"""

from dataclasses import dataclass, field

import chromadb
from chromadb.config import Settings as ChromaSettings


@dataclass
class Document:
    text: str
    metadata: dict = field(default_factory=dict)
    doc_id: str = ""


class BranchVectorStore:
    def __init__(self, branch_id: str, persist_dir: str) -> None:
        self.branch_id = branch_id
        self.client = chromadb.PersistentClient(
            path=f"{persist_dir}/{branch_id}",
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=branch_id,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, docs: list[Document], embeddings: list[list[float]]) -> None:
        if not docs:
            return
        self.collection.upsert(
            ids=[d.doc_id for d in docs],
            embeddings=embeddings,
            documents=[d.text for d in docs],
            metadatas=[d.metadata for d in docs],
        )

    def query(self, query_embedding: list[float], top_k: int = 5) -> list[Document]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )
        docs: list[Document] = []
        for text, meta in zip(
            results["documents"][0], results["metadatas"][0]
        ):
            docs.append(Document(text=text, metadata=meta))
        return docs

    def count(self) -> int:
        return self.collection.count()

    def clear(self) -> None:
        self.client.delete_collection(self.branch_id)
        self.collection = self.client.get_or_create_collection(
            name=self.branch_id,
            metadata={"hnsw:space": "cosine"},
        )


class VectorStoreRegistry:
    """Holds one BranchVectorStore per branch, initialised at startup."""

    def __init__(self) -> None:
        self._stores: dict[str, BranchVectorStore] = {}

    def init(self, branch_ids: list[str], persist_dir: str) -> None:
        for bid in branch_ids:
            self._stores[bid] = BranchVectorStore(bid, persist_dir)

    def get(self, branch_id: str) -> BranchVectorStore:
        if branch_id not in self._stores:
            raise KeyError(f"Unknown branch_id: '{branch_id}'. Available: {list(self._stores)}")
        return self._stores[branch_id]

    def all_counts(self) -> dict[str, int]:
        return {bid: store.count() for bid, store in self._stores.items()}


registry = VectorStoreRegistry()
