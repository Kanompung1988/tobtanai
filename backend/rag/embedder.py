"""Singleton embedder using fastembed (ONNX-based, fast, no torch dependency).

Uses paraphrase-multilingual-MiniLM-L12-v2 by default (384 dim, supports Thai).
"""

from fastembed import TextEmbedding


class Embedder:
    _instance: "Embedder | None" = None

    def __init__(self, model_name: str) -> None:
        self.model = TextEmbedding(model_name)

    @classmethod
    def get_instance(cls, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") -> "Embedder":
        if cls._instance is None:
            cls._instance = cls(model_name)
        return cls._instance

    def embed_query(self, text: str) -> list[float]:
        return list(self.model.embed([text]))[0].tolist()

    def embed_passage(self, text: str) -> list[float]:
        return list(self.model.embed([text]))[0].tolist()

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self.model.embed(texts)]
