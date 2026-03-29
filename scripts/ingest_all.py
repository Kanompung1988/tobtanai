#!/usr/bin/env python3
"""Build (or rebuild) ChromaDB knowledge base collections for all branches.

Usage:
  python scripts/ingest_all.py                    # ingest all branches
  python scripts/ingest_all.py --branch branch_main  # ingest one branch
  python scripts/ingest_all.py --clear            # clear before ingesting
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import settings
from backend.ingest.loader import load_branch
from backend.rag.embedder import Embedder
from backend.rag.vector_store import BranchVectorStore


def ingest_branch(branch_id: str, embedder: Embedder, clear: bool = False) -> None:
    print(f"\n[{branch_id}] Loading documents…")
    docs = load_branch(branch_id, settings.kb_dir)

    if not docs:
        print(f"[{branch_id}] No documents found — skipping.")
        return

    store = BranchVectorStore(branch_id, settings.chroma_persist_dir)

    if clear:
        print(f"[{branch_id}] Clearing existing collection…")
        store.clear()

    print(f"[{branch_id}] Embedding {len(docs)} chunks…")
    texts = [d.text for d in docs]
    embeddings = embedder.embed_passages(texts)

    store.add_documents(docs, embeddings)
    print(f"[{branch_id}] Done. Collection now has {store.count()} documents.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest knowledge base into ChromaDB")
    parser.add_argument("--branch", help="Ingest a single branch (default: all)")
    parser.add_argument("--clear", action="store_true", help="Clear collection before ingesting")
    args = parser.parse_args()

    print("Loading embedding model (first run may download ~500MB)…")
    embedder = Embedder.get_instance(settings.embedding_model)
    print("Embedding model ready.")

    branches = [args.branch] if args.branch else settings.branch_ids

    for branch_id in branches:
        if branch_id not in settings.branch_ids:
            print(f"Unknown branch_id: '{branch_id}'. Valid: {settings.branch_ids}")
            sys.exit(1)
        ingest_branch(branch_id, embedder, clear=args.clear)

    print("\nAll done! Start the backend with:")
    print("  uvicorn backend.main:app --port 8000 --reload")


if __name__ == "__main__":
    main()
