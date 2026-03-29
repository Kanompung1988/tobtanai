"""FastAPI application entry point.

Startup sequence:
  1. Load embedder singleton (downloads model on first run ~500MB)
  2. Init ChromaDB registry for all branches
  3. Mount API router
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.rag.embedder import Embedder
from backend.rag.vector_store import registry


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: warm up embedder and vector stores
    print("Loading embedding model…")
    Embedder.get_instance(settings.embedding_model)
    print("Initialising ChromaDB collections…")
    registry.init(settings.branch_ids, settings.chroma_persist_dir)
    print("Backend ready.")
    yield
    # Shutdown: nothing to clean up


app = FastAPI(title="Tobtan Clinic AI Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from backend.api.routes import router  # noqa: E402 — import after app creation

app.include_router(router)
