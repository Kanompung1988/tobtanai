"""FastAPI router.

Endpoints:
  GET  /api/health         → liveness check
  GET  /api/kb/status      → document counts per branch
  POST /api/chat           → non-streaming (JSON response) — compatible with current frontend
  POST /api/chat/stream    → SSE streaming (text/event-stream) — used by updated frontend
"""

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from backend.agent.chat_agent import stream_chat
from backend.models import ChatRequest, ChatResponse
from backend.rag.vector_store import registry

router = APIRouter(prefix="/api")


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/kb/status")
def kb_status():
    return {"branches": registry.all_counts()}


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest, req: Request):
    """SSE streaming endpoint — frontend receives tokens as they arrive."""
    import asyncio
    from backend.rag.embedder import Embedder
    from backend.config import settings

    embedder = Embedder.get_instance(settings.embedding_model)

    async def token_generator():
        loop = asyncio.get_event_loop()
        try:
            tokens = await loop.run_in_executor(
                None, lambda: list(stream_chat(request, embedder))
            )
            for token in tokens:
                yield f"data: {token}\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Non-streaming fallback — collects all tokens and returns full JSON."""
    from backend.rag.embedder import Embedder
    from backend.config import settings

    embedder = Embedder.get_instance(settings.embedding_model)
    tokens = list(stream_chat(request, embedder))
    return ChatResponse(response="".join(tokens))
