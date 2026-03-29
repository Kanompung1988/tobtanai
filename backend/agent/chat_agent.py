"""RAG-Agent orchestrator.

Flow:
  1. resolve branch_id from request
  2. classify intent → RETRIEVE | DIRECT
  3. if RETRIEVE: query ChromaDB, inject context into system prompt
  4. stream response from Typhoon
"""

import re
from collections.abc import Iterator

from openai import OpenAI

from backend.config import settings
from backend.models import ChatRequest
from backend.rag.embedder import Embedder
from backend.rag.retriever import format_context, retrieve
from backend.rag.vector_store import registry
from .intent_classifier import classify_intent

# -----------------------------------------------------------------------
# Branch name → branch_id mapping (matches CLINIC_CONFIG in app.py)
# -----------------------------------------------------------------------
_BRANCH_MAP = {
    "สาขาหลัก": "branch_main",
    "สาขา 2": "branch_2",
    "Full Body": "branch_fullbody",
}

_VALID_BRANCH_IDS = {"branch_main", "branch_2", "branch_fullbody"}

_CONTEXT_HEADER = (
    "\n\n---\n"
    "**ข้อมูลอ้างอิงจากฐานข้อมูลคลินิก (ใช้ข้อมูลนี้เป็นหลักในการตอบ):**\n\n"
)
_CONTEXT_FOOTER = (
    "\n\n---\n"
    "**กฎ: ตอบโดยอ้างอิงจากข้อมูลด้านบนเท่านั้น "
    "หากไม่พบข้อมูล ให้แจ้งว่าไม่ทราบและแนะนำให้ลูกค้าติดต่อคลินิกโดยตรง**"
)


def _resolve_branch_id(request: ChatRequest) -> str:
    # 1. Explicit branch_id in request
    if request.branch_id and request.branch_id in _VALID_BRANCH_IDS:
        return request.branch_id

    # 2. Parse from system_prompt
    for keyword, bid in _BRANCH_MAP.items():
        if keyword in request.system_prompt:
            return bid

    # 3. Fallback
    return "branch_main"


def _build_messages(
    system_prompt: str,
    chat_history: list[dict],
    user_message: str,
    image_base64: str | None,
) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    for turn in chat_history:
        messages.append({"role": turn["role"], "content": turn.get("content", "")})

    # Build user content (text ± image)
    if image_base64:
        content = [
            {"type": "text", "text": user_message},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
            },
        ]
    else:
        content = user_message

    messages.append({"role": "user", "content": content})
    return messages


def stream_chat(
    request: ChatRequest,
    embedder: Embedder,
) -> Iterator[str]:
    """Yield response tokens from Typhoon, with RAG if needed."""

    branch_id = _resolve_branch_id(request)
    intent = classify_intent(request.user_message)

    system_prompt = request.system_prompt

    if intent == "RETRIEVE":
        docs = retrieve(
            query=request.user_message,
            branch_id=branch_id,
            embedder=embedder,
            store_registry=registry,
            top_k=settings.top_k,
        )
        if docs:
            context = format_context(docs)
            system_prompt = system_prompt + _CONTEXT_HEADER + context + _CONTEXT_FOOTER

    messages = _build_messages(
        system_prompt=system_prompt,
        chat_history=request.chat_history,
        user_message=request.user_message,
        image_base64=request.image_base64,
    )

    client = OpenAI(
        api_key=settings.typhoon_api_key,
        base_url=settings.typhoon_base_url,
    )

    stream = client.chat.completions.create(
        model=settings.typhoon_model,
        messages=messages,
        temperature=0.3,
        max_tokens=4096,
        top_p=0.9,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
