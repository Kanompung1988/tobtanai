from pydantic import BaseModel


class ChatRequest(BaseModel):
    system_prompt: str
    chat_history: list[dict]
    user_message: str
    image_base64: str | None = None
    branch_id: str | None = None  # optional; fallback = parsed from system_prompt


class ChatResponse(BaseModel):
    response: str
    retrieved: bool = False
    sources: list[str] = []
