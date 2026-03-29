"""Intent classifier: decides whether to RETRIEVE from KB or answer DIRECTLY.

Returns "RETRIEVE" or "DIRECT".
Uses a fast, constrained LLM call (max_tokens=5, temperature=0).
"""

from openai import OpenAI

from backend.config import settings

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=settings.typhoon_api_key,
            base_url=settings.typhoon_base_url,
        )
    return _client


_CLASSIFY_SYSTEM = (
    "คุณเป็น classifier ที่ตอบได้เพียง RETRIEVE หรือ DIRECT เท่านั้น ห้ามตอบอื่น"
)

_CLASSIFY_PROMPT = """\
จำแนกข้อความต่อไปนี้ว่าต้องการค้นหาข้อมูลจากฐานข้อมูลคลินิกหรือไม่

ตอบ "RETRIEVE" ถ้าผู้ใช้ถามเกี่ยวกับ:
- ราคา / ค่าใช้จ่าย / แพ็กเกจ
- รายละเอียดบริการ / ขั้นตอนการรักษา
- โปรโมชั่น / ส่วนลด
- ข้อควรระวัง / การดูแลหลังรักษา
- เวลาทำการ / ที่ตั้ง / การนัดหมาย
- คำถามเฉพาะเกี่ยวกับคลินิก

ตอบ "DIRECT" ถ้าเป็น:
- การทักทาย / ลาจาก / ขอบคุณ
- สนทนาทั่วไป ไม่เกี่ยวกับบริการคลินิก
- คำถามที่ตอบได้จากความรู้ทั่วไปโดยไม่ต้องข้อมูลเฉพาะ

ข้อความ: {user_message}

ตอบเพียง RETRIEVE หรือ DIRECT"""


def classify_intent(user_message: str) -> str:
    """Returns 'RETRIEVE' or 'DIRECT'."""
    prompt = _CLASSIFY_PROMPT.format(user_message=user_message)
    try:
        response = _get_client().chat.completions.create(
            model=settings.typhoon_model,
            messages=[
                {"role": "system", "content": _CLASSIFY_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=5,
        )
        result = response.choices[0].message.content or ""
        result = result.strip().upper()
        return "RETRIEVE" if "RETRIEVE" in result else "DIRECT"
    except Exception:
        # Fail safe: default to RETRIEVE to avoid missing information
        return "RETRIEVE"
