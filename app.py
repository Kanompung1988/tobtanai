"""
Tobtan Clinic AI — Streamlit App
---------------------------------
Runs fully standalone — calls backend modules directly (no HTTP server needed).
"""

import base64

import streamlit as st

from backend.config import settings
from backend.models import ChatRequest
from backend.rag.embedder import Embedder
from backend.rag.vector_store import registry
from backend.agent.chat_agent import stream_chat

# ---------------------------------------------------------------------------
# One-time startup: load heavy resources once and cache across reruns
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="กำลังโหลดโมเดล AI...")
def _load_resources():
    embedder = Embedder.get_instance(settings.embedding_model)
    registry.init(settings.branch_ids, settings.chroma_persist_dir)
    return embedder

embedder = _load_resources()

CLINIC_CONFIG: dict[str, dict] = {
    "Tobtan Clinic (สาขาหลัก)": {
        "branch_id": "branch_main",
        "services": [
            "Facial treatments (ดูแลผิวหน้า)",
            "Botox",
            "Dermal Fillers (ฟิลเลอร์)",
            "Laser treatments (เลเซอร์)",
        ],
    },
    "Tobtan Clinic (สาขา 2)": {
        "branch_id": "branch_2",
        "services": [
            "Facial treatments (ดูแลผิวหน้า)",
            "Botox",
            "Dermal Fillers (ฟิลเลอร์)",
            "Breast Augmentation (เสริมหน้าอก)",
            "Liposuction (ดูดไขมัน)",
        ],
    },
    "Tobtan Clinic (Full Body)": {
        "branch_id": "branch_fullbody",
        "services": [
            "Facial treatments (ดูแลผิวหน้า)",
            "Botox",
            "Dermal Fillers (ฟิลเลอร์)",
            "Breast Augmentation (เสริมหน้าอก)",
            "Liposuction (ดูดไขมัน)",
            "Body contouring (เสริมรูปร่าง)",
            "Skin rejuvenation (ฟื้นฟูผิว)",
        ],
    },
}

# ---------------------------------------------------------------------------
# Helper: Build system prompt
# ---------------------------------------------------------------------------


def build_system_prompt(clinic_name: str) -> str:
    services = CLINIC_CONFIG[clinic_name]["services"]
    services_list = "\n".join(f"  - {s}" for s in services)
    return (
        f"คุณคือที่ปรึกษา AI ของ {clinic_name} คลินิกความงามมืออาชีพ "
        f"คุณมีความรู้ด้านความงามและการดูแลตัวเอง มีความเป็นมืออาชีพ และให้บริการด้วยความสุภาพ\n\n"
        f"**ภาษา:** ตอบเป็นภาษาไทยเป็นหลักเสมอ ใช้ภาษาสุภาพ เป็นทางการปานกลาง "
        f"หากลูกค้าพิมพ์ภาษาอังกฤษ ให้ตอบเป็นภาษาอังกฤษแทน\n\n"
        f"**บริการที่ {clinic_name} ให้บริการได้:**\n"
        f"{services_list}\n\n"
        f"**กฎสำคัญ:**\n"
        f"1. ตอบคำถามและให้คำปรึกษาเฉพาะบริการที่อยู่ในรายการข้างต้นเท่านั้น\n"
        f"2. หากลูกค้าถามเกี่ยวกับบริการที่ไม่อยู่ในรายการ ให้แจ้งอย่างสุภาพว่า "
        f"{clinic_name} ไม่ได้ให้บริการดังกล่าว\n"
        f"3. หากมีการส่งภาพมา ให้วิเคราะห์ภาพเฉพาะในบริบทของบริการที่คลินิกมีเท่านั้น\n"
        f"4. ห้ามให้ข้อมูล ราคา หรือคำแนะนำสำหรับบริการนอกเหนือรายการ\n"
        f"5. รักษาบทบาทที่ใส่ใจ เป็นมืออาชีพ และช่วยเหลือลูกค้าอยู่เสมอ"
    )


# ---------------------------------------------------------------------------
# Helper: Call backend directly, yield tokens
# ---------------------------------------------------------------------------


def stream_chat_api(
    system_prompt: str,
    chat_history: list[dict],
    user_message: str,
    image_base64: str | None,
    branch_id: str,
):
    """Generator that yields text tokens by calling backend directly."""
    request = ChatRequest(
        system_prompt=system_prompt,
        chat_history=chat_history,
        user_message=user_message,
        image_base64=image_base64,
        branch_id=branch_id,
    )
    yield from stream_chat(request, embedder)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Tobtan Clinic AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state["chat_history"]: list[dict] = []

if "current_clinic" not in st.session_state:
    st.session_state["current_clinic"] = list(CLINIC_CONFIG.keys())[0]

if "pending_image_b64" not in st.session_state:
    st.session_state["pending_image_b64"]: str | None = None

if "pending_image_name" not in st.session_state:
    st.session_state["pending_image_name"]: str | None = None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ✨ Tobtan Clinic AI")
    st.caption("ที่ปรึกษาด้านความงามอัจฉริยะ")
    st.divider()

    selected_clinic = st.selectbox(
        "เลือกสาขา",
        options=list(CLINIC_CONFIG.keys()),
        index=list(CLINIC_CONFIG.keys()).index(st.session_state["current_clinic"]),
        key="clinic_selector_widget",
    )

    if selected_clinic != st.session_state["current_clinic"]:
        st.session_state["chat_history"] = []
        st.session_state["pending_image_b64"] = None
        st.session_state["pending_image_name"] = None
        st.session_state["current_clinic"] = selected_clinic
        st.rerun()

    st.divider()

    with st.expander(
        f"บริการที่ให้ ({len(CLINIC_CONFIG[selected_clinic]['services'])} รายการ)",
        expanded=True,
    ):
        for svc in CLINIC_CONFIG[selected_clinic]["services"]:
            st.markdown(f"- {svc}")

    st.divider()

    if st.button("ล้างประวัติการสนทนา", use_container_width=True, type="secondary"):
        st.session_state["chat_history"] = []
        st.session_state["pending_image_b64"] = None
        st.session_state["pending_image_name"] = None
        st.rerun()

    st.divider()
    st.caption("🤖 Powered by Typhoon AI + ChromaDB RAG")

# ---------------------------------------------------------------------------
# Main header
# ---------------------------------------------------------------------------

st.title(f"💬 {st.session_state['current_clinic']}")
st.caption("สอบถามข้อมูลบริการ ราคา หรือขอคำปรึกษาด้านความงามได้เลยค่ะ")
st.divider()

# ---------------------------------------------------------------------------
# Chat history display
# ---------------------------------------------------------------------------

chat_container = st.container()

with chat_container:
    if not st.session_state["chat_history"]:
        st.info(
            "ยังไม่มีประวัติการสนทนา — เริ่มต้นด้วยการพิมพ์คำถามด้านล่างได้เลยค่ะ",
            icon="💡",
        )
    else:
        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                if msg.get("image_b64"):
                    st.image(base64.b64decode(msg["image_b64"]), width=320)
                st.markdown(msg["content"])

    if st.session_state["pending_image_b64"]:
        with st.chat_message("user"):
            st.image(
                base64.b64decode(st.session_state["pending_image_b64"]),
                caption=f"รูปที่เลือก: {st.session_state['pending_image_name']}",
                width=320,
            )
            st.caption("รูปนี้จะถูกส่งพร้อมข้อความถัดไปของคุณ")

# ---------------------------------------------------------------------------
# Image uploader
# ---------------------------------------------------------------------------

uploaded_file = st.file_uploader(
    "แนบรูปภาพ (ไม่บังคับ)",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
    key="image_uploader",
    help="รองรับ JPG และ PNG",
)

if uploaded_file is not None:
    if uploaded_file.name != st.session_state["pending_image_name"]:
        image_bytes = uploaded_file.read()
        st.session_state["pending_image_b64"] = base64.b64encode(image_bytes).decode("utf-8")
        st.session_state["pending_image_name"] = uploaded_file.name
        st.rerun()
else:
    if st.session_state["pending_image_b64"] is not None:
        st.session_state["pending_image_b64"] = None
        st.session_state["pending_image_name"] = None

# ---------------------------------------------------------------------------
# Chat input + streaming send
# ---------------------------------------------------------------------------

user_input = st.chat_input("พิมพ์ข้อความของคุณที่นี่...")

if user_input:
    current_clinic = st.session_state["current_clinic"]
    branch_id = CLINIC_CONFIG[current_clinic]["branch_id"]
    system_prompt = build_system_prompt(current_clinic)
    history_snapshot = list(st.session_state["chat_history"])
    image_to_send = st.session_state["pending_image_b64"]

    # Append user message
    st.session_state["chat_history"].append(
        {"role": "user", "content": user_input, "image_b64": image_to_send}
    )
    st.session_state["pending_image_b64"] = None
    st.session_state["pending_image_name"] = None

    # Show user message immediately
    with chat_container:
        with st.chat_message("user"):
            if image_to_send:
                st.image(base64.b64decode(image_to_send), width=320)
            st.markdown(user_input)

    # Stream assistant response
    assistant_reply = ""
    error_message = None

    with chat_container:
        with st.chat_message("assistant"):
            stream_placeholder = st.empty()
            try:
                for token in stream_chat_api(
                    system_prompt=system_prompt,
                    chat_history=history_snapshot,
                    user_message=user_input,
                    image_base64=image_to_send,
                    branch_id=branch_id,
                ):
                    assistant_reply += token
                    stream_placeholder.markdown(assistant_reply + "▌")
                stream_placeholder.markdown(assistant_reply)
            except RuntimeError as exc:
                error_message = str(exc)
                stream_placeholder.markdown(
                    f"⚠️ **เกิดข้อผิดพลาด:**\n\n{error_message}\n\n"
                    "_กรุณาลองใหม่อีกครั้ง หรือตรวจสอบการเชื่อมต่อ Backend_"
                )

    # Persist assistant reply to history
    st.session_state["chat_history"].append(
        {
            "role": "assistant",
            "content": assistant_reply if not error_message else (
                f"⚠️ **เกิดข้อผิดพลาด:**\n\n{error_message}\n\n"
                "_กรุณาลองใหม่อีกครั้ง หรือตรวจสอบการเชื่อมต่อ Backend_"
            ),
            "image_b64": None,
        }
    )
