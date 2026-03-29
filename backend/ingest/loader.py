"""Load YAML knowledge base files and produce Document chunks.

YAML conventions:
  - pricing.yaml  → entries[].service + entries[].variants[]
  - services.yaml → entries[].name + entries[].description
  - faq.yaml      → entries[].question + entries[].answer
  - promotions.yaml → entries[].title + entries[].detail + entries[].valid_until
"""

import hashlib
import os
from pathlib import Path

import yaml

from backend.rag.vector_store import Document
from .chunker import chunk_text


def _make_id(branch_id: str, source_file: str, index: int) -> str:
    raw = f"{branch_id}::{source_file}::{index}"
    return hashlib.md5(raw.encode()).hexdigest()


def _yaml_entry_to_text(entry: dict, doc_type: str) -> str:
    """Convert a single YAML entry to a human-readable Thai text chunk."""
    if doc_type == "pricing":
        service = entry.get("service", "")
        lines = [f"บริการ: {service}"]
        for v in entry.get("variants", []):
            price = v.get("price_thb", "")
            note = v.get("note", "")
            line = f"  - {v.get('name', '')}: {price:,} บาท" if isinstance(price, int) else f"  - {v.get('name', '')}: {price} บาท"
            if note:
                line += f" ({note})"
            lines.append(line)
        return "\n".join(lines)

    elif doc_type == "services":
        name = entry.get("name", "")
        desc = entry.get("description", "")
        duration = entry.get("duration", "")
        suitable = entry.get("suitable_for", "")
        lines = [f"บริการ: {name}"]
        if desc:
            lines.append(f"รายละเอียด: {desc}")
        if duration:
            lines.append(f"ระยะเวลา: {duration}")
        if suitable:
            lines.append(f"เหมาะสำหรับ: {suitable}")
        return "\n".join(lines)

    elif doc_type == "faq":
        q = entry.get("question", "")
        a = entry.get("answer", "")
        return f"คำถาม: {q}\nคำตอบ: {a}"

    elif doc_type == "promotions":
        title = entry.get("title", "")
        detail = entry.get("detail", "")
        valid = entry.get("valid_until", "")
        price = entry.get("price_thb", "")
        lines = [f"โปรโมชั่น: {title}"]
        if detail:
            lines.append(f"รายละเอียด: {detail}")
        if price:
            lines.append(f"ราคา: {price:,} บาท" if isinstance(price, int) else f"ราคา: {price} บาท")
        if valid:
            lines.append(f"ถึงวันที่: {valid}")
        return "\n".join(lines)

    else:
        # Generic fallback: dump all key-value pairs
        return "\n".join(f"{k}: {v}" for k, v in entry.items())


def load_branch(branch_id: str, kb_dir: str) -> list[Document]:
    """Load all YAML files for a branch and return Document chunks."""
    branch_path = Path(kb_dir) / branch_id
    docs: list[Document] = []

    if not branch_path.exists():
        return docs

    for yaml_file in sorted(branch_path.glob("*.yaml")):
        doc_type = yaml_file.stem  # e.g. "pricing", "faq"
        with open(yaml_file, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        entries = data.get("entries", []) if isinstance(data, dict) else []
        chunk_idx = 0

        for entry in entries:
            raw_text = _yaml_entry_to_text(entry, doc_type)
            # Split into smaller chunks if text is long
            for chunk in chunk_text(raw_text):
                doc_id = _make_id(branch_id, yaml_file.name, chunk_idx)
                docs.append(
                    Document(
                        text=chunk,
                        metadata={
                            "branch_id": branch_id,
                            "doc_type": doc_type,
                            "source_file": yaml_file.name,
                        },
                        doc_id=doc_id,
                    )
                )
                chunk_idx += 1

    return docs
