from pathlib import Path
from typing import List, Dict
import os

from dotenv import load_dotenv
from google import genai


ROOT_DIR = Path(__file__).resolve().parents[1]

EVIDENCE_PROMPT_PATH = ROOT_DIR / "prompts" / "evidence_prompt.txt"

GEMINI_MODEL = "gemini-2.5-flash"


NOT_ANSWERABLE_MESSAGE = (
    "I am an educational MDR assistant. I could not find enough relevant "
    "information in the provided MDR PDF to answer this specific question."
)


def load_prompt(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    return path.read_text(encoding="utf-8")


def format_context(chunks: List[Dict]) -> str:
    parts = []

    for chunk in chunks:
        parts.append(
            f"""
CHUNK ID: {chunk.get("chunk_id")}
TITLE: {chunk.get("title")}
TYPE: {chunk.get("type")}
PAGES: {chunk.get("page_start")} - {chunk.get("page_end")}

TEXT:
{chunk.get("text")}
"""
        )

    return "\n\n---\n\n".join(parts)


def get_first_label(text: str) -> str:
    if not text.strip():
        return "NOT_ANSWERABLE_FROM_MDR"

    first_line = text.strip().splitlines()[0].strip().upper()

    if "ANSWERABLE_FROM_MDR" in first_line:
        return "ANSWERABLE_FROM_MDR"

    if "NEEDS_MORE_DEVICE_INFO" in first_line:
        return "NEEDS_MORE_DEVICE_INFO"

    if "NOT_ANSWERABLE_FROM_MDR" in first_line:
        return "NOT_ANSWERABLE_FROM_MDR"

    return "NOT_ANSWERABLE_FROM_MDR"


def check_evidence_gate(question: str, retrieved_chunks: List[Dict]) -> Dict:
    if not retrieved_chunks:
        return {
            "passed": False,
            "reason": "not_answerable",
            "message": NOT_ANSWERABLE_MESSAGE,
        }

    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY in .env file")

    client = genai.Client(api_key=api_key)

    evidence_prompt = load_prompt(EVIDENCE_PROMPT_PATH)
    context = format_context(retrieved_chunks)

    full_prompt = f"""
{evidence_prompt}

USER QUESTION:
{question}

RETRIEVED MDR CHUNKS:
{context}
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=full_prompt,
    )

    result_text = (response.text or "").strip()
    label = get_first_label(result_text)

    if label == "ANSWERABLE_FROM_MDR":
        return {
            "passed": True,
            "reason": "answerable",
            "message": result_text,
        }

    if label == "NEEDS_MORE_DEVICE_INFO":
        return {
            "passed": False,
            "reason": "needs_more_device_info",
            "message": result_text,
        }

    return {
        "passed": False,
        "reason": "not_answerable",
        "message": NOT_ANSWERABLE_MESSAGE + "\n\n" + result_text,
    }