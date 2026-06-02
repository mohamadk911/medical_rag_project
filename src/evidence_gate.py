"""
evidence_gate.py

This file checks whether the retrieved MDR/MDCG evidence is sufficient before
the answer generator is allowed to create a final response.

Purpose:
1. Prevent answers when the retrieved evidence is weak or irrelevant.
2. Detect when the user asks about a specific device but did not provide enough
   device information.
3. Support the dual-track source design:
   - MDCG guidance evidence
   - MDR legal evidence

Input:
- user question
- retrieved chunks, either as one combined list or separate MDR/MDCG lists

Output:
- passed=True if answerable
- passed=False if evidence is insufficient or more device information is needed
"""

from pathlib import Path
from typing import List, Dict
import os

from dotenv import load_dotenv
from google import genai
from llm_utils import generate_content_with_retries


ROOT_DIR = Path(__file__).resolve().parents[1]

EVIDENCE_PROMPT_PATH = ROOT_DIR / "prompts" / "evidence_prompt.txt"

GEMINI_MODEL = "gemini-2.5-flash"

INSUFFICIENT_MESSAGE = (
    "I could not find sufficient evidence in the available MDR and MDCG "
    "sources to answer this question."
)


def load_prompt(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    return path.read_text(encoding="utf-8")


def format_context(chunks: List[Dict], label: str) -> str:
    parts = []

    for chunk in chunks:
        parts.append(
            f"""
SOURCE GROUP: {label}
SOURCE ID: {chunk.get("chunk_id")}
SOURCE TYPE: {chunk.get("source_type")}
DOCUMENT: {chunk.get("document_title")}
REFERENCE: {chunk.get("reference")}
TITLE: {chunk.get("title")}
TYPE: {chunk.get("type")}
PAGES: {chunk.get("page_start")} - {chunk.get("page_end")}

TEXT:
{chunk.get("text")}
"""
        )

    return "\n\n---\n\n".join(parts)


def split_sources(chunks: List[Dict]) -> tuple[List[Dict], List[Dict]]:
    mdr_chunks = [
        chunk for chunk in chunks
        if chunk.get("source_type") == "MDR"
    ]

    mdcg_chunks = [
        chunk for chunk in chunks
        if chunk.get("source_type") == "MDCG"
    ]

    return mdr_chunks, mdcg_chunks


def get_first_label(text: str) -> str:
    if not text.strip():
        return "INSUFFICIENT_EVIDENCE"

    first_line = text.strip().splitlines()[0].strip().upper()

    if "NEEDS_MORE_DEVICE_INFO" in first_line:
        return "NEEDS_MORE_DEVICE_INFO"

    if "INSUFFICIENT_EVIDENCE" in first_line:
        return "INSUFFICIENT_EVIDENCE"

    if "ANSWERABLE" in first_line:
        return "ANSWERABLE"

    return "INSUFFICIENT_EVIDENCE"


def check_evidence_gate(
    question: str,
    chunks_or_mdr_chunks: List[Dict],
    mdcg_chunks: List[Dict] | None = None,
) -> Dict:
    if mdcg_chunks is None:
        mdr_chunks, mdcg_chunks = split_sources(chunks_or_mdr_chunks)
    else:
        mdr_chunks = chunks_or_mdr_chunks

    all_chunks = mdr_chunks + mdcg_chunks

    if not all_chunks:
        return {
            "passed": False,
            "reason": "insufficient_evidence",
            "message": INSUFFICIENT_MESSAGE,
        }

    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY in .env file")

    client = genai.Client(api_key=api_key)

    evidence_prompt = load_prompt(EVIDENCE_PROMPT_PATH)

    mdcg_context = format_context(mdcg_chunks, "MDCG_GUIDANCE")
    mdr_context = format_context(mdr_chunks, "MDR_LEGAL_BASIS")

    full_prompt = f"""
{evidence_prompt}

USER QUESTION:
{question}

MDCG GUIDANCE SOURCES:
{mdcg_context}

MDR LEGAL SOURCES:
{mdr_context}
"""

    response = generate_content_with_retries(
        client=client,
        model=GEMINI_MODEL,
        contents=full_prompt,
    )

    result_text = (response.text or "").strip()
    label = get_first_label(result_text)

    if label == "ANSWERABLE":
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
        "reason": "insufficient_evidence",
        "message": INSUFFICIENT_MESSAGE + "\n\n" + result_text,
    }
