"""
grounding_check.py

This file verifies whether the generated answer is supported by the retrieved
MDR and MDCG evidence.

Purpose:
1. Prevent hallucinated or unsupported regulatory statements.
2. Check the generated answer against the actual retrieved source text.
3. Support dual-track evidence:
   - MDCG guidance context
   - MDR legal basis context

Input:
- user question
- generated answer
- retrieved chunks, either as:
  a) one combined list, or
  b) separate MDR chunks and MDCG chunks

Output:
- SUPPORTED
- PARTIALLY_SUPPORTED
- UNSUPPORTED

The app only accepts the final answer when the grounding result starts with
SUPPORTED.
"""

from pathlib import Path
from typing import List, Dict
import os

from dotenv import load_dotenv
from google import genai


ROOT_DIR = Path(__file__).resolve().parents[1]

GROUNDING_PROMPT_PATH = ROOT_DIR / "prompts" / "grounding_prompt.txt"

GEMINI_MODEL = "gemini-2.5-flash"


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
SOURCE TYPE: {chunk.get("source_type")}
DOCUMENT: {chunk.get("document_title")}
REFERENCE: {chunk.get("reference")}
CHUNK ID: {chunk.get("chunk_id")}
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


def check_grounding(
    question: str,
    answer: str,
    chunks_or_mdr_chunks: List[Dict],
    mdcg_chunks: List[Dict] | None = None,
) -> str:
    if mdcg_chunks is None:
        mdr_chunks, mdcg_chunks = split_sources(chunks_or_mdr_chunks)
    else:
        mdr_chunks = chunks_or_mdr_chunks

    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY in .env file")

    client = genai.Client(api_key=api_key)

    grounding_prompt = load_prompt(GROUNDING_PROMPT_PATH)

    mdcg_context = format_context(mdcg_chunks, "MDCG_GUIDANCE")
    mdr_context = format_context(mdr_chunks, "MDR_LEGAL_BASIS")

    full_prompt = f"""
{grounding_prompt}

USER QUESTION:
{question}

MDCG GUIDANCE CONTEXT:
{mdcg_context}

MDR LEGAL CONTEXT:
{mdr_context}

GENERATED ANSWER:
{answer}
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=full_prompt,
    )

    return response.text or ""


def grounding_is_supported(grounding_result: str) -> bool:
    if not grounding_result.strip():
        return False

    first_line = grounding_result.strip().splitlines()[0].upper()
    return first_line.startswith("SUPPORTED")


def main() -> None:
    print("This file is used by app.py.")


if __name__ == "__main__":
    main()