"""
generate_answer.py

This file generates the final regulatory answer after retrieval and reranking.

It receives two evidence tracks:
1. MDCG chunks
   - Used as explanatory guidance.
   - Helps the answer sound practical and understandable.
2. MDR chunks
   - Used as the binding legal basis.
   - Provides Article/Annex references.

The answer prompt decides how to write the final response, but this file ensures
that the LLM receives the sources separated as:
- MDCG GUIDANCE CONTEXT
- MDR LEGAL CONTEXT

This separation supports the intended answer style:
MDCG explanation first, then MDR legal basis.
"""

from pathlib import Path
from typing import List, Dict
import os

from dotenv import load_dotenv
from google import genai


ROOT_DIR = Path(__file__).resolve().parents[1]

ANSWER_PROMPT_PATH = ROOT_DIR / "prompts" / "answer_prompt.txt"

GEMINI_MODEL = "gemini-2.5-flash"


def load_prompt(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    return path.read_text(encoding="utf-8")


def format_context(chunks: List[Dict]) -> str:
    parts = []

    for chunk in chunks:
        parts.append(
            f"""
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


def generate_answer(
    question: str,
    chunks_or_mdr_chunks: List[Dict],
    mdcg_chunks: List[Dict] | None = None,
) -> str:
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY in .env file")

    client = genai.Client(api_key=api_key)

    answer_prompt = load_prompt(ANSWER_PROMPT_PATH)

    if mdcg_chunks is None:
        mdr_chunks, mdcg_chunks = split_sources(chunks_or_mdr_chunks)
    else:
        mdr_chunks = chunks_or_mdr_chunks

    mdcg_context = format_context(mdcg_chunks)
    mdr_context = format_context(mdr_chunks)

    full_prompt = f"""
{answer_prompt}

MDCG GUIDANCE CONTEXT:
{mdcg_context}

MDR LEGAL CONTEXT:
{mdr_context}

USER QUESTION:
{question}
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=full_prompt,
    )

    return response.text or ""


def main() -> None:
    print("This file is used by app.py.")


if __name__ == "__main__":
    main()