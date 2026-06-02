"""
llm_rerank.py

This file performs LLM-based reranking of retrieved regulatory chunks.

Purpose:
1. Take candidate chunks from retrieval.
2. Ask the LLM to select the most relevant chunks for the user question.
3. Support both MDR and MDCG chunk IDs:
   - mdr_chunk_0001
   - mdcg_chunk_0001
4. Keep the reranker separate from retrieval:
   - retrieve.py finds possible evidence
   - llm_rerank.py selects the best evidence
5. If the LLM gives an invalid format, the function safely falls back to the
   original top chunks instead of failing.

This file does not answer the user question.
It only returns a better ordered list of chunks.
"""

from pathlib import Path
from typing import List, Dict
import os
import re

from dotenv import load_dotenv
from google import genai
from llm_utils import generate_content_with_retries


ROOT_DIR = Path(__file__).resolve().parents[1]

RERANK_PROMPT_PATH = ROOT_DIR / "prompts" / "rerank_prompt.txt"

GEMINI_MODEL = "gemini-2.5-flash"

RERANK_OUTPUT_K = 8


def load_prompt(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    return path.read_text(encoding="utf-8")


def format_chunks_for_rerank(chunks: List[Dict]) -> str:
    parts = []

    for i, chunk in enumerate(chunks, start=1):
        text_preview = chunk.get("text", "")[:1800]

        parts.append(
            f"""
CANDIDATE {i}
CHUNK ID: {chunk.get("chunk_id")}
SOURCE TYPE: {chunk.get("source_type")}
DOCUMENT: {chunk.get("document_title")}
REFERENCE: {chunk.get("reference")}
TITLE: {chunk.get("title")}
TYPE: {chunk.get("type")}
PAGES: {chunk.get("page_start")} - {chunk.get("page_end")}
HYBRID SCORE: {chunk.get("hybrid_score")}

TEXT:
{text_preview}
"""
        )

    return "\n\n---\n\n".join(parts)


def extract_chunk_ids(text: str) -> list[str]:
    return re.findall(
        r"(?:mdr_chunk|mdcg_chunk)_\d{4}",
        text,
        flags=re.IGNORECASE,
    )


def llm_rerank_chunks(
    question: str,
    retrieved_chunks: List[Dict],
    top_k: int = RERANK_OUTPUT_K,
) -> List[Dict]:
    if not retrieved_chunks:
        return []

    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY in .env file")

    client = genai.Client(api_key=api_key)

    rerank_prompt = load_prompt(RERANK_PROMPT_PATH)
    candidate_text = format_chunks_for_rerank(retrieved_chunks)

    full_prompt = f"""
{rerank_prompt}

USER QUESTION:
{question}

RETRIEVED REGULATORY CHUNK CANDIDATES:
{candidate_text}
"""

    try:
        response = generate_content_with_retries(
            client=client,
            model=GEMINI_MODEL,
            contents=full_prompt,
        )
    except Exception as error:
        print(f"LLM reranker unavailable, using retrieval order: {error}")
        return retrieved_chunks[:top_k]

    selected_ids = extract_chunk_ids(response.text or "")

    chunk_map = {
        chunk["chunk_id"].lower(): chunk
        for chunk in retrieved_chunks
        if chunk.get("chunk_id")
    }

    reranked_chunks = []
    used_ids = set()

    for chunk_id in selected_ids:
        normalized_id = chunk_id.lower()

        if normalized_id in chunk_map and normalized_id not in used_ids:
            reranked_chunks.append(chunk_map[normalized_id])
            used_ids.add(normalized_id)

        if len(reranked_chunks) >= top_k:
            break

    if not reranked_chunks:
        return retrieved_chunks[:top_k]

    return reranked_chunks
