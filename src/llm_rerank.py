from pathlib import Path
from typing import List, Dict
import os
import re

from dotenv import load_dotenv
from google import genai


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
    return re.findall(r"mdr_chunk_\d{4}", text)


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

RETRIEVED MDR CHUNK CANDIDATES:
{candidate_text}
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=full_prompt,
    )

    selected_ids = extract_chunk_ids(response.text or "")

    chunk_map = {chunk["chunk_id"]: chunk for chunk in retrieved_chunks}

    reranked_chunks = []
    used_ids = set()

    for chunk_id in selected_ids:
        if chunk_id in chunk_map and chunk_id not in used_ids:
            reranked_chunks.append(chunk_map[chunk_id])
            used_ids.add(chunk_id)

        if len(reranked_chunks) >= top_k:
            break

    # Fallback: if Gemini returns bad format, keep original top chunks.
    if not reranked_chunks:
        return retrieved_chunks[:top_k]

    return reranked_chunks