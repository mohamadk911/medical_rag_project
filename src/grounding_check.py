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


def check_grounding(question: str, answer: str, retrieved_chunks: List[Dict]) -> str:
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY in .env file")

    client = genai.Client(api_key=api_key)

    grounding_prompt = load_prompt(GROUNDING_PROMPT_PATH)
    context = format_context(retrieved_chunks)

    full_prompt = f"""
{grounding_prompt}

USER QUESTION:
{question}

RETRIEVED MDR CONTEXT:
{context}

GENERATED ANSWER:
{answer}
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=full_prompt,
    )

    return response.text


def grounding_is_supported(grounding_result: str) -> bool:
    first_line = grounding_result.strip().splitlines()[0].upper()
    return first_line.startswith("SUPPORTED")


def main() -> None:
    print("This file is used by app.py later.")


if __name__ == "__main__":
    main()