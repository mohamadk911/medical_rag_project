"""
Source-based fallback answer formatting.

This is used when the retrieval pipeline succeeds but the external LLM quota is
exhausted. It does not invent a final regulatory interpretation; it shows the
best retrieved MDCG and MDR evidence so the CLI still produces useful output.
"""

from typing import Dict, List
import re


MAX_SNIPPET_CHARS = 900
MAX_MDCG_SOURCES = 4
MAX_MDR_SOURCES = 3


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def source_label(chunk: Dict) -> str:
    source_type = chunk.get("source_type")
    reference = chunk.get("reference")
    title = chunk.get("title") or chunk.get("document_title")

    if source_type == "MDR":
        return reference or title or "MDR source"

    return reference or title or chunk.get("document_title") or "MDCG source"


def format_source_block(chunk: Dict) -> str:
    snippet = clean_text(chunk.get("text", ""))[:MAX_SNIPPET_CHARS]
    pages = f"pages {chunk.get('page_start')}-{chunk.get('page_end')}"

    return (
        f"- {source_label(chunk)} ({chunk.get('document_title')}, {pages})\n"
        f"  {snippet}"
    )


def split_sources(chunks: List[Dict]) -> tuple[List[Dict], List[Dict]]:
    mdcg_chunks = [
        chunk for chunk in chunks
        if chunk.get("source_type") == "MDCG"
    ]
    mdr_chunks = [
        chunk for chunk in chunks
        if chunk.get("source_type") == "MDR"
    ]

    return mdcg_chunks, mdr_chunks


def generate_source_fallback_answer(question: str, chunks: List[Dict]) -> str:
    mdcg_chunks, mdr_chunks = split_sources(chunks)

    mdcg_blocks = [
        format_source_block(chunk)
        for chunk in mdcg_chunks[:MAX_MDCG_SOURCES]
    ]
    mdr_blocks = [
        format_source_block(chunk)
        for chunk in mdr_chunks[:MAX_MDR_SOURCES]
    ]

    sections = [
        (
            "Gemini returned a rate-limit or quota error, so I cannot generate "
            "the polished final answer right now."
        ),
        "Retrieval did succeed. Here are the strongest retrieved sources for your question:",
        f"Question: {question}",
    ]

    if mdcg_blocks:
        sections.append("MDCG guidance evidence:\n" + "\n\n".join(mdcg_blocks))

    if mdr_blocks:
        sections.append("MDR legal-basis evidence:\n" + "\n\n".join(mdr_blocks))

    sections.append(
        (
            "Try again after a short wait, or reduce top_k/context size if this "
            "happens repeatedly."
        )
    )

    return "\n\n".join(sections)
