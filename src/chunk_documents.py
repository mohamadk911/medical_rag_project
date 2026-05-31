"""
chunk_documents.py

This file builds the main RAG chunk dataset from the local regulatory sources.
It processes the MDR PDF and all downloaded MDCG PDFs in parallel as two source
families. MDR is chunked using the existing legal structure approach, keeping
articles/annex sections as separate chunks. MDCG guidance documents are kept as
one full-document chunk when they are small enough; larger MDCG PDFs are split
into large overlapping text chunks to avoid losing context. The final output is
one clean combined file: data/processed/chunks.json.
"""

from pathlib import Path
import json
import re
from typing import List, Dict

import fitz  # PyMuPDF


ROOT_DIR = Path(__file__).resolve().parents[1]

MDR_PATH = ROOT_DIR / "data" / "raw" / "MDR.pdf"
MDCG_DIR = ROOT_DIR / "data" / "raw" / "mdcg"
MDCG_CATALOG_PATH = MDCG_DIR / "mdcg_catalog_entries.json"

OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "chunks.json"

MDCG_MAX_CHARS = 45_000
MDCG_OVERLAP_CHARS = 6_000


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_pdf_pages(pdf_path: Path) -> List[Dict]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = []

    with fitz.open(pdf_path) as doc:
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()

            if text:
                pages.append({
                    "page": page_number,
                    "text": text,
                })

    return pages


def load_mdcg_catalog() -> Dict[str, Dict]:
    if not MDCG_CATALOG_PATH.exists():
        return {}

    with MDCG_CATALOG_PATH.open("r", encoding="utf-8") as f:
        entries = json.load(f)

    catalog = {}

    for entry in entries:
        local_filename = entry.get("local_filename")
        if local_filename:
            catalog[local_filename] = entry

    return catalog


def make_document_id(filename: str) -> str:
    name = Path(filename).stem.lower()
    name = re.sub(r"^\d+_", "", name)
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def chunk_mdr() -> List[Dict]:
    pages = extract_pdf_pages(MDR_PATH)

    full_text = "\n".join(page["text"] for page in pages)

    pattern = re.compile(
        r"(?m)^Article\s+\d+\s*$|^ANNEX\s+[IVXLC]+\s*$",
        re.IGNORECASE,
    )

    matches = list(pattern.finditer(full_text))
    chunks = []

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)

        raw_chunk_text = full_text[start:end]
        text = clean_text(raw_chunk_text)

        if len(text) < 100:
            continue

        first_line = raw_chunk_text.strip().splitlines()[0]
        title_match = re.match(
            r"(Article\s+\d+|ANNEX\s+[IVXLC]+)",
            first_line,
            re.IGNORECASE,
        )
        title = title_match.group(1) if title_match else "MDR section"

        page_start, page_end = find_page_range_from_text(pages, raw_chunk_text)

        chunks.append({
            "chunk_id": f"mdr_chunk_{len(chunks) + 1:04d}",
            "source_type": "MDR",
            "document_id": "mdr_2017_745",
            "document_title": "Regulation (EU) 2017/745 Medical Device Regulation",
            "reference": title,
            "source": "MDR.pdf",
            "type": "article_or_annex",
            "title": title,
            "page_start": page_start,
            "page_end": page_end,
            "text": text,
        })

    return chunks

def find_page_range_from_text(pages: List[Dict], chunk_text: str) -> tuple[int | None, int | None]:
    matched_pages = []

    chunk_sample_start = chunk_text[:300]
    chunk_sample_end = chunk_text[-300:]

    for page in pages:
        page_text = page["text"]

        if chunk_sample_start[:100] in page_text or chunk_sample_end[-100:] in page_text:
            matched_pages.append(page["page"])

    if not matched_pages:
        return None, None

    return min(matched_pages), max(matched_pages)


def split_large_text(text: str, max_chars: int, overlap_chars: int) -> List[str]:
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap_chars

    return chunks


def extract_full_pdf_text(pdf_path: Path) -> tuple[str, int]:
    pages = extract_pdf_pages(pdf_path)
    text = "\n\n".join(page["text"] for page in pages)
    return clean_text(text), len(pages)


def chunk_mdcg_documents() -> List[Dict]:
    if not MDCG_DIR.exists():
        return []

    catalog = load_mdcg_catalog()
    pdf_paths = sorted(MDCG_DIR.glob("*.pdf"))

    chunks = []

    for pdf_path in pdf_paths:
        try:
            text, page_count = extract_full_pdf_text(pdf_path)

            if not text:
                continue

            catalog_entry = catalog.get(pdf_path.name, {})
            if not catalog_entry:
                original_name = pdf_path.name.replace(".pdf", ".docx")
                catalog_entry = catalog.get(original_name, {})

            if not catalog_entry:
                original_name = pdf_path.name.replace(".pdf", ".doc")
                catalog_entry = catalog.get(original_name, {})

            document_id = make_document_id(pdf_path.name)
            document_title = (
                catalog_entry.get("reference")
                or catalog_entry.get("title")
                or pdf_path.stem
            )

            text_parts = split_large_text(
                text=text,
                max_chars=MDCG_MAX_CHARS,
                overlap_chars=MDCG_OVERLAP_CHARS,
            )

            for part_index, part_text in enumerate(text_parts, start=1):
                chunk_suffix = f"_part_{part_index}" if len(text_parts) > 1 else ""

                chunks.append({
                    "chunk_id": f"mdcg_chunk_{len(chunks) + 1:04d}",
                    "source_type": "MDCG",
                    "document_id": document_id,
                    "document_title": document_title,
                    "source": pdf_path.name,
                    "filename": pdf_path.name,
                    "url": catalog_entry.get("url"),
                    "category": catalog_entry.get("category"),
                    "type": "mdcg_guidance_document",
                    "title": f"{document_title}{chunk_suffix}",
                    "page_start": 1,
                    "page_end": page_count,
                    "text": part_text,
                })

        except Exception as error:
            print(f"Skipping {pdf_path.name}: {error}")

    return chunks


def save_chunks(chunks: List[Dict]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def main() -> None:
    print("Building MDR chunks...")
    mdr_chunks = chunk_mdr()

    print("Building MDCG chunks...")
    mdcg_chunks = chunk_mdcg_documents()

    all_chunks = mdr_chunks + mdcg_chunks

    save_chunks(all_chunks)

    print("\nDone.")
    print(f"MDR chunks: {len(mdr_chunks)}")
    print(f"MDCG chunks: {len(mdcg_chunks)}")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()