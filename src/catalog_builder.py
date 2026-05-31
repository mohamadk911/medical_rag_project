"""
catalog_builder.py

This file builds the document-level catalog used for Stage 1 retrieval.
Instead of searching inside all MDR/MDCG chunks immediately, the RAG first
searches this small catalog to decide which regulatory documents are likely
relevant. The catalog contains one MDR entry and one entry per MDCG document,
enriched with metadata from the MDCG catalog page and a short first-page preview
from each PDF.
"""

from pathlib import Path
import json
import re
from typing import List, Dict

import fitz  # PyMuPDF


ROOT_DIR = Path(__file__).resolve().parents[1]

MDR_PATH = ROOT_DIR / "data" / "raw" / "MDR.pdf"
MDCG_DIR = ROOT_DIR / "data" / "raw" / "mdcg"
MDCG_CATALOG_ENTRIES_PATH = MDCG_DIR / "mdcg_catalog_entries.json"

OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "catalog.json"


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def make_document_id(filename: str) -> str:
    name = Path(filename).stem.lower()
    name = re.sub(r"^\d+_", "", name)
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def get_pdf_preview(pdf_path: Path, max_pages: int = 2) -> tuple[str, int]:
    if not pdf_path.exists():
        return "", 0

    try:
        texts = []

        with fitz.open(pdf_path) as doc:
            page_count = len(doc)

            for page_index in range(min(max_pages, page_count)):
                text = doc[page_index].get_text("text")
                texts.append(text)

        return clean_text(" ".join(texts)), page_count

    except Exception as error:
        print(f"Skipping unreadable PDF preview: {pdf_path.name} | {error}")
        return "", 0

def load_mdcg_entries() -> List[Dict]:
    if not MDCG_CATALOG_ENTRIES_PATH.exists():
        raise FileNotFoundError(
            f"MDCG catalog entries not found: {MDCG_CATALOG_ENTRIES_PATH}\n"
            "Run: python src/download_mdcg_catalog.py"
        )

    with MDCG_CATALOG_ENTRIES_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_mdr_catalog_entry() -> Dict:
    preview, page_count = get_pdf_preview(MDR_PATH, max_pages=2)
    
    return {
        "document_id": "mdr_2017_745",
        "source_type": "MDR",
        "title": "Regulation (EU) 2017/745 Medical Device Regulation",
        "reference": "MDR 2017/745",
        "category": "Binding EU regulation",
        "local_path": str(MDR_PATH.relative_to(ROOT_DIR)),
        "url": None,
        "page_count": page_count,
        "search_text": clean_text(
            "Regulation EU 2017/745 Medical Device Regulation MDR "
            "definitions intended purpose classification conformity assessment "
            "technical documentation risk management clinical evaluation "
            "post-market surveillance vigilance labelling notified bodies "
            + preview
        ),
    }


def build_mdcg_catalog_entries() -> List[Dict]:
    raw_entries = load_mdcg_entries()
    catalog_entries = []

    seen_document_ids = set()

    for entry in raw_entries:
        local_path = entry.get("local_path")

        if not local_path:
            continue

        pdf_path = ROOT_DIR / local_path

        if not pdf_path.exists():
            continue

        document_id = make_document_id(pdf_path.name)

        if document_id in seen_document_ids:
            continue

        seen_document_ids.add(document_id)

        preview, page_count = get_pdf_preview(pdf_path, max_pages=2)
        if page_count == 0:
            continue

        title = entry.get("title") or pdf_path.stem
        reference = entry.get("reference") or title
        category = entry.get("category") or "MDCG guidance"

        search_text = clean_text(
            f"{category} {reference} {title} "
            f"{entry.get('filename_from_url') or ''} "
            f"{preview}"
        )

        catalog_entries.append({
            "document_id": document_id,
            "source_type": "MDCG",
            "title": title,
            "reference": reference,
            "category": category,
            "local_path": local_path,
            "url": entry.get("url"),
            "page_count": page_count,
            "search_text": search_text,
        })

    return catalog_entries


def save_catalog(catalog: List[Dict]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)


def main() -> None:
    print("Building document catalog...")

    catalog = [build_mdr_catalog_entry()]
    catalog.extend(build_mdcg_catalog_entries())

    save_catalog(catalog)

    print("\nDone.")
    print(f"Catalog entries: {len(catalog)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()