from pathlib import Path
import json
import re
import time
from typing import List, Dict, Optional


ROOT_DIR = Path(__file__).resolve().parents[1]

PAGES_PATH = ROOT_DIR / "data" / "processed" / "mdr_pages.json"
CHUNKS_PATH = ROOT_DIR / "data" / "processed" / "chunks.json"

MAX_FILE_AGE_HOURS = 24

MAX_WORDS = 1200
TARGET_WORDS = 1000
OVERLAP_WORDS = 150


ARTICLE_PATTERN = re.compile(r"(?m)^Article\s+\d+[a-zA-Z]?\s*$")
ANNEX_PATTERN = re.compile(r"(?m)^ANNEX\s+[IVXLCDM]+\s*$")

ANNEX_SECTION_PATTERN = re.compile(
    r"(?m)^(Rule\s+\d+|[0-9]+\.\s+.+|[A-Z]\.\s+.+)$"
)


def delete_if_older_than_24h(path: Path) -> None:
    if not path.exists():
        return

    age_seconds = time.time() - path.stat().st_mtime
    age_hours = age_seconds / 3600

    if age_hours > MAX_FILE_AGE_HOURS:
        path.unlink()
        print(f"Deleted old file: {path}")


def load_pages(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python src/extract_pdf.py first"
        )

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def clean_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_full_text_with_pages(pages: List[Dict]) -> str:
    parts = []

    for page in pages:
        page_num = page["page"]
        text = clean_text(page["text"])

        parts.append(f"\n\n[[PAGE:{page_num}]]\n{text}")

    return "\n".join(parts)


def find_page_range(text: str) -> tuple[Optional[int], Optional[int]]:
    page_numbers = re.findall(r"\[\[PAGE:(\d+)\]\]", text)

    if not page_numbers:
        return None, None

    page_numbers = [int(p) for p in page_numbers]
    return min(page_numbers), max(page_numbers)


def remove_page_markers(text: str) -> str:
    return re.sub(r"\[\[PAGE:\d+\]\]", "", text).strip()


def word_count(text: str) -> int:
    return len(text.split())


def split_long_text(text: str) -> List[str]:
    words = text.split()

    if len(words) <= MAX_WORDS:
        return [text]

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + TARGET_WORDS, len(words))
        chunks.append(" ".join(words[start:end]))

        if end == len(words):
            break

        start = max(0, end - OVERLAP_WORDS)

    return chunks


def detect_boundaries(full_text: str) -> List[tuple[int, str, str]]:
    boundaries = []

    for match in ARTICLE_PATTERN.finditer(full_text):
        boundaries.append((match.start(), "article", match.group().strip()))

    for match in ANNEX_PATTERN.finditer(full_text):
        boundaries.append((match.start(), "annex", match.group().strip()))

    boundaries.sort(key=lambda x: x[0])
    return boundaries


def split_annex_into_sections(annex_text: str) -> List[tuple[str, str]]:
    matches = list(ANNEX_SECTION_PATTERN.finditer(annex_text))

    if not matches:
        return [("annex_full", annex_text)]

    sections = []

    # keep intro before first section
    if matches[0].start() > 0:
        intro = annex_text[:matches[0].start()].strip()
        if intro:
            sections.append(("annex_intro", intro))

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(annex_text)

        title = match.group().strip()
        section_text = annex_text[start:end].strip()

        if section_text:
            sections.append((title, section_text))

    return sections


def create_chunks_from_section(
    section_text: str,
    chunk_type: str,
    title: str,
    chunk_counter: int,
) -> tuple[List[Dict], int]:
    chunks = []

    page_start, page_end = find_page_range(section_text)
    clean_section_text = remove_page_markers(section_text)

    subchunks = split_long_text(clean_section_text)

    for sub_index, sub_text in enumerate(subchunks, start=1):
        chunk = {
            "chunk_id": f"mdr_chunk_{chunk_counter:04d}",
            "source": "MDR.pdf",
            "type": chunk_type,
            "title": title,
            "page_start": page_start,
            "page_end": page_end,
            "subchunk": sub_index,
            "word_count": word_count(sub_text),
            "text": sub_text,
        }

        chunks.append(chunk)
        chunk_counter += 1

    return chunks, chunk_counter


def create_legal_chunks(full_text: str) -> List[Dict]:
    boundaries = detect_boundaries(full_text)

    if not boundaries:
        raise ValueError("No Article or Annex boundaries detected.")

    chunks = []
    chunk_counter = 1

    for i, (start, chunk_type, title) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(full_text)
        section_text = full_text[start:end].strip()

        if chunk_type == "article":
            new_chunks, chunk_counter = create_chunks_from_section(
                section_text=section_text,
                chunk_type="article",
                title=title,
                chunk_counter=chunk_counter,
            )
            chunks.extend(new_chunks)

        elif chunk_type == "annex":
            annex_sections = split_annex_into_sections(section_text)

            for section_title, annex_section_text in annex_sections:
                full_title = f"{title} - {section_title}"

                new_chunks, chunk_counter = create_chunks_from_section(
                    section_text=annex_section_text,
                    chunk_type="annex",
                    title=full_title,
                    chunk_counter=chunk_counter,
                )
                chunks.extend(new_chunks)

    return chunks


def save_chunks(chunks: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    # "w" overwrites the file. It does not append.
    with path.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def main() -> None:
    delete_if_older_than_24h(CHUNKS_PATH)

    pages = load_pages(PAGES_PATH)
    full_text = build_full_text_with_pages(pages)
    chunks = create_legal_chunks(full_text)

    save_chunks(chunks, CHUNKS_PATH)

    print(f"Created chunks: {len(chunks)}")
    print(f"Saved to: {CHUNKS_PATH}")


if __name__ == "__main__":
    main()