from pathlib import Path
import json
import time
import fitz  # PyMuPDF


ROOT_DIR = Path(__file__).resolve().parents[1]

PDF_PATH = ROOT_DIR / "data" / "raw" / "MDR.pdf"
OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "mdr_pages.json"

MAX_FILE_AGE_HOURS = 24


def delete_if_older_than_24h(path: Path) -> None:
    if not path.exists():
        return

    age_seconds = time.time() - path.stat().st_mtime
    age_hours = age_seconds / 3600

    if age_hours > MAX_FILE_AGE_HOURS:
        path.unlink()
        print(f"Deleted old file: {path}")


def extract_pdf_pages(pdf_path: Path) -> list[dict]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = []

    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()

            pages.append({
                "page": page_index,
                "text": text
            })

    return pages


def save_pages(pages: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # "w" overwrites the file. It does not append.
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)


def main() -> None:
    delete_if_older_than_24h(OUTPUT_PATH)

    pages = extract_pdf_pages(PDF_PATH)
    save_pages(pages, OUTPUT_PATH)

    print(f"Extracted pages: {len(pages)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()