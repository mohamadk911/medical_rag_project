from pathlib import Path
from urllib.parse import urljoin
import json
import re

import requests
from bs4 import BeautifulSoup


ROOT_DIR = Path(__file__).resolve().parents[1]

MDCG_PAGE_URL = (
    "https://health.ec.europa.eu/medical-devices-sector/"
    "new-regulations/guidance-mdcg-endorsed-documents-and-other-guidance_en"
)

OUTPUT_DIR = ROOT_DIR / "data" / "raw" / "mdcg"

CATALOG_HTML_PATH = OUTPUT_DIR / "mdcg_catalog_page.html"
CATALOG_ENTRIES_PATH = OUTPUT_DIR / "mdcg_catalog_entries.json"
MANIFEST_PATH = OUTPUT_DIR / "mdcg_manifest.json"


def clean_text(text: str) -> str:
    return " ".join(text.split())


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {}

    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    return {item.get("url"): item for item in manifest}


def download_catalog_page() -> str:
    response = requests.get(MDCG_PAGE_URL, timeout=60)
    response.raise_for_status()
    return response.text


def extract_catalog_entries(html: str, manifest_by_url: dict) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")

    entries = []
    current_category = "Uncategorized"

    for element in soup.find_all(["h2", "h3", "a"]):
        if element.name in ["h2", "h3"]:
            current_category = clean_text(element.get_text(" ", strip=True))

        if element.name == "a" and element.get("href"):
            url = urljoin(MDCG_PAGE_URL, element["href"])
            title = clean_text(element.get_text(" ", strip=True))

            if "/document/download/" not in url:
                continue

            filename_match = re.search(r"filename=([^&]+)", url)
            reference_match = re.search(r"(MDCG\s*\d{4}[-–]\d+[^,\n]*)", title, re.IGNORECASE)

            manifest_item = manifest_by_url.get(url, {})

            entries.append({
                "source_type": "MDCG",
                "category": current_category,
                "title": title,
                "reference": reference_match.group(1) if reference_match else title,
                "url": url,
                "filename_from_url": filename_match.group(1) if filename_match else None,
                "local_filename": manifest_item.get("filename"),
                "local_path": manifest_item.get("path"),
                "download_status": manifest_item.get("status"),
                "valid_pdf": manifest_item.get("valid_pdf"),
                "search_text": clean_text(
                    f"{current_category} {title} {filename_match.group(1) if filename_match else ''}"
                ),
            })

    return entries


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_by_url = load_manifest()

    print("Downloading MDCG catalog page...")
    html = download_catalog_page()

    CATALOG_HTML_PATH.write_text(html, encoding="utf-8")

    print("Extracting catalog entries...")
    entries = extract_catalog_entries(html, manifest_by_url)

    with CATALOG_ENTRIES_PATH.open("w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"Saved HTML page: {CATALOG_HTML_PATH}")
    print(f"Saved catalog entries: {CATALOG_ENTRIES_PATH}")
    print(f"Catalog entries: {len(entries)}")


if __name__ == "__main__":
    main()