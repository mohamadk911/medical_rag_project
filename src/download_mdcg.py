from pathlib import Path
from urllib.parse import urljoin, urlparse
import json
import re
import time

import requests
from bs4 import BeautifulSoup


ROOT_DIR = Path(__file__).resolve().parents[1]

MDCG_PAGE_URL = (
    "https://health.ec.europa.eu/medical-devices-sector/"
    "new-regulations/guidance-mdcg-endorsed-documents-and-other-guidance_en"
)

OUTPUT_DIR = ROOT_DIR / "data" / "raw" / "mdcg"
MANIFEST_PATH = OUTPUT_DIR / "mdcg_manifest.json"
ERRORS_PATH = OUTPUT_DIR / "mdcg_download_errors.json"

REQUEST_TIMEOUT = 60
SLEEP_SECONDS = 10
MAX_RETRIES = 5
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/pdf,text/html,*/*",
}


def safe_filename(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[^\w\-\.]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text[:180]


def is_probable_pdf_url(url: str) -> bool:
    lower = url.lower()
    return (
        lower.endswith(".pdf")
        or "/document/download/" in lower
        or "filename=" in lower and ".pdf" in lower
    )


def get_html(url: str) -> str:
    response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.text


def collect_pdf_links(page_url: str) -> list[dict]:
    html = get_html(page_url)
    soup = BeautifulSoup(html, "html.parser")

    links = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        full_url = urljoin(page_url, href)

        if not is_probable_pdf_url(full_url):
            continue

        title = " ".join(a.get_text(" ", strip=True).split())
        if not title:
            title = Path(urlparse(full_url).path).stem

        links.append({
            "title": title,
            "url": full_url,
        })

    # deduplicate by URL
    unique = {}
    for item in links:
        unique[item["url"]] = item

    return list(unique.values())


def guess_filename(url: str, title: str, index: int) -> str:
    parsed = urlparse(url)

    # Try filename parameter first
    match = re.search(r"filename=([^&]+)", url)
    if match:
        filename = match.group(1)
    else:
        filename = Path(parsed.path).name

    if not filename.lower().endswith(".pdf"):
        filename = f"{safe_filename(title)}.pdf"

    filename = safe_filename(filename)

    if not filename:
        filename = f"mdcg_document_{index:03d}.pdf"

    return f"{index:03d}_{filename}"


def download_file(url: str, output_path: Path) -> dict:
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with requests.get(
                url,
                headers=HEADERS,
                stream=True,
                timeout=REQUEST_TIMEOUT,
            ) as response:
                response.raise_for_status()

                content_type = response.headers.get("content-type", "")
                total_bytes = int(response.headers.get("content-length", 0))

                with output_path.open("wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)

            size_bytes = output_path.stat().st_size

            return {
                "content_type": content_type,
                "expected_bytes": total_bytes,
                "downloaded_bytes": size_bytes,
            }

        except requests.RequestException as error:
            last_error = error

            if attempt == MAX_RETRIES:
                raise

            time.sleep(min(SLEEP_SECONDS, attempt * 2))

    raise RuntimeError(f"Failed to download {url}: {last_error}")


def validate_pdf(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False

    with path.open("rb") as f:
        header = f.read(5)

    return header == b"%PDF-"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Collecting MDCG PDF links...")
    pdf_links = collect_pdf_links(MDCG_PAGE_URL)

    print(f"Found PDF-like links: {len(pdf_links)}")

    manifest = []
    errors = []

    for index, item in enumerate(pdf_links, start=1):
        title = item["title"]
        url = item["url"]
        filename = guess_filename(url, title, index)
        output_path = OUTPUT_DIR / filename

        print(f"[{index}/{len(pdf_links)}] {title}")

        try:
            if output_path.exists() and validate_pdf(output_path):
                status = "already_exists"
                download_info = {
                    "content_type": None,
                    "expected_bytes": output_path.stat().st_size,
                    "downloaded_bytes": output_path.stat().st_size,
                }
            else:
                download_info = download_file(url, output_path)
                status = "downloaded"

            is_valid_pdf = validate_pdf(output_path)

            manifest.append({
                "title": title,
                "url": url,
                "filename": filename,
                "path": str(output_path.relative_to(ROOT_DIR)),
                "status": status,
                "valid_pdf": is_valid_pdf,
                **download_info,
            })

            if not is_valid_pdf:
                errors.append({
                    "title": title,
                    "url": url,
                    "filename": filename,
                    "error": "Downloaded file is not a valid PDF",
                })

            time.sleep(SLEEP_SECONDS)

        except Exception as error:
            errors.append({
                "title": title,
                "url": url,
                "filename": filename,
                "error": str(error),
            })

    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    with ERRORS_PATH.open("w", encoding="utf-8") as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"Downloaded/checked documents: {len(manifest)}")
    print(f"Errors: {len(errors)}")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Errors file: {ERRORS_PATH}")


if __name__ == "__main__":
    main()
