"""
build_index.py

This file builds the searchable indexes for the two-stage MDR/MDCG RAG pipeline.
It creates one document-level catalog index for Stage 1 retrieval and one
chunk-level index for Stage 2 retrieval. Both stages use hybrid search:
BM25 for exact regulatory keywords and FAISS/FastEmbed for semantic similarity.
The saved index files are generated artifacts and should not be committed to Git.
"""

from pathlib import Path
import json
import pickle
from collections import Counter
from typing import List, Dict

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from fastembed import TextEmbedding


ROOT_DIR = Path(__file__).resolve().parents[1]

CHUNKS_PATH = ROOT_DIR / "data" / "processed" / "chunks.json"
CATALOG_PATH = ROOT_DIR / "data" / "processed" / "catalog.json"

INDEX_DIR = ROOT_DIR / "indexes"

CHUNK_FAISS_PATH = INDEX_DIR / "chunk_faiss.index"
CHUNK_BM25_PATH = INDEX_DIR / "chunk_bm25.pkl"
CHUNK_METADATA_PATH = INDEX_DIR / "chunk_metadata.json"

CATALOG_FAISS_PATH = INDEX_DIR / "catalog_faiss.index"
CATALOG_BM25_PATH = INDEX_DIR / "catalog_bm25.pkl"
CATALOG_METADATA_PATH = INDEX_DIR / "catalog_metadata.json"

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"


def load_json(path: Path, label: str) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    clean_items = []

    for item in data:
        text = get_search_text(item).strip()

        if not text:
            continue

        clean_items.append(item)

    if not clean_items:
        raise ValueError(f"No valid items found in {label}")

    return clean_items


def get_search_text(item: Dict) -> str:
    if "text" in item:
        return item.get("text", "")

    return " ".join([
        str(item.get("source_type", "")),
        str(item.get("category", "")),
        str(item.get("reference", "")),
        str(item.get("title", "")),
        str(item.get("search_text", "")),
    ])


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def build_bm25(items: List[Dict]) -> BM25Okapi:
    texts = [get_search_text(item) for item in items]
    tokenized = [tokenize(text) for text in texts]
    return BM25Okapi(tokenized)


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms


def build_faiss_index(items: List[Dict], model: TextEmbedding) -> faiss.IndexFlatIP:
    texts = [get_search_text(item) for item in items]

    embeddings = list(model.embed(texts))
    embeddings = np.array(embeddings, dtype="float32")
    embeddings = normalize_vectors(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


def save_index_outputs(
    faiss_index: faiss.IndexFlatIP,
    bm25_index: BM25Okapi,
    metadata: List[Dict],
    faiss_path: Path,
    bm25_path: Path,
    metadata_path: Path,
) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(faiss_index, str(faiss_path))

    with bm25_path.open("wb") as f:
        pickle.dump(bm25_index, f)

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def print_statistics(label: str, items: List[Dict]) -> None:
    source_counts = Counter(item.get("source_type", "Unknown") for item in items)

    print(f"\n{label} statistics:")
    print(f"- Total items: {len(items)}")

    for source_type, count in source_counts.items():
        print(f"- {source_type}: {count}")


def main() -> None:
    print("Loading catalog and chunks...")
    catalog_items = load_json(CATALOG_PATH, "catalog.json")
    chunk_items = load_json(CHUNKS_PATH, "chunks.json")

    print_statistics("Catalog", catalog_items)
    print_statistics("Chunks", chunk_items)

    print("\nLoading FastEmbed model...")
    model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)

    print("\nBuilding catalog BM25 index...")
    catalog_bm25 = build_bm25(catalog_items)

    print("Building catalog FAISS index...")
    catalog_faiss = build_faiss_index(catalog_items, model)

    print("\nBuilding chunk BM25 index...")
    chunk_bm25 = build_bm25(chunk_items)

    print("Building chunk FAISS index...")
    chunk_faiss = build_faiss_index(chunk_items, model)

    save_index_outputs(
        faiss_index=catalog_faiss,
        bm25_index=catalog_bm25,
        metadata=catalog_items,
        faiss_path=CATALOG_FAISS_PATH,
        bm25_path=CATALOG_BM25_PATH,
        metadata_path=CATALOG_METADATA_PATH,
    )

    save_index_outputs(
        faiss_index=chunk_faiss,
        bm25_index=chunk_bm25,
        metadata=chunk_items,
        faiss_path=CHUNK_FAISS_PATH,
        bm25_path=CHUNK_BM25_PATH,
        metadata_path=CHUNK_METADATA_PATH,
    )

    print("\nIndexes saved:")
    print(f"- {CATALOG_FAISS_PATH}")
    print(f"- {CATALOG_BM25_PATH}")
    print(f"- {CATALOG_METADATA_PATH}")
    print(f"- {CHUNK_FAISS_PATH}")
    print(f"- {CHUNK_BM25_PATH}")
    print(f"- {CHUNK_METADATA_PATH}")


if __name__ == "__main__":
    main()