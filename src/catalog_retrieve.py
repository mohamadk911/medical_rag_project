"""
catalog_retrieve.py

This file performs Stage 1 retrieval for the MDR/MDCG RAG pipeline.
It loads the prebuilt document-level catalog indexes created by build_index.py.
The goal is to select the most relevant regulatory documents first, before
Stage 2 searches inside their chunks.
"""

from pathlib import Path
import json
import pickle
from typing import List, Dict, Tuple

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from fastembed import TextEmbedding


ROOT_DIR = Path(__file__).resolve().parents[1]

INDEX_DIR = ROOT_DIR / "indexes"
CATALOG_FAISS_PATH = INDEX_DIR / "catalog_faiss.index"
CATALOG_BM25_PATH = INDEX_DIR / "catalog_bm25.pkl"
CATALOG_METADATA_PATH = INDEX_DIR / "catalog_metadata.json"

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

BM25_WEIGHT = 0.6
DENSE_WEIGHT = 0.4
DEFAULT_TOP_K = 20


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    if len(scores) == 0:
        return scores

    min_score = float(np.min(scores))
    max_score = float(np.max(scores))

    if max_score == min_score:
        return np.ones_like(scores) if max_score > 0 else np.zeros_like(scores)

    return (scores - min_score) / (max_score - min_score)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)

    if norm == 0:
        return vector

    return vector / norm


def load_catalog_indexes() -> Tuple[faiss.Index, BM25Okapi, List[Dict]]:
    if not CATALOG_FAISS_PATH.exists():
        raise FileNotFoundError("Catalog FAISS index not found. Run: python src/build_index.py")

    if not CATALOG_BM25_PATH.exists():
        raise FileNotFoundError("Catalog BM25 index not found. Run: python src/build_index.py")

    if not CATALOG_METADATA_PATH.exists():
        raise FileNotFoundError("Catalog metadata not found. Run: python src/build_index.py")

    faiss_index = faiss.read_index(str(CATALOG_FAISS_PATH))

    with CATALOG_BM25_PATH.open("rb") as f:
        bm25_index = pickle.load(f)

    with CATALOG_METADATA_PATH.open("r", encoding="utf-8") as f:
        catalog = json.load(f)

    return faiss_index, bm25_index, catalog


def catalog_retrieve(query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict]:
    faiss_index, bm25_index, catalog = load_catalog_indexes()

    # BM25 scores over catalog documents
    bm25_scores = np.array(
        bm25_index.get_scores(tokenize(query)),
        dtype="float32",
    )
    bm25_norm = normalize_scores(bm25_scores)

    # Dense query search over catalog FAISS index
    print("Loading FastEmbed model for catalog retrieval...")
    model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)

    query_embedding = list(model.embed([query]))[0]
    query_embedding = normalize_vector(np.array(query_embedding, dtype="float32"))
    query_embedding = np.expand_dims(query_embedding, axis=0)

    dense_scores_raw, dense_indices = faiss_index.search(query_embedding, len(catalog))

    dense_scores = np.zeros(len(catalog), dtype="float32")

    for score, idx in zip(dense_scores_raw[0], dense_indices[0]):
        if idx == -1:
            continue
        dense_scores[int(idx)] = float(score)

    dense_norm = normalize_scores(dense_scores)

    hybrid_scores = (BM25_WEIGHT * bm25_norm) + (DENSE_WEIGHT * dense_norm)

    ranked_indices = np.argsort(hybrid_scores)[::-1][:top_k]

    results = []

    for rank, idx in enumerate(ranked_indices, start=1):
        item = catalog[int(idx)].copy()
        item["catalog_rank"] = rank
        item["catalog_bm25_score"] = float(bm25_scores[idx])
        item["catalog_dense_score"] = float(dense_scores[idx])
        item["catalog_hybrid_score"] = float(hybrid_scores[idx])
        results.append(item)

    return results


def main() -> None:
    query = input("Ask regulatory question for catalog retrieval: ")

    results = catalog_retrieve(query)

    print("\nTop catalog documents:\n")

    for result in results:
        print("=" * 80)
        print(f"Rank: {result.get('catalog_rank')}")
        print(f"Document ID: {result.get('document_id')}")
        print(f"Source type: {result.get('source_type')}")
        print(f"Reference: {result.get('reference')}")
        print(f"Title: {result.get('title')}")
        print(f"Category: {result.get('category')}")
        print(f"Pages: {result.get('page_count')}")
        print(f"Score: {result.get('catalog_hybrid_score')}")
        print()


if __name__ == "__main__":
    main()