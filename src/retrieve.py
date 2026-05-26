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
FAISS_PATH = INDEX_DIR / "faiss.index"
BM25_PATH = INDEX_DIR / "bm25.pkl"
METADATA_PATH = INDEX_DIR / "metadata.json"

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

BM25_TOP_K = 50
DENSE_TOP_K = 50
FINAL_TOP_K = 15  # keep more candidates, ready for LLM reranker later

BM25_WEIGHT = 0.6
DENSE_WEIGHT = 0.4


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def normalize_scores(results: List[Dict], score_key: str, normalized_key: str) -> None:
    scores = [result.get(score_key, 0.0) for result in results]

    if not scores:
        return

    min_score = min(scores)
    max_score = max(scores)

    for result in results:
        score = result.get(score_key, 0.0)

        if max_score == min_score:
            result[normalized_key] = 1.0 if score > 0 else 0.0
        else:
            result[normalized_key] = (score - min_score) / (max_score - min_score)


def load_indexes() -> Tuple[faiss.Index, BM25Okapi, List[Dict]]:
    if not FAISS_PATH.exists():
        raise FileNotFoundError("FAISS index not found. Run: python src/build_index.py")

    if not BM25_PATH.exists():
        raise FileNotFoundError("BM25 index not found. Run: python src/build_index.py")

    if not METADATA_PATH.exists():
        raise FileNotFoundError("metadata.json not found. Run: python src/build_index.py")

    faiss_index = faiss.read_index(str(FAISS_PATH))

    with BM25_PATH.open("rb") as f:
        bm25_index = pickle.load(f)

    with METADATA_PATH.open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    return faiss_index, bm25_index, chunks


def bm25_search(query: str, bm25_index: BM25Okapi, chunks: List[Dict], top_k: int) -> List[Dict]:
    scores = bm25_index.get_scores(tokenize(query))
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for rank, idx in enumerate(top_indices, start=1):
        chunk = chunks[int(idx)].copy()
        chunk["bm25_score"] = float(scores[idx])
        chunk["dense_score"] = 0.0
        chunk["bm25_rank"] = rank
        chunk["dense_rank"] = None
        chunk["retrieval_source"] = "bm25"
        results.append(chunk)

    return results


def dense_search(query: str, faiss_index, chunks: List[Dict], model: TextEmbedding, top_k: int) -> List[Dict]:
    query_embedding = list(model.embed([query]))[0]
    query_embedding = np.array(query_embedding, dtype="float32")
    query_embedding = normalize_vector(query_embedding)
    query_embedding = np.expand_dims(query_embedding, axis=0)

    scores, indices = faiss_index.search(query_embedding, top_k)

    results = []

    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx == -1:
            continue

        chunk = chunks[int(idx)].copy()
        chunk["dense_score"] = float(score)
        chunk["bm25_score"] = 0.0
        chunk["dense_rank"] = rank
        chunk["bm25_rank"] = None
        chunk["retrieval_source"] = "dense"
        results.append(chunk)

    return results


def merge_results(bm25_results: List[Dict], dense_results: List[Dict]) -> List[Dict]:
    all_results = bm25_results + dense_results

    normalize_scores(bm25_results, "bm25_score", "bm25_score_norm")
    normalize_scores(dense_results, "dense_score", "dense_score_norm")

    merged = {}

    for result in all_results:
        chunk_id = result["chunk_id"]

        if chunk_id not in merged:
            merged[chunk_id] = result
            merged[chunk_id].setdefault("bm25_score_norm", 0.0)
            merged[chunk_id].setdefault("dense_score_norm", 0.0)
        else:
            existing = merged[chunk_id]

            existing["bm25_score"] = max(existing.get("bm25_score", 0.0), result.get("bm25_score", 0.0))
            existing["dense_score"] = max(existing.get("dense_score", 0.0), result.get("dense_score", 0.0))

            existing["bm25_score_norm"] = max(
                existing.get("bm25_score_norm", 0.0),
                result.get("bm25_score_norm", 0.0)
            )
            existing["dense_score_norm"] = max(
                existing.get("dense_score_norm", 0.0),
                result.get("dense_score_norm", 0.0)
            )

            if result.get("bm25_rank") is not None:
                existing["bm25_rank"] = result["bm25_rank"]

            if result.get("dense_rank") is not None:
                existing["dense_rank"] = result["dense_rank"]

            existing["retrieval_source"] = "hybrid"

    results = list(merged.values())

    for result in results:
        bm25_norm = result.get("bm25_score_norm", 0.0)
        dense_norm = result.get("dense_score_norm", 0.0)

        result["hybrid_score"] = (BM25_WEIGHT * bm25_norm) + (DENSE_WEIGHT * dense_norm)

    results.sort(key=lambda x: x["hybrid_score"], reverse=True)

    return results


def retrieve(query: str, final_top_k: int = FINAL_TOP_K) -> List[Dict]:
    faiss_index, bm25_index, chunks = load_indexes()

    print("Loading FastEmbed model...")
    model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)

    bm25_results = bm25_search(query, bm25_index, chunks, BM25_TOP_K)
    dense_results = dense_search(query, faiss_index, chunks, model, DENSE_TOP_K)

    merged_results = merge_results(bm25_results, dense_results)

    return merged_results[:final_top_k]


def main() -> None:
    query = input("Ask MDR question: ")

    results = retrieve(query)

    print("\nTop retrieved chunks:\n")

    for i, chunk in enumerate(results, start=1):
        print("=" * 80)
        print(f"Result {i}")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Title: {chunk.get('title')}")
        print(f"Type: {chunk.get('type')}")
        print(f"Pages: {chunk.get('page_start')} - {chunk.get('page_end')}")
        print(f"Source: {chunk.get('retrieval_source')}")
        print(f"BM25 score norm: {chunk.get('bm25_score_norm')}")
        print(f"Dense score norm: {chunk.get('dense_score_norm')}")
        print(f"Hybrid score: {chunk.get('hybrid_score')}")
        print()
        print(chunk["text"][:1000])
        print()


if __name__ == "__main__":
    main()