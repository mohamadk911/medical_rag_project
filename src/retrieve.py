"""
retrieve.py

This file performs Stage 2 retrieval for the MDR/MDCG RAG pipeline.

Method used:
1. Stage 1 document retrieval is handled by catalog_retrieve.py.
   It searches the document-level catalog and selects relevant MDR/MDCG documents.
2. This file then performs Stage 2 chunk retrieval only inside selected documents.
3. MDR and MDCG are retrieved as two separate evidence tracks:
   - MDR chunks = binding legal basis
   - MDCG chunks = explanatory guidance
4. Retrieval uses hybrid search:
   - BM25 for exact keyword/legal term matching
   - FAISS/FastEmbed for semantic similarity
5. The result is returned as separate lists:
   {
       "mdr_chunks": [...],
       "mdcg_chunks": [...],
       "selected_documents": [...]
   }

This file does not generate answers. It only retrieves evidence.
"""

from pathlib import Path
import json
import pickle
from typing import List, Dict, Tuple

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from fastembed import TextEmbedding

from catalog_retrieve import catalog_retrieve


ROOT_DIR = Path(__file__).resolve().parents[1]

INDEX_DIR = ROOT_DIR / "indexes"

CHUNK_FAISS_PATH = INDEX_DIR / "chunk_faiss.index"
CHUNK_BM25_PATH = INDEX_DIR / "chunk_bm25.pkl"
CHUNK_METADATA_PATH = INDEX_DIR / "chunk_metadata.json"

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

BM25_TOP_K = 80
DENSE_TOP_K = 80

MDR_FINAL_TOP_K = 6
MDCG_FINAL_TOP_K = 8

MDCG_DOCS_PER_ATTEMPT = 5
CATALOG_TOP_K = 40

BM25_WEIGHT = 0.6
DENSE_WEIGHT = 0.4


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def normalize_scores(results: List[Dict], score_key: str, norm_key: str) -> None:
    scores = [r.get(score_key, 0.0) for r in results]

    if not scores:
        return

    min_score = min(scores)
    max_score = max(scores)

    for result in results:
        score = result.get(score_key, 0.0)

        if max_score == min_score:
            result[norm_key] = 1.0 if score > 0 else 0.0
        else:
            result[norm_key] = (score - min_score) / (max_score - min_score)


def load_chunk_indexes() -> Tuple[faiss.Index, BM25Okapi, List[Dict]]:
    if not CHUNK_FAISS_PATH.exists():
        raise FileNotFoundError("chunk_faiss.index not found. Run: python src/build_index.py")

    if not CHUNK_BM25_PATH.exists():
        raise FileNotFoundError("chunk_bm25.pkl not found. Run: python src/build_index.py")

    if not CHUNK_METADATA_PATH.exists():
        raise FileNotFoundError("chunk_metadata.json not found. Run: python src/build_index.py")

    faiss_index = faiss.read_index(str(CHUNK_FAISS_PATH))

    with CHUNK_BM25_PATH.open("rb") as f:
        bm25_index = pickle.load(f)

    with CHUNK_METADATA_PATH.open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    return faiss_index, bm25_index, chunks


def select_documents(query: str, attempt: int = 1) -> Dict[str, List[Dict]]:
    catalog_results = catalog_retrieve(query, top_k=CATALOG_TOP_K)

    mdr_docs = [
        doc for doc in catalog_results
        if doc.get("source_type") == "MDR"
    ]

    if not mdr_docs:
        mdr_docs = [{
            "document_id": "mdr_2017_745",
            "source_type": "MDR",
            "title": "Regulation (EU) 2017/745 Medical Device Regulation",
        }]

    mdcg_docs_all = [
        doc for doc in catalog_results
        if doc.get("source_type") == "MDCG"
    ]

    start = (attempt - 1) * MDCG_DOCS_PER_ATTEMPT
    end = start + MDCG_DOCS_PER_ATTEMPT

    return {
        "mdr_docs": mdr_docs[:1],
        "mdcg_docs": mdcg_docs_all[start:end],
    }


def chunk_allowed(
    chunk: Dict,
    source_type: str,
    document_ids: set[str],
) -> bool:
    return (
        chunk.get("source_type") == source_type
        and chunk.get("document_id") in document_ids
    )


def bm25_search_filtered(
    query: str,
    bm25_index: BM25Okapi,
    chunks: List[Dict],
    source_type: str,
    document_ids: set[str],
    top_k: int,
) -> List[Dict]:
    scores = bm25_index.get_scores(tokenize(query))
    ranked_indices = np.argsort(scores)[::-1]

    results = []

    for idx in ranked_indices:
        chunk = chunks[int(idx)]

        if not chunk_allowed(chunk, source_type, document_ids):
            continue

        result = chunk.copy()
        result["bm25_score"] = float(scores[idx])
        result["dense_score"] = 0.0
        result["retrieval_source"] = "bm25"
        results.append(result)

        if len(results) >= top_k:
            break

    return results


def dense_search_filtered(
    query: str,
    faiss_index: faiss.Index,
    chunks: List[Dict],
    model: TextEmbedding,
    source_type: str,
    document_ids: set[str],
    top_k: int,
) -> List[Dict]:
    query_embedding = list(model.embed([query]))[0]
    query_embedding = np.array(query_embedding, dtype="float32")
    query_embedding = normalize_vector(query_embedding)
    query_embedding = np.expand_dims(query_embedding, axis=0)

    scores, indices = faiss_index.search(query_embedding, len(chunks))

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        chunk = chunks[int(idx)]

        if not chunk_allowed(chunk, source_type, document_ids):
            continue

        result = chunk.copy()
        result["dense_score"] = float(score)
        result["bm25_score"] = 0.0
        result["retrieval_source"] = "dense"
        results.append(result)

        if len(results) >= top_k:
            break

    return results


def merge_results(
    bm25_results: List[Dict],
    dense_results: List[Dict],
    final_top_k: int,
) -> List[Dict]:
    normalize_scores(bm25_results, "bm25_score", "bm25_score_norm")
    normalize_scores(dense_results, "dense_score", "dense_score_norm")

    merged = {}

    for result in bm25_results + dense_results:
        chunk_id = result["chunk_id"]

        if chunk_id not in merged:
            merged[chunk_id] = result
            merged[chunk_id].setdefault("bm25_score_norm", 0.0)
            merged[chunk_id].setdefault("dense_score_norm", 0.0)
        else:
            existing = merged[chunk_id]

            existing["bm25_score"] = max(
                existing.get("bm25_score", 0.0),
                result.get("bm25_score", 0.0),
            )
            existing["dense_score"] = max(
                existing.get("dense_score", 0.0),
                result.get("dense_score", 0.0),
            )
            existing["bm25_score_norm"] = max(
                existing.get("bm25_score_norm", 0.0),
                result.get("bm25_score_norm", 0.0),
            )
            existing["dense_score_norm"] = max(
                existing.get("dense_score_norm", 0.0),
                result.get("dense_score_norm", 0.0),
            )
            existing["retrieval_source"] = "hybrid"

    results = list(merged.values())

    for result in results:
        result["hybrid_score"] = (
            BM25_WEIGHT * result.get("bm25_score_norm", 0.0)
            + DENSE_WEIGHT * result.get("dense_score_norm", 0.0)
        )

    results.sort(key=lambda x: x["hybrid_score"], reverse=True)

    return results[:final_top_k]


def retrieve_track(
    query: str,
    source_type: str,
    document_ids: set[str],
    faiss_index: faiss.Index,
    bm25_index: BM25Okapi,
    chunks: List[Dict],
    model: TextEmbedding,
    final_top_k: int,
) -> List[Dict]:
    if not document_ids:
        return []

    bm25_results = bm25_search_filtered(
        query=query,
        bm25_index=bm25_index,
        chunks=chunks,
        source_type=source_type,
        document_ids=document_ids,
        top_k=BM25_TOP_K,
    )

    dense_results = dense_search_filtered(
        query=query,
        faiss_index=faiss_index,
        chunks=chunks,
        model=model,
        source_type=source_type,
        document_ids=document_ids,
        top_k=DENSE_TOP_K,
    )

    return merge_results(
        bm25_results=bm25_results,
        dense_results=dense_results,
        final_top_k=final_top_k,
    )


def retrieve(query: str, attempt: int = 1) -> Dict:
    selected = select_documents(query, attempt=attempt)

    mdr_doc_ids = {
        doc["document_id"]
        for doc in selected["mdr_docs"]
        if doc.get("document_id")
    }

    mdcg_doc_ids = {
        doc["document_id"]
        for doc in selected["mdcg_docs"]
        if doc.get("document_id")
    }

    faiss_index, bm25_index, chunks = load_chunk_indexes()

    print("Loading FastEmbed model for chunk retrieval...")
    model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)

    mdr_chunks = retrieve_track(
        query=query,
        source_type="MDR",
        document_ids=mdr_doc_ids,
        faiss_index=faiss_index,
        bm25_index=bm25_index,
        chunks=chunks,
        model=model,
        final_top_k=MDR_FINAL_TOP_K,
    )

    mdcg_chunks = retrieve_track(
        query=query,
        source_type="MDCG",
        document_ids=mdcg_doc_ids,
        faiss_index=faiss_index,
        bm25_index=bm25_index,
        chunks=chunks,
        model=model,
        final_top_k=MDCG_FINAL_TOP_K,
    )

    return {
        "attempt": attempt,
        "selected_mdr_documents": selected["mdr_docs"],
        "selected_mdcg_documents": selected["mdcg_docs"],
        "mdr_chunks": mdr_chunks,
        "mdcg_chunks": mdcg_chunks,
    }


def main() -> None:
    query = input("Ask regulatory question: ")

    result = retrieve(query)

    print("\nSelected MDR documents:")
    for doc in result["selected_mdr_documents"]:
        print("-", doc.get("reference") or doc.get("title"))

    print("\nSelected MDCG documents:")
    for doc in result["selected_mdcg_documents"]:
        print("-", doc.get("reference") or doc.get("title"))

    print("\nTop MDR chunks:")
    for chunk in result["mdr_chunks"]:
        print("-", chunk.get("title"), "|", chunk.get("hybrid_score"))

    print("\nTop MDCG chunks:")
    for chunk in result["mdcg_chunks"]:
        print("-", chunk.get("title"), "|", chunk.get("hybrid_score"))


if __name__ == "__main__":
    main()