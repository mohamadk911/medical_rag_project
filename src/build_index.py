from pathlib import Path
import json
import pickle

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from fastembed import TextEmbedding


ROOT_DIR = Path(__file__).resolve().parents[1]

CHUNKS_PATH = ROOT_DIR / "data" / "processed" / "chunks.json"

INDEX_DIR = ROOT_DIR / "indexes"
FAISS_PATH = INDEX_DIR / "faiss.index"
BM25_PATH = INDEX_DIR / "bm25.pkl"
METADATA_PATH = INDEX_DIR / "metadata.json"

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"


def load_chunks(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"chunks.json not found: {path}\nRun: python src/chunk_mdr.py first"
        )

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def build_bm25(chunks: list[dict]) -> BM25Okapi:
    tokenized_chunks = [tokenize(chunk["text"]) for chunk in chunks]
    return BM25Okapi(tokenized_chunks)


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms


def build_faiss_index(chunks: list[dict]) -> faiss.IndexFlatIP:
    texts = [chunk["text"] for chunk in chunks]

    print("Loading FastEmbed model...")
    model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)

    print("Creating embeddings...")
    embeddings = list(model.embed(texts))

    embeddings = np.array(embeddings, dtype="float32")
    embeddings = normalize_vectors(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


def save_outputs(
    faiss_index: faiss.IndexFlatIP,
    bm25_index: BM25Okapi,
    chunks: list[dict],
) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(faiss_index, str(FAISS_PATH))

    with BM25_PATH.open("wb") as f:
        pickle.dump(bm25_index, f)

    with METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def main() -> None:
    chunks = load_chunks(CHUNKS_PATH)

    print(f"Loaded chunks: {len(chunks)}")

    print("Building BM25 keyword index...")
    bm25_index = build_bm25(chunks)

    print("Building FAISS vector index...")
    faiss_index = build_faiss_index(chunks)

    save_outputs(faiss_index, bm25_index, chunks)

    print("Indexes saved:")
    print(f"- {FAISS_PATH}")
    print(f"- {BM25_PATH}")
    print(f"- {METADATA_PATH}")


if __name__ == "__main__":
    main()