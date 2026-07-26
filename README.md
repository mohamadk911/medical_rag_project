# MDR / MDCG Regulatory RAG

This project is a thesis prototype for answering regulatory questions using the
EU Medical Device Regulation (MDR 2017/745) and downloaded Medical Device
Coordination Group (MDCG) guidance documents. The system is implemented as a
local retrieval-augmented generation (RAG) pipeline with a Streamlit interface
for testing and demonstration.

The prototype is intended to support regulatory analysis by retrieving relevant
MDR and MDCG evidence, generating an answer with Gemini, and exposing the
retrieved sources so the answer can be checked against the underlying documents.
It is not a replacement for official legal or regulatory review.

## Main Features

- PDF text extraction from MDR and MDCG source documents using PyMuPDF.
- Legal-structure chunking for the MDR, based primarily on Articles and Annexes.
- Full-document or overlapping large-text chunking for MDCG guidance documents.
- Document-level catalog retrieval followed by chunk-level evidence retrieval.
- Hybrid retrieval using BM25 keyword search and dense FAISS vector search.
- Dense embeddings generated with `BAAI/bge-small-en-v1.5` through FastEmbed.
- Gemini-based answer generation from retrieved evidence.
- Optional LLM reranking from the Streamlit sidebar.
- Langfuse tracing for observability and evaluation metadata.
- Runtime external-access monitoring to verify whether live web search or
  unexpected external APIs were used during a run.

## Architecture Overview

The implemented workflow has three main phases:

1. **Knowledge preparation**
   - `src/chunk_documents.py` extracts text from the MDR PDF and local MDCG PDFs.
   - MDR content is split using detected `Article` and `ANNEX` headings.
   - MDCG documents are kept as one chunk when small enough, or split into large
     overlapping chunks when necessary.
   - `src/catalog_builder.py` creates a document-level catalog containing the MDR
     and one entry per MDCG document.

2. **Indexing**
   - `src/build_index.py` builds two index levels:
     - a document-level catalog index;
     - a chunk-level evidence index.
   - Each level stores:
     - a BM25 index;
     - a FAISS dense-vector index;
     - associated metadata.
   - Both retrieval levels use the same hybrid weighting:
     `0.6 BM25 + 0.4 dense retrieval`.

3. **Question answering**
   - `src/rag_core/pipeline.py` validates the question and calls the retrieval
     pipeline.
   - `src/retrieve.py` performs catalog-guided retrieval and chunk-level evidence
     retrieval.
   - Retrieved chunks are optionally reranked.
   - Gemini generates the final answer using only the selected evidence context.
   - The Streamlit UI displays the answer, sources, retrieved chunks, and metadata.

## Retrieval Method

The retrieval implementation is two-level and hybrid.

First, the system searches the document catalog to identify relevant MDR and MDCG
documents. The catalog itself is indexed with BM25 and FAISS dense retrieval.

Second, the system searches the chunk-level evidence index. MDR chunks and MDCG
chunks are retrieved as separate evidence tracks. BM25 supports exact regulatory
terminology matching, while dense retrieval supports semantic similarity.

For MDCG evidence, catalog-guided chunk retrieval is supplemented by direct
hybrid retrieval across all MDCG chunks, and both result sets are merged before
reranking and final context selection.

The pipeline allows up to three retrieval attempts. If optional evidence-gate or
grounding checks reject the evidence or generated answer, the next attempt uses a
different batch of catalog-selected MDCG documents and a different slice of
directly retrieved MDCG chunks. This retry mechanism prevents an infinite loop
while giving the system additional chances to find sufficient evidence.

## Streamlit Workflow

The standard Streamlit workflow is designed for stable professor/testing use.
It enables:

- question input;
- top-k control;
- optional reranking;
- answer generation;
- source and evidence display;
- metadata display.

The evidence-gate and grounding-verification modules are implemented in the
backend pipeline as optional validation components, but they are disabled in the
standard Streamlit workflow to reduce latency and avoid additional LLM quota
dependency.

## Observability and Logging

The project records runtime information for evaluation:

- `logs/usage.log` stores request and response metadata.
- `logs/external_access_check.log` stores whether live web search or unexpected
  external API calls were detected.
- Langfuse tracing records retrieval, reranking, answer generation, source
  support metadata, and external-access summaries when Langfuse credentials are
  configured.

Expected external services are Gemini for answer generation and Langfuse for
observability. The RAG retrieval itself uses local indexes and does not require
live web search.

## Project Structure

```text
src/
  build_index.py              Build BM25 and FAISS indexes
  catalog_builder.py          Build document-level catalog
  catalog_retrieve.py         Stage 1 catalog retrieval
  chunk_documents.py          Extract and chunk MDR/MDCG PDFs
  retrieve.py                 Stage 2 hybrid evidence retrieval
  generate_answer.py          Gemini answer generation
  llm_rerank.py               Optional Gemini reranking
  evidence_gate.py            Optional evidence sufficiency check
  grounding_check.py          Optional grounding verification
  external_access_check.py    Runtime external-access monitor
  rag_core/pipeline.py        Streamlit-callable RAG pipeline
  ui/streamlit_app.py         Browser UI

prompts/                       Prompt templates
data/raw/                      Source PDFs
data/processed/                Processed catalog and chunk JSON files
indexes/                       Local BM25, FAISS, and metadata indexes
logs/                          Runtime logs
tests/                         Unit tests
```

## Setup

Create and activate a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file with the required credentials:

```bash
GEMINI_API_KEY=your_gemini_key_here
STREAMLIT_APP_PASSWORD=choose_a_password_here

# Optional Langfuse tracing
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_BASE_URL=https://cloud.langfuse.com

# Optional, enabled by default if omitted
EXTERNAL_ACCESS_CHECK_ENABLED=true
```

Do not commit real API keys or passwords.

## Building Data and Indexes

If processed data and indexes already exist, this step can be skipped. To rebuild
from local source PDFs:

```bash
python src/chunk_documents.py
python src/catalog_builder.py
python src/build_index.py
```

If MDCG source documents need to be downloaded again, use:

```bash
python src/download_mdcg.py
python src/download_mdcg_catalog.py
```

The indexing step creates the catalog-level and chunk-level BM25/FAISS indexes
used by the retrieval pipeline.

## Running the Streamlit Demo

```bash
streamlit run src/ui/streamlit_app.py --server.port 8501
```

Then open:

```text
http://localhost:8501
```

The UI requires the password configured in `STREAMLIT_APP_PASSWORD`.

## Sharing With ngrok

Keep Streamlit running locally:

```bash
streamlit run src/ui/streamlit_app.py --server.port 8501
```

In another terminal:

```bash
ngrok http 8501
```

Share the public `https://...ngrok-free.app` URL and the
`STREAMLIT_APP_PASSWORD` with the tester. Anyone with both can use the running
application and the configured Gemini quota while the tunnel remains active.

## Tests

Install `pytest` if it is not already available, then run the available unit
tests with:

```bash
pip install pytest
python -m pytest
```

## Current Limitations

- The system is a thesis prototype, not a certified regulatory decision system.
- MDCG chunk page ranges are stored at document level rather than as exact
  per-chunk page ranges.
- Some optional validation modules require additional Gemini calls and are
  disabled in the default Streamlit workflow.
- Generated answers should always be checked against the displayed MDR and MDCG
  sources.
