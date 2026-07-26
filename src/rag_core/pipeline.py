"""
Callable MDR/MDCG RAG pipeline used by the Streamlit UI.

This module wraps the existing project functions instead of replacing the CLI
or duplicating retrieval/generation logic.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time
from typing import Any


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evidence_gate import check_evidence_gate
from external_access_check import (
    ExternalAccessMonitor,
    external_access_check_enabled,
)
from fallback_answer import generate_source_fallback_answer
from generate_answer import GEMINI_MODEL as ANSWER_MODEL
from generate_answer import generate_answer
from grounding_check import check_grounding, grounding_is_supported
from guardrails import (
    validate_answer_text,
    validate_question,
    validate_retrieved_chunks,
)
from llm_rerank import GEMINI_MODEL as RERANK_MODEL
from llm_rerank import llm_rerank_chunks
from llm_utils import is_quota_error
from observability.langfuse_tracing import (
    chunks_trace_metadata,
    create_rag_trace,
)
from retrieve import retrieve
from rag_core.usage_logger import log_request


MAX_RETRIEVAL_ATTEMPTS = 3
DEFAULT_TOP_K = 18
DEFAULT_MDR_RERANK_K = 5
DEFAULT_MDCG_RERANK_K = 6


def _setting(settings: dict | None, name: str, default: Any) -> Any:
    if not settings:
        return default
    return settings.get(name, default)


def _chunk_source(chunk: dict) -> dict:
    return {
        "chunk_id": chunk.get("chunk_id"),
        "source_type": chunk.get("source_type"),
        "document_id": chunk.get("document_id"),
        "document_title": chunk.get("document_title"),
        "reference": chunk.get("reference"),
        "title": chunk.get("title"),
        "source": chunk.get("source"),
        "page_start": chunk.get("page_start"),
        "page_end": chunk.get("page_end"),
    }


def _evidence_item(chunk: dict) -> dict:
    item = _chunk_source(chunk)
    item.update({
        "hybrid_score": chunk.get("hybrid_score"),
        "retrieval_source": chunk.get("retrieval_source"),
        "text": chunk.get("text", ""),
    })
    return item


def _trace_settings(
    *,
    settings: dict | None,
    top_k: int,
    enable_reranking: bool,
    enable_evidence_gate: bool,
    enable_grounding_check: bool,
) -> dict:
    return {
        "selected_documents": (settings or {}).get("selected_documents"),
        "reranking_enabled": enable_reranking,
        "evidence_gate_enabled": enable_evidence_gate,
        "grounding_check_enabled": enable_grounding_check,
        "top_k": top_k,
        "answer_model_name": ANSWER_MODEL,
        "rerank_model_name": RERANK_MODEL if enable_reranking else None,
        "raw_settings": settings or {},
    }


def _clamp_score(value: float | None) -> float | None:
    if value is None:
        return None
    return max(0.0, min(1.0, float(value)))


def _score_summary(answer: str, sources: list, evidence: list, metadata: dict) -> dict:
    scores = {
        "retrieval_relevance_score": None,
        "retrieval_relevance_reason": "No retrieved chunks with hybrid retrieval scores were available.",
        "source_support_score": None,
        "source_support_reason": "Grounding/source-support check was not run.",
        "traceability_score": None,
        "traceability_reason": "No retrieved chunks were available for traceability scoring.",
        "citation_completeness_score": None,
        "citation_completeness_reason": (
            "No claim-level citation completeness scorer is implemented yet."
        ),
        "answer_completeness_score": None,
        "answer_completeness_reason": (
            "No answer completeness scorer is implemented yet."
        ),
        "practical_usefulness_score": None,
        "practical_usefulness_reason": (
            "No practical usefulness scorer is implemented yet."
        ),
    }

    retrieval_scores = [
        item.get("hybrid_score")
        for item in evidence
        if isinstance(item.get("hybrid_score"), (int, float))
    ]
    if retrieval_scores:
        scores["retrieval_relevance_score"] = _clamp_score(
            sum(retrieval_scores) / len(retrieval_scores)
        )
        scores["retrieval_relevance_reason"] = (
            "Computed as the mean normalized hybrid retrieval score for final evidence chunks."
        )

    grounding_status = metadata.get("grounding_status")
    if isinstance(grounding_status, str):
        first_line = grounding_status.strip().splitlines()[0].upper() if grounding_status.strip() else ""
        if first_line.startswith("SUPPORTED"):
            scores["source_support_score"] = 1.0
            scores["source_support_reason"] = "Grounding checker returned SUPPORTED."
        elif first_line.startswith("PARTIALLY_SUPPORTED"):
            scores["source_support_score"] = 0.5
            scores["source_support_reason"] = "Grounding checker returned PARTIALLY_SUPPORTED."
        elif first_line.startswith("UNSUPPORTED"):
            scores["source_support_score"] = 0.0
            scores["source_support_reason"] = "Grounding checker returned UNSUPPORTED."

    if evidence:
        required_fields = [
            "source_type",
            "document_title",
            "page_start",
            "title",
            "chunk_id",
        ]
        available = 0
        total = len(evidence) * len(required_fields)
        for item in evidence:
            for field in required_fields:
                if item.get(field) not in (None, ""):
                    available += 1
        scores["traceability_score"] = _clamp_score(available / total if total else None)
        scores["traceability_reason"] = (
            "Computed from source type, document name, page, section/title, and chunk ID metadata completeness."
        )

    if answer and sources:
        scores["citation_completeness_reason"] = (
            "Sources were packaged with the answer, but claim-level citation coverage is not scored yet."
        )

    return scores


def _chunk_text_char_count(chunks: list[dict]) -> int:
    return sum(len(chunk.get("text", "") or "") for chunk in chunks)


def _finalize_langfuse_trace(
    *,
    tracer: Any,
    question: str,
    answer: str,
    sources: list,
    evidence: list,
    metadata: dict,
) -> None:
    if not tracer:
        return

    scores = _score_summary(answer, sources, evidence, metadata)
    metadata["evaluation_scores"] = scores

    tracer.update(
        output={
            "final_answer": answer,
            "citations": sources,
            "grounding_status": metadata.get("grounding_status"),
        },
        metadata={
            "evaluation_scores": scores,
            "evidence_count": len(evidence),
            "source_count": len(sources),
        },
    )

    for score_name in [
        "retrieval_relevance_score",
        "source_support_score",
        "traceability_score",
        "citation_completeness_score",
        "answer_completeness_score",
        "practical_usefulness_score",
    ]:
        reason_name = score_name.replace("_score", "_reason")
        tracer.score(score_name, scores.get(score_name), comment=scores.get(reason_name))

    tracer.flush()


def _dedupe_sources(chunks: list[dict]) -> list[dict]:
    sources = []
    seen = set()

    for chunk in chunks:
        key = (
            chunk.get("source_type"),
            chunk.get("document_id"),
            chunk.get("reference"),
            chunk.get("document_title"),
        )
        if key in seen:
            continue

        seen.add(key)
        sources.append({
            "source_type": chunk.get("source_type"),
            "document_id": chunk.get("document_id"),
            "document_title": chunk.get("document_title"),
            "reference": chunk.get("reference"),
            "source": chunk.get("source"),
        })

    return sources


def _limit_chunks(mdcg_chunks: list[dict], mdr_chunks: list[dict], top_k: int) -> list[dict]:
    combined = mdcg_chunks + mdr_chunks
    if top_k <= 0:
        return combined
    return combined[:top_k]


def _domain_gate(question: str) -> dict:
    """
    Placeholder domain-gate wrapper.

    src/domain_gate.py currently has no callable implementation, so validation is
    handled by guardrails.validate_question. This keeps the pipeline stage visible
    in metadata without inventing new blocking behavior.
    """
    return {
        "passed": True,
        "reason": "not_configured",
        "message": "No separate domain gate is currently configured.",
    }


def _failure_result(
    *,
    question: str,
    answer: str,
    sources: list,
    evidence: list,
    metadata: dict,
    start_time: float,
    tracer: Any = None,
    external_monitor: ExternalAccessMonitor | None = None,
) -> dict:
    runtime_seconds = round(time.perf_counter() - start_time, 3)
    metadata["runtime_seconds"] = runtime_seconds
    if external_monitor:
        metadata["external_access_check"] = external_monitor.summary()

    if tracer:
        with tracer.span(
            "final_answer_packaging",
            input={
                "question": question,
                "answer": answer,
                "source_count": len(sources),
                "evidence_count": len(evidence),
            },
        ) as span:
            span.end(output={
                "final_answer": answer,
                "citations": sources,
                "grounding_status": metadata.get("grounding_status"),
            })

        if external_monitor:
            external_monitor.trace_summary()

        _finalize_langfuse_trace(
            tracer=tracer,
            question=question,
            answer=answer,
            sources=sources,
            evidence=evidence,
            metadata=metadata,
        )

    result = {
        "answer": answer,
        "sources": sources,
        "evidence": evidence,
        "metadata": metadata,
    }
    log_request(
        question=question,
        answer=answer,
        sources=sources,
        metadata=metadata,
        runtime_seconds=runtime_seconds,
    )
    return result


def answer_question(question: str, settings: dict | None = None) -> dict:
    monitor = ExternalAccessMonitor(
        enabled=external_access_check_enabled(settings),
        question=question,
    )

    with monitor:
        result = _answer_question_impl(
            question,
            settings=settings,
            external_monitor=monitor,
        )

    if isinstance(result, dict):
        result.setdefault("metadata", {})["external_access_check"] = monitor.summary()

    return result


def _answer_question_impl(
    question: str,
    settings: dict | None = None,
    external_monitor: ExternalAccessMonitor | None = None,
) -> dict:
    """
    Returns:
    {
      "answer": str,
      "sources": list,
      "evidence": list,
      "metadata": dict
    }
    """
    start_time = time.perf_counter()
    metadata: dict[str, Any] = {
        "settings": settings or {},
        "selected_documents": {},
        "answerability_status": "not_run",
        "grounding_status": "not_run",
        "attempts": [],
    }

    enable_reranking = bool(_setting(settings, "enable_reranking", False))
    enable_evidence_gate = bool(_setting(settings, "enable_evidence_gate", False))
    enable_grounding_check = bool(_setting(settings, "enable_grounding_check", False))
    top_k = int(_setting(settings, "top_k", DEFAULT_TOP_K))
    trace_settings = _trace_settings(
        settings=settings,
        top_k=top_k,
        enable_reranking=enable_reranking,
        enable_evidence_gate=enable_evidence_gate,
        enable_grounding_check=enable_grounding_check,
    )
    tracer = create_rag_trace(
        question=question,
        settings=trace_settings,
        selected_documents=metadata["selected_documents"],
    )
    if external_monitor:
        external_monitor.tracer = tracer

    question = validate_question(question)
    domain_gate_result = _domain_gate(question)
    metadata["domain_gate"] = domain_gate_result

    if not domain_gate_result["passed"]:
        return _failure_result(
            question=question,
            answer=domain_gate_result["message"],
            sources=[],
            evidence=[],
            metadata=metadata,
            start_time=start_time,
            tracer=tracer,
            external_monitor=external_monitor,
        )

    last_sources: list = []
    last_evidence: list = []

    for attempt in range(1, MAX_RETRIEVAL_ATTEMPTS + 1):
        retrieval_result = retrieve(query=question, attempt=attempt, tracer=tracer)
        mdr_chunks = retrieval_result.get("mdr_chunks", [])
        mdcg_chunks = retrieval_result.get("mdcg_chunks", [])

        validate_retrieved_chunks(mdr_chunks, expected_source_type="MDR")
        validate_retrieved_chunks(
            mdcg_chunks,
            expected_source_type="MDCG",
            allow_empty=True,
        )

        with tracer.span(
            "reranking",
            input={
                "enabled": enable_reranking,
                "mdcg_chunks": chunks_trace_metadata(mdcg_chunks),
                "mdr_chunks": chunks_trace_metadata(mdr_chunks),
                "model_name": RERANK_MODEL if enable_reranking else None,
            },
        ) as span:
            if enable_reranking:
                mdcg_chunks = llm_rerank_chunks(
                    question,
                    mdcg_chunks,
                    top_k=DEFAULT_MDCG_RERANK_K,
                )
                mdr_chunks = llm_rerank_chunks(
                    question,
                    mdr_chunks,
                    top_k=DEFAULT_MDR_RERANK_K,
                )
            selected_after_rerank = {
                chunk.get("chunk_id")
                for chunk in (mdcg_chunks + mdr_chunks)
                if chunk.get("chunk_id")
            }
            span.end(output={
                "enabled": enable_reranking,
                "mdcg_chunks": chunks_trace_metadata(
                    mdcg_chunks,
                    selected_chunk_ids=selected_after_rerank,
                ),
                "mdr_chunks": chunks_trace_metadata(
                    mdr_chunks,
                    selected_chunk_ids=selected_after_rerank,
                ),
            })

        with tracer.span(
            "final_context_selection",
            input={
                "top_k": top_k,
                "mdcg_candidate_count": len(mdcg_chunks),
                "mdr_candidate_count": len(mdr_chunks),
            },
        ) as span:
            combined_chunks = _limit_chunks(mdcg_chunks, mdr_chunks, top_k)
            final_chunk_ids = {
                chunk.get("chunk_id")
                for chunk in combined_chunks
                if chunk.get("chunk_id")
            }
            span.end(output={
                "final_context_count": len(combined_chunks),
                "chunks": chunks_trace_metadata(
                    combined_chunks,
                    selected_chunk_ids=final_chunk_ids,
                ),
            })
        last_sources = _dedupe_sources(combined_chunks)
        last_evidence = [_evidence_item(chunk) for chunk in combined_chunks]

        attempt_metadata = {
            "attempt": attempt,
            "selected_mdr_documents": retrieval_result.get("selected_mdr_documents", []),
            "selected_mdcg_documents": retrieval_result.get("selected_mdcg_documents", []),
            "mdr_chunk_count": len(mdr_chunks),
            "mdcg_chunk_count": len(mdcg_chunks),
            "final_evidence_count": len(combined_chunks),
        }
        metadata["attempts"].append(attempt_metadata)
        metadata["selected_documents"] = {
            "mdr": attempt_metadata["selected_mdr_documents"],
            "mdcg": attempt_metadata["selected_mdcg_documents"],
        }
        tracer.update(input={
            "user_question": question,
            "selected_documents": metadata["selected_documents"],
            "settings": trace_settings,
        })

        if enable_evidence_gate:
            try:
                gate_result = check_evidence_gate(question, combined_chunks)
            except Exception as error:
                if not is_quota_error(error):
                    raise
                gate_result = {
                    "passed": True,
                    "reason": "quota_skip",
                    "message": "Skipped because Gemini quota is exhausted.",
                }

            metadata["answerability_status"] = gate_result

            if not gate_result["passed"]:
                if attempt < MAX_RETRIEVAL_ATTEMPTS:
                    continue
                return _failure_result(
                    question=question,
                    answer=gate_result["message"],
                    sources=last_sources,
                    evidence=last_evidence,
                    metadata=metadata,
                    start_time=start_time,
                    tracer=tracer,
                    external_monitor=external_monitor,
                )
        else:
            metadata["answerability_status"] = {
                "passed": True,
                "reason": "disabled",
                "message": "Evidence gate disabled.",
            }

        with tracer.span(
            "answer_generation",
            input={
                "question": question,
                "model_name": ANSWER_MODEL,
                "chunk_count": len(combined_chunks),
                "context_text_chars": _chunk_text_char_count(combined_chunks),
                "chunks": chunks_trace_metadata(
                    combined_chunks,
                    selected_chunk_ids=final_chunk_ids,
                ),
            },
        ) as span:
            try:
                answer = generate_answer(question, combined_chunks)
                span.end(output={"answer": answer, "generation_status": "generated"})
            except Exception as error:
                if not is_quota_error(error):
                    raise
                answer = generate_source_fallback_answer(question, combined_chunks)
                metadata["generation_status"] = "quota_fallback"
                metadata["generation_error"] = {
                    "type": type(error).__name__,
                    "message": str(error),
                    "context_chunk_count": len(combined_chunks),
                    "context_text_chars": _chunk_text_char_count(combined_chunks),
                }
                span.end(output={
                    "answer": answer,
                    "generation_status": "quota_fallback",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                })
                return _failure_result(
                    question=question,
                    answer=answer,
                    sources=last_sources,
                    evidence=last_evidence,
                    metadata=metadata,
                    start_time=start_time,
                    tracer=tracer,
                    external_monitor=external_monitor,
                )

        if not validate_answer_text(answer):
            return _failure_result(
                question=question,
                answer="Answer blocked because it contains unsupported wording.",
                sources=last_sources,
                evidence=last_evidence,
                metadata=metadata,
                start_time=start_time,
                tracer=tracer,
                external_monitor=external_monitor,
            )

        with tracer.span(
            "grounding_source_support_check",
            input={
                "enabled": enable_grounding_check,
                "question": question,
                "answer": answer,
                "chunks": chunks_trace_metadata(
                    combined_chunks,
                    selected_chunk_ids=final_chunk_ids,
                ),
            },
        ) as span:
            if enable_grounding_check:
                try:
                    grounding_result = check_grounding(question, answer, combined_chunks)
                except Exception as error:
                    if not is_quota_error(error):
                        raise
                    grounding_result = "SUPPORTED_QUOTA_SKIPPED"

                metadata["grounding_status"] = grounding_result
                span.end(output={"grounding_status": grounding_result})
            else:
                metadata["grounding_status"] = "disabled"
                span.end(output={
                    "grounding_status": "disabled",
                    "reason": "Grounding check disabled in active settings.",
                })

        if enable_grounding_check and not grounding_is_supported(grounding_result):
            if attempt < MAX_RETRIEVAL_ATTEMPTS:
                continue
            return _failure_result(
                question=question,
                answer="Answer blocked by grounding checker.",
                sources=last_sources,
                evidence=last_evidence,
                metadata=metadata,
                start_time=start_time,
                tracer=tracer,
                external_monitor=external_monitor,
            )

        metadata["generation_status"] = "generated"
        return _failure_result(
            question=question,
            answer=answer,
            sources=last_sources,
            evidence=last_evidence,
            metadata=metadata,
            start_time=start_time,
            tracer=tracer,
            external_monitor=external_monitor,
        )

    return _failure_result(
        question=question,
        answer="I could not find sufficient evidence in the available MDR and MDCG sources to answer this question.",
        sources=last_sources,
        evidence=last_evidence,
        metadata=metadata,
        start_time=start_time,
        tracer=tracer,
        external_monitor=external_monitor,
    )
