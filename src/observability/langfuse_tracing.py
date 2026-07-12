"""
Langfuse tracing helpers for thesis evaluation of the MDR/MDCG RAG app.

Langfuse is used here to evaluate traceability, source-supported answers,
retrieval relevance, grounding/source support, and answer quality. It is
intentionally optional: missing credentials or SDK import errors disable
tracing without changing the RAG behavior.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
import os
from typing import Any

from dotenv import load_dotenv


TRACE_NAME = "mdr_mdcg_rag_question"


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_present(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def chunk_trace_metadata(
    chunk: dict[str, Any],
    *,
    selected_for_final_context: bool | None = None,
    used_in_final_answer: bool | None = None,
) -> dict[str, Any]:
    """Return compact, source-focused chunk metadata for Langfuse traces."""
    return {
        "source_type": chunk.get("source_type"),
        "document_name": _first_present(
            chunk.get("document_title"),
            chunk.get("document_name"),
            chunk.get("reference"),
            chunk.get("document_id"),
        ),
        "page_number": _first_present(chunk.get("page_number"), chunk.get("page_start")),
        "page_start": chunk.get("page_start"),
        "page_end": chunk.get("page_end"),
        "article_or_annex_or_section": _first_present(
            chunk.get("article"),
            chunk.get("annex"),
            chunk.get("section"),
            chunk.get("title"),
            chunk.get("reference"),
        ),
        "chunk_id": chunk.get("chunk_id"),
        "retrieval_score": _safe_float(
            _first_present(
                chunk.get("hybrid_score"),
                chunk.get("retrieval_score"),
                chunk.get("bm25_score"),
                chunk.get("dense_score"),
            )
        ),
        "rerank_score": _safe_float(chunk.get("rerank_score")),
        "selected_for_final_context": selected_for_final_context,
        "used_in_final_answer": used_in_final_answer,
    }


def chunks_trace_metadata(
    chunks: list[dict[str, Any]],
    *,
    selected_chunk_ids: set[str] | None = None,
    used_chunk_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    selected_chunk_ids = selected_chunk_ids or set()
    used_chunk_ids = used_chunk_ids or set()

    return [
        chunk_trace_metadata(
            chunk,
            selected_for_final_context=(
                chunk.get("chunk_id") in selected_chunk_ids
                if selected_chunk_ids
                else None
            ),
            used_in_final_answer=(
                chunk.get("chunk_id") in used_chunk_ids
                if used_chunk_ids
                else None
            ),
        )
        for chunk in chunks
    ]


class LangfuseSpan(AbstractContextManager):
    def __init__(
        self,
        parent: Any,
        *,
        name: str,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._parent = parent
        self._name = name
        self._input = input
        self._metadata = metadata or {}
        self._context = None
        self._span = None
        self._ended = False

    def __enter__(self) -> "LangfuseSpan":
        if self._parent is None:
            return self

        try:
            if hasattr(self._parent, "start_as_current_observation"):
                try:
                    self._context = self._parent.start_as_current_observation(
                        as_type="span",
                        name=self._name,
                        input=self._input,
                        metadata=self._metadata,
                    )
                except TypeError:
                    self._context = self._parent.start_as_current_observation(
                        as_type="span",
                        name=self._name,
                        input=self._input,
                    )
                self._span = self._context.__enter__()
                if self._metadata and hasattr(self._span, "update"):
                    self._span.update(metadata=self._metadata)
            elif hasattr(self._parent, "span"):
                self._span = self._parent.span(
                    name=self._name,
                    input=self._input,
                    metadata=self._metadata,
                )
        except Exception:
            self._context = None
            self._span = None
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc is not None:
            self.end(metadata={"error": str(exc)})
        elif not self._ended:
            self.end()
        return None

    def end(
        self,
        *,
        output: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self._span is None or self._ended:
            return

        kwargs: dict[str, Any] = {}
        if output is not None:
            kwargs["output"] = output
        if metadata:
            kwargs["metadata"] = metadata

        try:
            if self._context is not None:
                if kwargs and hasattr(self._span, "update"):
                    self._span.update(**kwargs)
                self._context.__exit__(None, None, None)
            else:
                self._span.end(**kwargs)
        except Exception:
            pass
        self._ended = True


class RagLangfuseTrace:
    def __init__(
        self,
        root: Any = None,
        client: Any = None,
        context: Any = None,
        *,
        enabled: bool = False,
        legacy_trace: Any = None,
    ) -> None:
        self._root = root
        self._client = client
        self._context = context
        self._legacy_trace = legacy_trace
        self.enabled = enabled and (root is not None or legacy_trace is not None)

    def span(
        self,
        name: str,
        *,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> LangfuseSpan:
        if not self.enabled:
            return LangfuseSpan(None, name=name, input=input, metadata=metadata)
        parent = self._root or self._legacy_trace
        return LangfuseSpan(parent, name=name, input=input, metadata=metadata)

    def update(
        self,
        *,
        input: Any = None,
        output: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return

        kwargs: dict[str, Any] = {}
        if input is not None:
            kwargs["input"] = input
        if output is not None:
            kwargs["output"] = output
        if metadata:
            kwargs["metadata"] = metadata

        try:
            target = self._root or self._legacy_trace
            target.update(**kwargs)
        except Exception:
            pass

    def score(self, name: str, value: float | None, *, comment: str | None = None) -> None:
        if not self.enabled or value is None:
            return

        try:
            if hasattr(self._client, "score_current_trace"):
                self._client.score_current_trace(
                    name=name,
                    value=float(value),
                    data_type="NUMERIC",
                    comment=comment,
                )
            elif hasattr(self._legacy_trace, "score"):
                self._legacy_trace.score(name=name, value=float(value), comment=comment)
        except Exception:
            pass

    def flush(self) -> None:
        if not self.enabled:
            return

        if self._context is not None:
            try:
                self._context.__exit__(None, None, None)
            except Exception:
                pass
            self._context = None

        if self._client is None:
            return

        try:
            self._client.flush()
        except Exception:
            pass


def create_rag_trace(
    *,
    question: str,
    settings: dict[str, Any],
    selected_documents: dict[str, Any] | None = None,
) -> RagLangfuseTrace:
    load_dotenv()

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST")
    base_url = os.getenv("LANGFUSE_BASE_URL") or host

    if not public_key or not secret_key:
        return RagLangfuseTrace()

    try:
        from langfuse import Langfuse
    except Exception:
        return RagLangfuseTrace()

    try:
        if base_url:
            os.environ.setdefault("LANGFUSE_BASE_URL", base_url)

        client_kwargs = {
            "public_key": public_key,
            "secret_key": secret_key,
        }
        if base_url:
            client_kwargs["base_url"] = base_url

        client = Langfuse(**client_kwargs)
        trace_input = {
            "user_question": question,
            "selected_documents": selected_documents or {},
            "settings": settings,
        }
        trace_metadata = {
            "purpose": "thesis_evaluation",
            "evaluation_focus": [
                "traceability",
                "source_supported_answers",
                "retrieval_relevance",
                "grounding_source_support",
                "answer_quality",
            ],
        }

        if hasattr(client, "start_as_current_observation"):
            try:
                context = client.start_as_current_observation(
                    as_type="span",
                    name=TRACE_NAME,
                    input=trace_input,
                    metadata=trace_metadata,
                )
            except TypeError:
                context = client.start_as_current_observation(
                    as_type="span",
                    name=TRACE_NAME,
                    input=trace_input,
                )
            root = context.__enter__()
            if hasattr(root, "update"):
                root.update(metadata=trace_metadata)
            return RagLangfuseTrace(
                root=root,
                client=client,
                context=context,
                enabled=True,
            )

        trace = client.trace(
            name=TRACE_NAME,
            input={
                "user_question": question,
                "selected_documents": selected_documents or {},
                "settings": settings,
            },
            metadata=trace_metadata,
        )
    except Exception:
        return RagLangfuseTrace()

    return RagLangfuseTrace(client=client, enabled=True, legacy_trace=trace)
