"""
Runtime verification for live web-search or unexpected external access.

This module is enabled by default and is intended for thesis verification runs.
It records whether the RAG runtime used live web search, and whether code appears
to configure Gemini with a Google Search/Grounding tool.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


ROOT_DIR = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT_DIR / "logs" / "external_access_check.log"

TRUTHY = {"1", "true", "yes", "on"}

SEARCH_PROVIDERS = {
    "google_search": ("google.com", "customsearch.googleapis.com"),
    "bing_search": ("bing.com", "api.bing.microsoft.com"),
    "duckduckgo_search": ("duckduckgo.com", "api.duckduckgo.com"),
    "serpapi": ("serpapi.com",),
    "brave_search": ("api.search.brave.com",),
    "tavily": ("tavily.com", "api.tavily.com"),
}

GEMINI_HOSTS = ("generativelanguage.googleapis.com",)
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", ""}

GEMINI_SEARCH_TOOL_MARKERS = (
    "google_search",
    "googlesearch",
    "google_search_retrieval",
    "googlesearchretrieval",
    "grounding_metadata",
    "groundingmetadata",
)


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in TRUTHY


def external_access_check_enabled(settings: dict | None = None) -> bool:
    if settings and "enable_external_access_check" in settings:
        return bool(settings["enable_external_access_check"])
    return env_flag("EXTERNAL_ACCESS_CHECK_ENABLED", True)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_log(record: dict[str, Any]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def provider_for_url(url: str) -> str | None:
    host = (urlparse(url).hostname or "").lower()
    for provider, host_markers in SEARCH_PROVIDERS.items():
        if any(marker in host for marker in host_markers):
            return provider
    return None


def is_external_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    return (parsed.hostname or "").lower() not in LOCAL_HOSTS


def is_allowed_external_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    langfuse_host = (urlparse(os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST") or "").hostname or "").lower()
    if any(marker in host for marker in GEMINI_HOSTS):
        return True
    return bool(langfuse_host and host == langfuse_host)


def query_from_url(url: str) -> str | None:
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    for key in ("q", "query", "search"):
        values = params.get(key)
        if values:
            return values[0]
    return None


def returned_urls_from_response(response: Any) -> list[str]:
    urls: list[str] = []

    try:
        data = response.json()
    except Exception:
        data = None

    if isinstance(data, dict):
        for item in data.get("items", []) or data.get("results", []) or []:
            if isinstance(item, dict):
                url = item.get("link") or item.get("url")
                if url:
                    urls.append(str(url))

    return urls[:20]


def verify_gemini_grounding_configuration() -> dict[str, Any]:
    """
    Static verification: current Gemini calls do not pass tool config objects, so
    this checks the source for known Google Search/Grounding tool markers.
    """
    matches = []
    for path in sorted((ROOT_DIR / "src").glob("*.py")):
        if path.name == "external_access_check.py":
            continue

        text = path.read_text(encoding="utf-8").lower()
        if any(marker in text for marker in GEMINI_SEARCH_TOOL_MARKERS):
            matches.append(str(path.relative_to(ROOT_DIR)))

    return {
        "gemini_google_search_tool_configured": bool(matches),
        "evidence": matches,
    }


class ExternalAccessMonitor:
    def __init__(
        self,
        *,
        enabled: bool,
        question: str,
        tracer: Any = None,
    ) -> None:
        self.enabled = enabled
        self.question = question
        self.tracer = tracer
        self.web_search_calls: list[dict[str, Any]] = []
        self.external_api_calls: list[dict[str, Any]] = []
        self.gemini_grounding = verify_gemini_grounding_configuration()
        self._original_request = None
        self._original_httpx_request = None

    def __enter__(self) -> "ExternalAccessMonitor":
        if not self.enabled:
            return self

        try:
            import requests
        except Exception:
            requests = None

        monitor = self

        if requests is not None:
            self._original_request = requests.sessions.Session.request

            def checked_request(session, method, url, **kwargs):
                response = monitor._original_request(session, method, url, **kwargs)
                monitor.record_http_call(method=method, url=str(url), response=response)
                return response

            requests.sessions.Session.request = checked_request

        try:
            import httpx
        except Exception:
            return self

        self._original_httpx_request = httpx.Client.request

        def checked_httpx_request(client, method, url, **kwargs):
            response = monitor._original_httpx_request(client, method, url, **kwargs)
            monitor.record_http_call(method=method, url=str(url), response=response)
            return response

        httpx.Client.request = checked_httpx_request
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._original_request is not None:
            try:
                import requests
                requests.sessions.Session.request = self._original_request
            except Exception:
                pass

        if self._original_httpx_request is not None:
            try:
                import httpx
                httpx.Client.request = self._original_httpx_request
            except Exception:
                pass

        if self.enabled:
            append_log(self.summary())

    def record_http_call(self, *, method: str, url: str, response: Any = None) -> None:
        if not is_external_url(url):
            return

        record = {
            "timestamp": utc_timestamp(),
            "method": method.upper(),
            "url": url,
            "allowed_provider": is_allowed_external_url(url),
        }

        if not record["allowed_provider"]:
            self.external_api_calls.append(record)

        provider = provider_for_url(url)
        if not provider:
            return

        returned_urls = returned_urls_from_response(response)
        search_record = {
            "timestamp": record["timestamp"],
            "provider": provider,
            "query": query_from_url(url),
            "returned_urls": returned_urls,
            "result_count": len(returned_urls),
        }
        self.web_search_calls.append(search_record)
        self._trace_web_search(search_record)

    def _trace_web_search(self, search_record: dict[str, Any]) -> None:
        if not self.tracer:
            return

        with self.tracer.span(
            "web_search",
            input={
                "provider": search_record["provider"],
                "query": search_record["query"],
            },
            metadata={"timestamp": search_record["timestamp"]},
        ) as span:
            span.end(output={
                "returned_urls": search_record["returned_urls"],
                "result_count": search_record["result_count"],
            })

    def trace_summary(self) -> None:
        if not self.enabled or not self.tracer:
            return

        summary = self.summary()
        with self.tracer.span(
            "external_access_check",
            input={"question": self.question},
            metadata={
                "web_access_available": summary["web_search_used"],
                "web_search_used": summary["web_search_used"],
                "unexpected_external_api_used": summary[
                    "external_api_other_than_gemini_langfuse_used"
                ],
            },
        ) as span:
            span.end(output=summary)

    def summary(self) -> dict[str, Any]:
        return {
            "timestamp": utc_timestamp(),
            "enabled": self.enabled,
            "question": self.question,
            "web_search_used": bool(self.web_search_calls),
            "web_access_available": bool(self.web_search_calls),
            "web_search_calls": self.web_search_calls,
            "external_api_other_than_gemini_langfuse_used": bool(self.external_api_calls),
            "external_api_calls": self.external_api_calls,
            **self.gemini_grounding,
        }
