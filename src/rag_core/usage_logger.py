"""
JSONL usage logging for the browser demo.

The logger records request/response metadata only. It never reads or writes API
keys, and callers should avoid passing secrets in metadata.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
LOG_PATH = ROOT_DIR / "logs" / "usage.log"


def _json_default(value: Any) -> str:
    return str(value)


def log_request(
    *,
    question: str,
    answer: str,
    sources: list,
    metadata: dict,
    runtime_seconds: float,
) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "answer": answer,
        "sources": sources,
        "metadata": metadata,
        "runtime_seconds": runtime_seconds,
    }

    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
