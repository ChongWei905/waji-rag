"""JSONL logging helpers for model API requests."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


SENSITIVE_HEADER_NAMES = {"authorization", "api-key", "x-api-key", "cookie", "set-cookie"}
SENSITIVE_QUERY_NAMES = {"api_key", "apikey", "key", "token", "access_token"}


def append_model_api_request_log(
    *,
    enabled: bool,
    log_path: str,
    service: str,
    provider: str,
    model: str,
    method: str,
    url: str,
    headers: dict[str, str],
    payload: dict[str, object],
    elapsed_ms: int,
    status: str,
    status_code: int | None = None,
    response: object | None = None,
    error: str | None = None,
) -> None:
    """Append one model API request record to a JSONL file without breaking the caller."""

    if not enabled:
        return
    try:
        path = Path(log_path or "logs/model_api_requests.jsonl").expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "service": service,
            "provider": provider,
            "model": model,
            "request": {
                "method": method,
                "url": redact_url(url),
                "headers": redact_headers(headers),
                "payload": payload,
            },
            "result": {
                "status": status,
                "status_code": status_code,
                "elapsed_ms": elapsed_ms,
                "response_summary": summarize_response(response),
                "error": error,
            },
        }
        with path.open("a", encoding="utf-8") as file_obj:
            file_obj.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError:
        return


def redact_headers(headers: dict[str, str]) -> dict[str, str]:
    """Return headers with sensitive values redacted."""

    redacted: dict[str, str] = {}
    for key, value in headers.items():
        redacted[key] = "<redacted>" if key.lower() in SENSITIVE_HEADER_NAMES and value else value
    return redacted


def redact_url(url: str) -> str:
    """Return a URL with secret-looking query parameters redacted."""

    split_url = urlsplit(url)
    if not split_url.query:
        return url
    query_items = [
        (key, "<redacted>" if key.lower() in SENSITIVE_QUERY_NAMES and value else value)
        for key, value in parse_qsl(split_url.query, keep_blank_values=True)
    ]
    return urlunsplit((split_url.scheme, split_url.netloc, split_url.path, urlencode(query_items), split_url.fragment))


def summarize_response(response: object | None) -> dict[str, object] | None:
    """Return a compact non-vector response summary for model-call debugging."""

    if response is None:
        return None
    if not isinstance(response, dict):
        return {"type": type(response).__name__}
    summary: dict[str, object] = {"keys": sorted(str(key) for key in response.keys())}
    usage = response.get("usage")
    if isinstance(usage, dict):
        summary["usage"] = usage
    data = response.get("data")
    if isinstance(data, list):
        summary["data_count"] = len(data)
        if data and isinstance(data[0], dict) and isinstance(data[0].get("embedding"), list):
            summary["first_embedding_dimensions"] = len(data[0]["embedding"])
    embeddings = response.get("embeddings")
    if isinstance(embeddings, list):
        summary["embedding_count"] = len(embeddings)
        if embeddings and isinstance(embeddings[0], list):
            summary["first_embedding_dimensions"] = len(embeddings[0])
    choices = response.get("choices")
    if isinstance(choices, list):
        summary["choice_count"] = len(choices)
        if choices and isinstance(choices[0], dict):
            summary["finish_reason"] = choices[0].get("finish_reason")
    return summary
