"""Optional external embedding providers."""

from __future__ import annotations

import ipaddress
import json
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from waji_rag.config import DEFAULT_EMBEDDING_NO_PROXY_HOSTS, EmbeddingConfig
from waji_rag.model_call_logging import append_model_api_request_log


@dataclass(slots=True)
class EmbeddingProviderError(RuntimeError):
    """Raised when an embedding provider cannot return usable vectors."""

    message: str

    def __str__(self) -> str:
        """Return the error message."""

        return self.message


class CommandEmbeddingProvider:
    """Call an external command that embeds text batches via JSON stdin/stdout.

    The command receives ``{"model": "...", "texts": [...]}`` on stdin and must
    return ``{"embeddings": [[...], ...]}`` on stdout. This keeps the core app
    model-agnostic while still enabling hybrid retrieval when a Windows machine
    has a local embedding model or service wrapper available.
    """

    def __init__(self, config: EmbeddingConfig, *, timeout_seconds: int = 120) -> None:
        """Store command embedding configuration."""

        if not config.is_available():
            raise ValueError("embedding config is not available")
        self.config = config
        self.timeout_seconds = timeout_seconds

    def embed_texts(self, texts: list[str], *, text_type: str = "document") -> list[list[float]]:
        """Embed a batch of texts and validate the returned vector payload."""

        if not texts:
            return []
        request = {"model": self.config.model, "texts": texts, "text_type": text_type}
        try:
            completed = subprocess.run(
                self.config.command,
                input=json.dumps(request, ensure_ascii=False),
                text=True,
                encoding="utf-8",
                capture_output=True,
                check=False,
                timeout=self.timeout_seconds,
            )
        except OSError as exc:
            raise EmbeddingProviderError(f"embedding command failed to start: {exc}") from exc
        except subprocess.TimeoutExpired as exc:
            raise EmbeddingProviderError(f"embedding command timed out after {self.timeout_seconds}s") from exc

        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            raise EmbeddingProviderError(f"embedding command exited {completed.returncode}: {stderr}") from None

        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise EmbeddingProviderError(f"embedding command returned invalid JSON: {exc}") from exc
        return validate_embedding_payload(payload, texts=texts)


class OpenAICompatibleEmbeddingProvider:
    """Call an OpenAI-compatible embeddings endpoint over HTTP."""

    def __init__(self, config: EmbeddingConfig) -> None:
        """Store HTTP embedding configuration."""

        if not config.is_available():
            raise ValueError("embedding config is not available")
        self.config = config

    def embed_texts(self, texts: list[str], *, text_type: str = "document") -> list[list[float]]:
        """Embed texts through an OpenAI-compatible embeddings endpoint."""

        if not texts:
            return []
        vectors: list[list[float]] = []
        batch_size = max(1, self.config.batch_size)
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vectors.extend(self._embed_batch(batch, text_type=text_type))
        return vectors

    def _embed_batch(self, texts: list[str], *, text_type: str) -> list[list[float]]:
        """Embed one batch and validate the response."""

        payload: dict[str, object] = {
            "input": texts,
        }
        provider = self.config.provider.lower().strip()
        if self.config.model:
            payload["model"] = self.config.model
        if provider != "vllm":
            payload["encoding_format"] = "float"
        if self.config.dimensions:
            payload["dimensions"] = self.config.dimensions
        if provider == "dashscope":
            payload["text_type"] = text_type
        response = post_json(
            f"{self.config.base_url.rstrip('/')}/embeddings",
            payload=payload,
            api_key=self.config.api_key,
            timeout_seconds=self.config.timeout_seconds,
            no_proxy_hosts=self.config.no_proxy_hosts,
            provider=self.config.provider,
            model=self.config.model,
            log_requests_enabled=self.config.log_requests_enabled,
            request_log_path=self.config.request_log_path,
        )
        return validate_embedding_payload(response, texts=texts)


def validate_embedding_payload(payload: object, *, texts: list[str]) -> list[list[float]]:
    """Validate known embedding response shapes and return vectors."""

    embeddings: object
    if isinstance(payload, dict) and isinstance(payload.get("embeddings"), list):
        embeddings = payload.get("embeddings")
    elif isinstance(payload, dict) and isinstance(payload.get("data"), list):
        rows = sorted(payload["data"], key=lambda item: int(item.get("index", 0)) if isinstance(item, dict) else 0)
        embeddings = [row.get("embedding") for row in rows if isinstance(row, dict)]
    else:
        embeddings = None

    if not isinstance(embeddings, list) or len(embeddings) != len(texts):
        raise EmbeddingProviderError("embedding provider must return one embedding per input text")

    vectors: list[list[float]] = []
    for index, embedding in enumerate(embeddings):
        if not isinstance(embedding, list) or not embedding:
            raise EmbeddingProviderError(f"embedding {index} is not a non-empty list")
        try:
            vectors.append([float(value) for value in embedding])
        except (TypeError, ValueError) as exc:
            raise EmbeddingProviderError(f"embedding {index} contains a non-numeric value") from exc
    return vectors


def post_json(
    url: str,
    *,
    payload: dict[str, object],
    api_key: str,
    timeout_seconds: float,
    no_proxy_hosts: list[str] | None = None,
    provider: str = "",
    model: str = "",
    log_requests_enabled: bool = False,
    request_log_path: str = "logs/model_api_requests.jsonl",
) -> dict[str, Any]:
    """POST JSON to an embedding endpoint and parse the JSON response."""

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    start = time.time()
    opener = (
        urllib.request.build_opener(urllib.request.ProxyHandler({}))
        if should_bypass_proxy(url, no_proxy_hosts or [])
        else urllib.request.build_opener()
    )
    status_code: int | None = None
    try:
        with opener.open(request, timeout=timeout_seconds) as response:
            status_code = int(response.status)
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        append_model_api_request_log(
            enabled=log_requests_enabled,
            log_path=request_log_path,
            service="embedding",
            provider=provider,
            model=model,
            method="POST",
            url=url,
            headers=headers,
            payload=payload,
            elapsed_ms=int((time.time() - start) * 1000),
            status="http_error",
            status_code=exc.code,
            response={"body_preview": body[:1000]},
            error=f"HTTP {exc.code}",
        )
        raise EmbeddingProviderError(f"embedding HTTP {exc.code}: {body[:500]}") from exc
    except urllib.error.URLError as exc:
        append_model_api_request_log(
            enabled=log_requests_enabled,
            log_path=request_log_path,
            service="embedding",
            provider=provider,
            model=model,
            method="POST",
            url=url,
            headers=headers,
            payload=payload,
            elapsed_ms=int((time.time() - start) * 1000),
            status="request_error",
            error=str(exc),
        )
        raise EmbeddingProviderError(f"embedding request failed after {time.time() - start:.2f}s: {exc}") from exc

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        append_model_api_request_log(
            enabled=log_requests_enabled,
            log_path=request_log_path,
            service="embedding",
            provider=provider,
            model=model,
            method="POST",
            url=url,
            headers=headers,
            payload=payload,
            elapsed_ms=int((time.time() - start) * 1000),
            status="invalid_json",
            status_code=status_code,
            response={"body_preview": body[:1000]},
            error=str(exc),
        )
        raise EmbeddingProviderError(f"embedding endpoint returned invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        append_model_api_request_log(
            enabled=log_requests_enabled,
            log_path=request_log_path,
            service="embedding",
            provider=provider,
            model=model,
            method="POST",
            url=url,
            headers=headers,
            payload=payload,
            elapsed_ms=int((time.time() - start) * 1000),
            status="invalid_payload",
            status_code=status_code,
            response=parsed,
            error="non-object JSON payload",
        )
        raise EmbeddingProviderError("embedding endpoint returned a non-object JSON payload")
    append_model_api_request_log(
        enabled=log_requests_enabled,
        log_path=request_log_path,
        service="embedding",
        provider=provider,
        model=model,
        method="POST",
        url=url,
        headers=headers,
        payload=payload,
        elapsed_ms=int((time.time() - start) * 1000),
        status="ok",
        status_code=status_code,
        response=parsed,
    )
    return parsed


def is_local_url(url: str) -> bool:
    """Return whether a URL targets the built-in local no-proxy hosts."""

    return should_bypass_proxy(url, list(DEFAULT_EMBEDDING_NO_PROXY_HOSTS))


def should_bypass_proxy(url: str, no_proxy_hosts: list[str]) -> bool:
    """Return whether a URL host matches configured no-proxy host patterns."""

    host = (urlparse(url).hostname or "").strip().lower()
    if not host:
        return False
    for raw_pattern in no_proxy_hosts:
        pattern = str(raw_pattern).strip().lower()
        if not pattern:
            continue
        if matches_no_proxy_pattern(host, pattern):
            return True
    return False


def matches_no_proxy_pattern(host: str, pattern: str) -> bool:
    """Match one host against exact, wildcard, suffix, or CIDR no-proxy patterns."""

    if pattern == "*":
        return True
    if pattern.startswith("*."):
        suffix = pattern[1:]
        return host.endswith(suffix)
    if pattern.startswith("."):
        return host == pattern[1:] or host.endswith(pattern)
    if "/" in pattern:
        try:
            return ipaddress.ip_address(host) in ipaddress.ip_network(pattern, strict=False)
        except ValueError:
            return False
    return host == pattern
