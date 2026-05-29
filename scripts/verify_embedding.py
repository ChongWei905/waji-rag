from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from waji_rag.config import load_config, load_env_file  # noqa: E402
from waji_rag.embedding import CommandEmbeddingProvider, OpenAICompatibleEmbeddingProvider  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description="Verify the configured embedding endpoint.")
    parser.add_argument("--env-file", help="Dotenv file containing embedding API key and model defaults.")
    parser.add_argument("--config", help="Optional JSON config file.")
    parser.add_argument("--provider", default="dashscope", help="Embedding provider name. Defaults to dashscope.")
    parser.add_argument("--model", default="text-embedding-v4", help="Embedding model name.")
    parser.add_argument("--base-url", help="Embedding base URL.")
    parser.add_argument("--api-key", help="Embedding API key. When set, this takes precedence over env/config values.")
    parser.add_argument("--api-key-env", default="DOCARBOR_EMBEDDING_API_KEY", help="API key env var name.")
    parser.add_argument(
        "--fallback-api-key-env",
        default="DOCARBOR_LLM_API_KEY",
        help="Fallback API key env var name when --api-key-env is absent. Defaults to DOCARBOR_LLM_API_KEY.",
    )
    parser.add_argument("--dimensions", type=int, default=1024, help="Expected embedding dimensions. Defaults to 1024.")
    parser.add_argument("--batch-size", type=int, default=2, help="Embedding batch size. Defaults to 2.")
    parser.add_argument("--timeout-seconds", type=float, default=180.0, help="HTTP timeout in seconds.")
    parser.add_argument("--retries", type=int, default=2, help="Retry count for transient network failures. Defaults to 2.")
    parser.add_argument(
        "--command",
        nargs="+",
        help="Command provider executable and args. Example: --provider command --command python embedder.py",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run embedding endpoint verification."""

    args = build_parser().parse_args(argv)
    started_at = time.time()
    try:
        config = load_embedding_config(args)
        provider = build_provider(config.embedding)
        document_vectors = call_with_retries(
            lambda: provider.embed_texts(
                ["风扇皮带异响，发动机前端有尖叫声。", "行走单边慢，右侧行走无力。"],
                text_type="document",
            ),
            retries=args.retries,
        )
        query_vectors = call_with_retries(
            lambda: provider.embed_texts(["用户报修风扇皮带异响，需要诊断原因和备件。"], text_type="query"),
            retries=args.retries,
        )
        document_dim = len(document_vectors[0])
        query_dim = len(query_vectors[0])
        expected_dim = config.embedding.dimensions
        if expected_dim and (document_dim != expected_dim or query_dim != expected_dim):
            raise RuntimeError(
                f"embedding dimension mismatch: expected={expected_dim}, document={document_dim}, query={query_dim}"
            )
        payload = {
            "status": "ok",
            "provider": config.embedding.provider,
            "model": config.embedding.model,
            "base_url": config.embedding.base_url,
            "api_key_source": api_key_source(args),
            "api_key_env": config.embedding.api_key_env,
            "document_vectors": len(document_vectors),
            "query_vectors": len(query_vectors),
            "document_dim": document_dim,
            "query_dim": query_dim,
            "document_vector_head": rounded_head(document_vectors[0]),
            "query_vector_head": rounded_head(query_vectors[0]),
            "elapsed_ms": int((time.time() - started_at) * 1000),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001 - verification script should report all failures.
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "elapsed_ms": int((time.time() - started_at) * 1000),
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1


def load_embedding_config(args: argparse.Namespace) -> Any:
    """Load app config with embedding explicitly enabled."""

    embedding_overrides: dict[str, object] = {
        "enabled": True,
        "provider": args.provider,
        "model": args.model,
        "api_key_env": args.api_key_env,
        "dimensions": args.dimensions,
        "batch_size": args.batch_size,
        "timeout_seconds": args.timeout_seconds,
    }
    if args.api_key:
        embedding_overrides["api_key"] = args.api_key
    fallback_key = "" if args.api_key else load_fallback_api_key(args)
    if fallback_key:
        embedding_overrides["api_key"] = fallback_key
        embedding_overrides["api_key_env"] = args.fallback_api_key_env

    overrides: dict[str, object] = {
        "embedding": {
            **embedding_overrides,
        }
    }
    if args.base_url:
        overrides["embedding"]["base_url"] = args.base_url  # type: ignore[index]
    if args.command:
        overrides["embedding"]["command"] = args.command  # type: ignore[index]
    config = load_config(
        Path(args.config) if args.config else None,
        overrides=overrides,
        env_path=Path(args.env_file) if args.env_file else None,
    )
    if not config.embedding.is_available():
        raise RuntimeError(
            "embedding config is not available. Check env file, API key env var, provider, model, and base URL."
        )
    return config


def api_key_source(args: argparse.Namespace) -> str:
    """Return a non-secret description of where the API key came from."""

    if args.api_key:
        return "cli"
    if args.env_file:
        values = load_env_file(Path(args.env_file))
        if values.get(args.api_key_env):
            return args.api_key_env
        if args.fallback_api_key_env and values.get(args.fallback_api_key_env):
            return args.fallback_api_key_env
    if os.getenv(args.api_key_env):
        return args.api_key_env
    if args.fallback_api_key_env and os.getenv(args.fallback_api_key_env):
        return args.fallback_api_key_env
    return "config"


def load_fallback_api_key(args: argparse.Namespace) -> str:
    """Return a fallback API key when the primary embedding key is absent."""

    if not args.fallback_api_key_env or args.fallback_api_key_env == args.api_key_env:
        return ""
    values: dict[str, str] = {}
    if args.env_file:
        values = load_env_file(Path(args.env_file))
    primary = values.get(args.api_key_env) or os.getenv(args.api_key_env, "")
    fallback = values.get(args.fallback_api_key_env) or os.getenv(args.fallback_api_key_env, "")
    if primary or not fallback:
        return ""
    return fallback


def build_provider(config: Any) -> CommandEmbeddingProvider | OpenAICompatibleEmbeddingProvider:
    """Build the selected embedding provider."""

    if config.provider.lower().strip() == "command":
        return CommandEmbeddingProvider(config, timeout_seconds=int(config.timeout_seconds))
    return OpenAICompatibleEmbeddingProvider(config)


def rounded_head(vector: list[float], *, limit: int = 5) -> list[float]:
    """Return a short rounded vector prefix for diagnostics."""

    return [round(float(value), 6) for value in vector[:limit]]


def call_with_retries(operation: Any, *, retries: int) -> Any:
    """Run an operation with simple retry handling for transient API failures."""

    last_error: Exception | None = None
    for attempt in range(max(0, retries) + 1):
        try:
            return operation()
        except Exception as exc:  # noqa: BLE001 - verification script retries broad transient failures.
            last_error = exc
            if attempt >= max(0, retries):
                break
            time.sleep(1 + attempt)
    if last_error is None:
        raise RuntimeError("operation failed without an exception")
    raise last_error


if __name__ == "__main__":
    raise SystemExit(main())
