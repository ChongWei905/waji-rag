from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from waji_rag.config import load_config  # noqa: E402
from waji_rag.llm import OpenAICompatibleChatClient  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description="Verify the configured LLM chat-completions endpoint.")
    parser.add_argument("--env-file", help="Dotenv file containing LLM API key and model defaults.")
    parser.add_argument("--config", help="Optional JSON config file.")
    parser.add_argument("--provider", default="dashscope", help="LLM provider name. Defaults to dashscope.")
    parser.add_argument("--model", default="qwen3.5-plus", help="LLM model name.")
    parser.add_argument("--base-url", help="LLM base URL.")
    parser.add_argument("--api-key", help="LLM API key. When set, this takes precedence over env/config values.")
    parser.add_argument("--api-key-env", default="DOCARBOR_LLM_API_KEY", help="API key env var name.")
    parser.add_argument("--timeout-seconds", type=float, default=180.0, help="HTTP timeout in seconds.")
    parser.add_argument("--retries", type=int, default=2, help="Retry count for transient network failures. Defaults to 2.")
    parser.add_argument("--max-tokens", type=int, default=120, help="Max output tokens. Defaults to 120.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature. Defaults to 0.")
    parser.add_argument(
        "--prompt",
        default="请只用一句中文回复：LLM 接口连通，挖掘机诊断问答可以继续。",
        help="Prompt used for verification.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run LLM endpoint verification."""

    args = build_parser().parse_args(argv)
    started_at = time.time()
    try:
        config = load_llm_config(args)
        client = OpenAICompatibleChatClient(config.llm)
        result = call_with_retries(
            lambda: client.complete(
                [
                    {"role": "system", "content": "你是接口连通性测试助手。回答要简短。"},
                    {"role": "user", "content": args.prompt},
                ]
            ),
            retries=args.retries,
        )
        if not result.text:
            raise RuntimeError("LLM returned an empty response")
        payload = {
            "status": "ok",
            "provider": config.llm.provider,
            "model": config.llm.model,
            "base_url": config.llm.base_url,
            "api_key_source": "cli" if args.api_key else config.llm.api_key_env,
            "api_key_env": config.llm.api_key_env,
            "text": result.text,
            "debug": result.debug,
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


def load_llm_config(args: argparse.Namespace) -> Any:
    """Load app config with LLM explicitly enabled."""

    overrides: dict[str, object] = {
        "llm": {
            "enabled": True,
            "provider": args.provider,
            "model": args.model,
            "api_key_env": args.api_key_env,
            "timeout_seconds": args.timeout_seconds,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        }
    }
    if args.api_key:
        overrides["llm"]["api_key"] = args.api_key  # type: ignore[index]
    if args.base_url:
        overrides["llm"]["base_url"] = args.base_url  # type: ignore[index]
    config = load_config(
        Path(args.config) if args.config else None,
        overrides=overrides,
        env_path=Path(args.env_file) if args.env_file else None,
    )
    if not config.llm.is_available():
        raise RuntimeError("LLM config is not available. Check env file, API key env var, provider, model, and base URL.")
    return config


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
