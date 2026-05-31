"""Runtime configuration for retrieval, rerank, answer generation, and optional embeddings."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_DASHSCOPE_RERANK_BASE_URL = "https://dashscope.aliyuncs.com/compatible-api/v1"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_DOCARBOR_ENV_PATH = "/Users/weichong/Documents/new_working_area/file_tree/DocArbor/.env"
DEFAULT_EMBEDDING_NO_PROXY_HOSTS = ("localhost", "127.0.0.1", "127.0.0.0/8", "::1")
DEFAULT_MODEL_API_REQUEST_LOG_PATH = "logs/model_api_requests.jsonl"
SECRET_KEY_PATTERN = re.compile(
    r"^(api[_-]?key|secret|password|bearer[_-]?token|access[_-]?token|refresh[_-]?token)$",
    re.IGNORECASE,
)


@dataclass(slots=True)
class EmbeddingConfig:
    """Configuration for optional document and query embeddings."""

    enabled: bool = False
    provider: str = "dashscope"
    model: str = "text-embedding-v4"
    base_url: str = DEFAULT_DASHSCOPE_BASE_URL
    api_key: str = ""
    api_key_env: str = "DOCARBOR_EMBEDDING_API_KEY"
    command: list[str] = field(default_factory=list)
    dimensions: int | None = None
    batch_size: int = 10
    timeout_seconds: float = 180.0
    no_proxy_hosts: list[str] = field(default_factory=lambda: list(DEFAULT_EMBEDDING_NO_PROXY_HOSTS))
    log_requests_enabled: bool = True
    request_log_path: str = DEFAULT_MODEL_API_REQUEST_LOG_PATH

    def is_available(self) -> bool:
        """Return whether this config can create embeddings."""

        if not self.enabled:
            return False
        provider = self.provider.lower().strip()
        if provider == "command":
            return bool(self.command)
        if provider == "vllm":
            return bool(self.base_url)
        if provider == "openai":
            return bool(self.model and self.base_url)
        return bool(self.api_key and self.model and self.base_url)


@dataclass(slots=True)
class RetrievalConfig:
    """Configuration for BM25 and optional hybrid retrieval."""

    mode: str = "auto"
    bm25_top_k: int = 20
    vector_top_k: int = 20
    hybrid_alpha: float = 0.75
    work_order_candidate_top_k: int = 50
    work_order_min_relative_score: float = 0.45
    work_order_max_hits: int = 10


@dataclass(slots=True)
class RerankConfig:
    """Configuration for optional cross-encoder style reranking."""

    enabled: bool = False
    provider: str = "dashscope"
    model: str = "qwen3-rerank"
    base_url: str = DEFAULT_DASHSCOPE_RERANK_BASE_URL
    api_key: str = ""
    api_key_env: str = "DOCARBOR_RERANK_API_KEY"
    top_n: int = 8
    doc_char_limit: int = 1200
    timeout_seconds: float = 180.0
    no_proxy_hosts: list[str] = field(default_factory=lambda: list(DEFAULT_EMBEDDING_NO_PROXY_HOSTS))

    def is_available(self) -> bool:
        """Return whether this config can rerank retrieved evidence."""

        return self.enabled and bool(self.api_key and self.model and self.base_url)


@dataclass(slots=True)
class LLMConfig:
    """Configuration for optional final answer generation."""

    enabled: bool = False
    provider: str = "dashscope"
    model: str = "qwen3.5-plus"
    base_url: str = DEFAULT_DASHSCOPE_BASE_URL
    api_key: str = ""
    api_key_env: str = "DOCARBOR_LLM_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 1400
    timeout_seconds: float = 180.0
    disable_thinking: bool = True
    no_proxy_hosts: list[str] = field(default_factory=lambda: list(DEFAULT_EMBEDDING_NO_PROXY_HOSTS))
    log_requests_enabled: bool = True
    request_log_path: str = DEFAULT_MODEL_API_REQUEST_LOG_PATH

    def is_available(self) -> bool:
        """Return whether this config can generate final answers."""

        if self.provider.lower().strip() == "vllm":
            return self.enabled and bool(self.model and self.base_url)
        return self.enabled and bool(self.api_key and self.model and self.base_url)


@dataclass(slots=True)
class AnswerConfig:
    """Configuration for evidence packaging before final answer generation."""

    enabled: bool = True
    evidence_top_k: int = 8
    include_debug: bool = True


@dataclass(slots=True)
class AppConfig:
    """Top-level application configuration."""

    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    answer: AnswerConfig = field(default_factory=AnswerConfig)

    def to_dict(self) -> dict[str, object]:
        """Return this config as a redacted JSON-serializable dictionary."""

        return redact_secrets(asdict(self))

    def retrieval_mode(self) -> str:
        """Resolve the effective retrieval mode."""

        requested = self.retrieval.mode.lower().strip()
        if requested == "bm25":
            return "bm25"
        if requested == "hybrid":
            return "hybrid" if self.embedding.is_available() else "bm25"
        return "hybrid" if self.embedding.is_available() else "bm25"


def load_config(
    path: Path | None,
    *,
    overrides: dict[str, Any] | None = None,
    env_path: Path | None = None,
) -> AppConfig:
    """Load app config from JSON, env, and explicit overrides."""

    env_values = load_env_file(env_path) if env_path else {}
    payload = build_env_config_payload(env_values)
    if path is not None:
        file_payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(file_payload, dict):
            raise ValueError("config file must contain a JSON object")
        payload = deep_merge(payload, file_payload)
    if overrides:
        payload = deep_merge(payload, overrides)
    return config_from_payload(payload, env_values=env_values)


def config_from_payload(payload: dict[str, Any], *, env_values: dict[str, str] | None = None) -> AppConfig:
    """Build an application config from a plain dictionary."""

    if not isinstance(payload, dict):
        raise ValueError("config payload must be an object")
    env_values = env_values or {}
    retrieval_payload = object_payload(payload.get("retrieval"))
    embedding_payload = object_payload(payload.get("embedding"))
    rerank_payload = object_payload(payload.get("rerank"))
    llm_payload = object_payload(payload.get("llm"))
    answer_payload = object_payload(payload.get("answer"))
    default_log_enabled = as_bool(payload.get("log_requests_enabled", True))
    default_log_path = str(payload.get("request_log_path") or DEFAULT_MODEL_API_REQUEST_LOG_PATH)

    embedding_api_key_env = str(embedding_payload.get("api_key_env") or "DOCARBOR_EMBEDDING_API_KEY")
    llm_api_key_env = str(llm_payload.get("api_key_env") or "DOCARBOR_LLM_API_KEY")
    rerank_api_key_env = str(rerank_payload.get("api_key_env") or "DOCARBOR_RERANK_API_KEY")
    rerank_key = first_non_empty(
        str(rerank_payload.get("api_key") or ""),
        env_values.get(rerank_api_key_env),
        env_values.get("DASHSCOPE_API_KEY"),
        env_values.get("DOCARBOR_LLM_API_KEY"),
        os.getenv(rerank_api_key_env),
        os.getenv("DASHSCOPE_API_KEY"),
        os.getenv("DOCARBOR_LLM_API_KEY"),
    )

    return AppConfig(
        retrieval=RetrievalConfig(
            mode=str(retrieval_payload.get("mode", "auto")),
            bm25_top_k=int(retrieval_payload.get("bm25_top_k", 20)),
            vector_top_k=int(retrieval_payload.get("vector_top_k", 20)),
            hybrid_alpha=float(retrieval_payload.get("hybrid_alpha", 0.75)),
            work_order_candidate_top_k=max(1, int(retrieval_payload.get("work_order_candidate_top_k", 50))),
            work_order_min_relative_score=clamp_float(
                retrieval_payload.get("work_order_min_relative_score", 0.45),
                minimum=0.0,
                maximum=1.0,
            ),
            work_order_max_hits=max(0, int(retrieval_payload.get("work_order_max_hits", 10))),
        ),
        embedding=EmbeddingConfig(
            enabled=as_bool(embedding_payload.get("enabled", False)),
            provider=str(embedding_payload.get("provider", "dashscope")),
            model=normalize_model_name(str(embedding_payload.get("model", "text-embedding-v4"))),
            base_url=str(embedding_payload.get("base_url") or DEFAULT_DASHSCOPE_BASE_URL),
            api_key=first_non_empty(
                str(embedding_payload.get("api_key") or ""),
                env_values.get(embedding_api_key_env),
                env_values.get("DASHSCOPE_API_KEY"),
                os.getenv(embedding_api_key_env),
                os.getenv("DASHSCOPE_API_KEY"),
            ),
            api_key_env=embedding_api_key_env,
            command=[str(item) for item in embedding_payload.get("command", [])],
            dimensions=optional_int(embedding_payload.get("dimensions"), default=None),
            batch_size=max(1, int(embedding_payload.get("batch_size", 10))),
            timeout_seconds=float(embedding_payload.get("timeout_seconds", 180.0)),
            no_proxy_hosts=parse_string_list(
                embedding_payload.get("no_proxy_hosts", embedding_payload.get("no_proxy")),
                default=list(DEFAULT_EMBEDDING_NO_PROXY_HOSTS),
            ),
            log_requests_enabled=as_bool(embedding_payload.get("log_requests_enabled", default_log_enabled)),
            request_log_path=str(embedding_payload.get("request_log_path") or default_log_path),
        ),
        rerank=RerankConfig(
            enabled=as_bool(rerank_payload.get("enabled", False)),
            provider=str(rerank_payload.get("provider", "dashscope")),
            model=normalize_model_name(str(rerank_payload.get("model", "qwen3-rerank"))),
            base_url=str(rerank_payload.get("base_url") or DEFAULT_DASHSCOPE_RERANK_BASE_URL),
            api_key=rerank_key,
            api_key_env=rerank_api_key_env,
            top_n=max(1, int(rerank_payload.get("top_n", 8))),
            doc_char_limit=max(100, int(rerank_payload.get("doc_char_limit", 1200))),
            timeout_seconds=float(rerank_payload.get("timeout_seconds", 180.0)),
            no_proxy_hosts=parse_string_list(
                rerank_payload.get("no_proxy_hosts", rerank_payload.get("no_proxy")),
                default=list(DEFAULT_EMBEDDING_NO_PROXY_HOSTS),
            ),
        ),
        llm=LLMConfig(
            enabled=as_bool(llm_payload.get("enabled", False)),
            provider=str(llm_payload.get("provider", "dashscope")),
            model=normalize_model_name(str(llm_payload.get("model", "qwen3.5-plus"))),
            base_url=str(llm_payload.get("base_url") or DEFAULT_DASHSCOPE_BASE_URL),
            api_key=first_non_empty(
                str(llm_payload.get("api_key") or ""),
                env_values.get(llm_api_key_env),
                env_values.get("DASHSCOPE_API_KEY"),
                os.getenv(llm_api_key_env),
                os.getenv("DASHSCOPE_API_KEY"),
            ),
            api_key_env=llm_api_key_env,
            temperature=float(llm_payload.get("temperature", 0.0)),
            max_tokens=max(1, int(llm_payload.get("max_tokens", 1400))),
            timeout_seconds=float(llm_payload.get("timeout_seconds", 180.0)),
            disable_thinking=as_bool(llm_payload.get("disable_thinking", True)),
            no_proxy_hosts=parse_string_list(
                llm_payload.get("no_proxy_hosts", llm_payload.get("no_proxy")),
                default=list(DEFAULT_EMBEDDING_NO_PROXY_HOSTS),
            ),
            log_requests_enabled=as_bool(llm_payload.get("log_requests_enabled", default_log_enabled)),
            request_log_path=str(llm_payload.get("request_log_path") or default_log_path),
        ),
        answer=AnswerConfig(
            enabled=as_bool(answer_payload.get("enabled", True)),
            evidence_top_k=max(1, int(answer_payload.get("evidence_top_k", 8))),
            include_debug=as_bool(answer_payload.get("include_debug", True)),
        ),
    )


def build_env_config_payload(env_values: dict[str, str]) -> dict[str, Any]:
    """Build a config payload using DocArbor-compatible env names."""

    return {
        "embedding": {
            "provider": env_values.get("DOCARBOR_EMBEDDING_API_PROVIDER", "dashscope"),
            "model": env_values.get("DOCARBOR_RETRIEVAL_EMBEDDING_MODEL", "text-embedding-v4"),
            "base_url": env_values.get("DOCARBOR_EMBEDDING_BASE_URL", DEFAULT_DASHSCOPE_BASE_URL),
            "api_key_env": "DOCARBOR_EMBEDDING_API_KEY",
            "dimensions": optional_int(env_values.get("DOCARBOR_EMBEDDING_DIMENSIONS"), default=None),
            "no_proxy_hosts": parse_string_list(
                env_values.get("DOCARBOR_EMBEDDING_NO_PROXY_HOSTS"),
                default=list(DEFAULT_EMBEDDING_NO_PROXY_HOSTS),
            ),
            "log_requests_enabled": env_values.get("DOCARBOR_MODEL_REQUEST_LOG_ENABLED", "true"),
            "request_log_path": env_values.get("DOCARBOR_MODEL_REQUEST_LOG_PATH", DEFAULT_MODEL_API_REQUEST_LOG_PATH),
        },
        "rerank": {
            "provider": "dashscope",
            "model": env_values.get("DOCARBOR_STAGE3_RERANK_MODEL", "qwen3-rerank"),
            "base_url": env_values.get("DOCARBOR_RERANK_BASE_URL", DEFAULT_DASHSCOPE_RERANK_BASE_URL),
            "api_key_env": "DOCARBOR_RERANK_API_KEY",
            "no_proxy_hosts": parse_string_list(
                env_values.get("DOCARBOR_RERANK_NO_PROXY_HOSTS", env_values.get("DOCARBOR_MODEL_NO_PROXY_HOSTS")),
                default=list(DEFAULT_EMBEDDING_NO_PROXY_HOSTS),
            ),
        },
        "llm": {
            "provider": env_values.get("DOCARBOR_LLM_API_PROVIDER", "dashscope"),
            "model": env_values.get(
                "DOCARBOR_RETRIEVAL_ANSWER_MODEL",
                env_values.get("DOCARBOR_BUILD_NATIVE_MODEL", "dashscope/qwen3.5-plus"),
            ),
            "base_url": env_values.get("DOCARBOR_LLM_BASE_URL", DEFAULT_DASHSCOPE_BASE_URL),
            "api_key_env": "DOCARBOR_LLM_API_KEY",
            "no_proxy_hosts": parse_string_list(
                env_values.get("DOCARBOR_LLM_NO_PROXY_HOSTS", env_values.get("DOCARBOR_MODEL_NO_PROXY_HOSTS")),
                default=list(DEFAULT_EMBEDDING_NO_PROXY_HOSTS),
            ),
            "log_requests_enabled": env_values.get("DOCARBOR_MODEL_REQUEST_LOG_ENABLED", "true"),
            "request_log_path": env_values.get("DOCARBOR_MODEL_REQUEST_LOG_PATH", DEFAULT_MODEL_API_REQUEST_LOG_PATH),
        },
    }


def load_env_file(path: Path | None) -> dict[str, str]:
    """Load key-value pairs from a dotenv-style file without mutating os.environ."""

    if path is None:
        return {}
    resolved = path.expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"env file does not exist: {resolved}")
    values: dict[str, str] = {}
    for raw_line in resolved.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        values[key] = value
    return values


def write_default_config(path: Path) -> None:
    """Write a default retrieval config JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(AppConfig().to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def object_payload(value: object) -> dict[str, Any]:
    """Return an object payload or an empty mapping."""

    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("config section must be an object")
    return dict(value)


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Merge two nested dictionaries without mutating either input."""

    merged = dict(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def redact_secrets(value: Any) -> Any:
    """Return a copy of a JSON-like value with secret-looking fields redacted."""

    if isinstance(value, dict):
        return {
            key: ("<redacted>" if SECRET_KEY_PATTERN.search(str(key)) and item else redact_secrets(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    return value


def normalize_model_name(model_name: str | None) -> str:
    """Strip optional provider prefixes such as ``dashscope/`` from model names."""

    if not model_name:
        return ""
    if "/" in model_name:
        return model_name.split("/", 1)[-1]
    return model_name


def optional_int(value: object, *, default: int | None = None) -> int | None:
    """Parse an optional integer."""

    if value in (None, ""):
        return default
    return int(value)


def clamp_float(value: object, *, minimum: float, maximum: float) -> float:
    """Parse and clamp a float to an inclusive range."""

    return min(max(float(value), minimum), maximum)


def parse_string_list(value: object, *, default: list[str] | None = None) -> list[str]:
    """Parse a comma-separated string or JSON-like list into strings."""

    if value in (None, ""):
        return list(default or [])
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def as_bool(value: object) -> bool:
    """Parse common boolean representations."""

    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def first_non_empty(*values: object) -> str:
    """Return the first non-empty string-like value."""

    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""
