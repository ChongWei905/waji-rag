"""OpenAI-compatible LLM, rerank, and answer generation helpers."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from typing import Any

from waji_rag.config import LLMConfig, RerankConfig
from waji_rag.embedding import should_bypass_proxy
from waji_rag.model_call_logging import append_model_api_request_log


@dataclass(slots=True)
class ModelCallResult:
    """One model call result with lightweight diagnostics."""

    text: str
    debug: dict[str, object]


@dataclass(slots=True)
class QueryParseResult:
    """Structured query parse payload returned by the LLM."""

    payload: dict[str, object]
    debug: dict[str, object]


@dataclass(slots=True)
class RerankItem:
    """One reranked document result."""

    index: int
    score: float
    document: str | None = None


class OpenAICompatibleChatClient:
    """Minimal OpenAI-compatible chat-completions HTTP client."""

    def __init__(self, config: LLMConfig) -> None:
        """Store chat model configuration."""

        if not config.is_available():
            raise ValueError("LLM config is not available")
        self.config = config

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        service: str = "llm",
        error_prefix: str = "chat",
    ) -> ModelCallResult:
        """Run one chat completion and return text plus diagnostics."""

        payload: dict[str, object] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.provider.lower().strip() == "dashscope" and self.config.disable_thinking:
            payload["enable_thinking"] = False
        started_at = time.time()
        response = post_json(
            f"{self.config.base_url.rstrip('/')}/chat/completions",
            payload=payload,
            api_key=self.config.api_key,
            timeout_seconds=self.config.timeout_seconds,
            error_prefix=error_prefix,
            no_proxy_hosts=self.config.no_proxy_hosts,
            service=service,
            provider=self.config.provider,
            model=self.config.model,
            log_requests_enabled=self.config.log_requests_enabled,
            request_log_path=self.config.request_log_path,
        )
        elapsed_ms = int((time.time() - started_at) * 1000)
        choices = response.get("choices") if isinstance(response, dict) else None
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("chat endpoint returned no choices")
        first_choice = choices[0]
        message = first_choice.get("message", {}) if isinstance(first_choice, dict) else {}
        text = str(message.get("content") or "") if isinstance(message, dict) else ""
        return ModelCallResult(
            text=text.strip(),
            debug={
                "provider": self.config.provider,
                "model": self.config.model,
                "elapsed_ms": elapsed_ms,
                "usage": response.get("usage") if isinstance(response.get("usage"), dict) else None,
                "finish_reason": first_choice.get("finish_reason") if isinstance(first_choice, dict) else None,
            },
        )


class DashScopeRerankClient:
    """Minimal DashScope-compatible rerank HTTP client."""

    def __init__(self, config: RerankConfig) -> None:
        """Store rerank configuration."""

        if not config.is_available():
            raise ValueError("rerank config is not available")
        self.config = config

    def rerank(self, *, query: str, documents: list[str], top_n: int | None = None) -> tuple[list[RerankItem], dict[str, object]]:
        """Rerank candidate documents for one query."""

        if not documents:
            return [], {"status": "skipped", "reason": "empty_documents"}
        request_body: dict[str, object] = {
            "model": self.config.model,
            "query": query,
            "documents": documents,
            "return_documents": False,
            "top_n": int(top_n or self.config.top_n),
        }
        started_at = time.time()
        response = post_json(
            f"{self.config.base_url.rstrip('/')}/reranks",
            payload=request_body,
            api_key=self.config.api_key,
            timeout_seconds=self.config.timeout_seconds,
            error_prefix="rerank",
            no_proxy_hosts=self.config.no_proxy_hosts,
        )
        elapsed_ms = int((time.time() - started_at) * 1000)
        results = extract_rerank_results(response)
        debug = {
            "status": "ok",
            "provider": self.config.provider,
            "model": self.config.model,
            "elapsed_ms": elapsed_ms,
            "input_count": len(documents),
            "returned_count": len(results),
            "usage": response.get("usage") if isinstance(response.get("usage"), dict) else None,
        }
        return results, debug


def generate_diagnostic_answer(
    *,
    query: str,
    evidence_items: list[dict[str, object]],
    part_candidates: list[dict[str, object]],
    config: LLMConfig,
) -> ModelCallResult:
    """Generate the final diagnostic answer from retrieved evidence."""

    client = OpenAICompatibleChatClient(config)
    evidence_text = json.dumps(
        {"evidence": evidence_items, "part_candidates": part_candidates},
        ensure_ascii=False,
        indent=2,
    )
    messages = [
        {
            "role": "system",
            "content": (
                "你是挖掘机售后维修诊断助手。只能基于给定证据回答，不要编造备件编码。"
                "输出中文，先列可能故障原因，再列处理建议，再列备件候选表。"
                "如果备件信息来自历史工单，要说明未经过当前设备适配校验。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"用户问题：{query}\n\n"
                "检索证据如下。请严格引用其中的工单、手册或备件候选，不要输出不存在的编码。\n"
                f"{evidence_text}"
            ),
        },
    ]
    result = client.complete(messages, service="llm", error_prefix="llm")
    if not result.text:
        result.text = build_fallback_answer(query=query, evidence_items=evidence_items, part_candidates=part_candidates)
    return result


def parse_diagnostic_query_constraints(*, query: str, config: LLMConfig) -> QueryParseResult:
    """Parse a diagnostic query into fault phrase, component anchors, and symptom terms."""

    parser_config = replace(config, temperature=0.0, max_tokens=max(300, min(config.max_tokens, 700)))
    client = OpenAICompatibleChatClient(parser_config)
    messages = [
        {
            "role": "system",
            "content": (
                "你是挖掘机售后诊断问题解析器。"
                "只做原始问题解析，不做术语标准化、不推断系统、不扩展同义词。"
                "你必须只输出一个 JSON 对象，不要 Markdown，不要解释。"
                "JSON schema: {"
                "\"fault_phrase\": string, "
                "\"component_text\": string, "
                "\"component_terms\": string[], "
                "\"required_component_terms\": string[], "
                "\"symptom_terms\": string[]"
                "}。"
                "规则：fault_phrase 是用户实际报修的故障短语；component_text 是明确部件；"
                "component_terms 是用于匹配证据的部件锚点，只能来自用户原文中的部件词；"
                "required_component_terms 是复合部件必须同时命中的短词，例如“风扇皮带”对应[\"风扇\",\"皮带\"]；"
                "symptom_terms 是异常表现词，例如异响、慢、漏油、报警。"
            ),
        },
        {
            "role": "user",
            "content": json.dumps({"query": query}, ensure_ascii=False),
        },
    ]
    result = client.complete(messages, service="query_parser", error_prefix="query_parser")
    payload = parse_query_constraints_json(result.text)
    debug = dict(result.debug)
    debug["raw_text_preview"] = result.text[:1000]
    return QueryParseResult(payload=payload, debug=debug)


def parse_query_constraints_json(text: str) -> dict[str, object]:
    """Parse a JSON object from a chat completion response."""

    raw = strip_json_code_fence(text).strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(raw[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("query parser output must be a JSON object")
    return payload


def strip_json_code_fence(text: str) -> str:
    """Remove common Markdown JSON fences from model output."""

    raw = str(text or "").strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1) if match else raw


def build_fallback_answer(
    *,
    query: str,
    evidence_items: list[dict[str, object]],
    part_candidates: list[dict[str, object]],
) -> str:
    """Build a deterministic answer when no LLM output is available."""

    lines = [f"问题：{query}", "", "可能故障原因："]
    manual_titles = [
        str(item.get("title"))
        for item in evidence_items
        if item.get("doc_type") == "manual_typical_fault" and item.get("title")
    ]
    if manual_titles:
        for title in manual_titles[:5]:
            lines.append(f"- {title}")
    else:
        lines.append("- 未从手册证据中召回明确故障标题。")
    lines.extend(["", "处理建议："])
    for item in evidence_items[:5]:
        preview = str(item.get("body_preview") or "")
        if preview:
            lines.append(f"- 参考 {item.get('doc_id')}: {preview[:180]}")
    lines.extend(["", "备件候选："])
    if part_candidates:
        for part in part_candidates:
            lines.append(
                "- "
                f"名称={part.get('part_name') or part.get('part_number_name') or '未知'}, "
                f"编码={part.get('part_code') or '未提供'}, "
                f"数量={part.get('quantity') or '未提供'}, "
                f"来源工单={part.get('work_order_id') or '未知'}"
            )
    else:
        lines.append("- 未召回明确备件候选。")
    lines.append("")
    lines.append("注意：以上备件来自历史工单或手册证据，未经过当前设备适配校验。")
    return "\n".join(lines)


def extract_rerank_results(payload: object) -> list[RerankItem]:
    """Extract rerank results from common DashScope/OpenAI-compatible shapes."""

    if not isinstance(payload, dict):
        return []
    raw_results = payload.get("results") or payload.get("data") or []
    if not isinstance(raw_results, list):
        return []
    results: list[RerankItem] = []
    for raw_item in raw_results:
        if not isinstance(raw_item, dict):
            continue
        document_payload = raw_item.get("document")
        document_text: str | None = None
        if isinstance(document_payload, str):
            document_text = document_payload
        elif isinstance(document_payload, dict) and document_payload.get("text") is not None:
            document_text = str(document_payload.get("text"))
        score = raw_item.get("relevance_score") if raw_item.get("relevance_score") is not None else raw_item.get("score", 0.0)
        results.append(
            RerankItem(
                index=int(raw_item.get("index", 0) or 0),
                score=float(score or 0.0),
                document=document_text,
            )
        )
    results.sort(key=lambda item: (-item.score, item.index))
    return results


def post_json(
    url: str,
    *,
    payload: dict[str, object],
    api_key: str,
    timeout_seconds: float,
    error_prefix: str,
    no_proxy_hosts: list[str] | None = None,
    service: str = "",
    provider: str = "",
    model: str = "",
    log_requests_enabled: bool = False,
    request_log_path: str = "logs/model_api_requests.jsonl",
) -> dict[str, Any]:
    """POST JSON to a model endpoint and parse the JSON response."""

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
    opener = (
        urllib.request.build_opener(urllib.request.ProxyHandler({}))
        if should_bypass_proxy(url, no_proxy_hosts or [])
        else urllib.request.build_opener()
    )
    start = time.time()
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
            service=service or error_prefix,
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
        raise RuntimeError(f"{error_prefix} HTTP {exc.code}: {body[:500]}") from exc
    except urllib.error.URLError as exc:
        append_model_api_request_log(
            enabled=log_requests_enabled,
            log_path=request_log_path,
            service=service or error_prefix,
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
        raise RuntimeError(f"{error_prefix} request failed: {exc}") from exc

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        compact = re.sub(r"\s+", " ", body[:500])
        append_model_api_request_log(
            enabled=log_requests_enabled,
            log_path=request_log_path,
            service=service or error_prefix,
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
        raise RuntimeError(f"{error_prefix} endpoint returned invalid JSON: {compact}") from exc
    if not isinstance(parsed, dict):
        append_model_api_request_log(
            enabled=log_requests_enabled,
            log_path=request_log_path,
            service=service or error_prefix,
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
        raise RuntimeError(f"{error_prefix} endpoint returned a non-object JSON payload")
    append_model_api_request_log(
        enabled=log_requests_enabled,
        log_path=request_log_path,
        service=service or error_prefix,
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
