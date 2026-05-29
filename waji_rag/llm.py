"""OpenAI-compatible LLM, rerank, and answer generation helpers."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from waji_rag.config import LLMConfig, RerankConfig
from waji_rag.embedding import should_bypass_proxy


@dataclass(slots=True)
class ModelCallResult:
    """One model call result with lightweight diagnostics."""

    text: str
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

    def complete(self, messages: list[dict[str, str]]) -> ModelCallResult:
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
            error_prefix="chat",
            no_proxy_hosts=self.config.no_proxy_hosts,
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
    result = client.complete(messages)
    if not result.text:
        result.text = build_fallback_answer(query=query, evidence_items=evidence_items, part_candidates=part_candidates)
    return result


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
    try:
        with opener.open(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{error_prefix} HTTP {exc.code}: {body[:500]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{error_prefix} request failed: {exc}") from exc

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        compact = re.sub(r"\s+", " ", body[:500])
        raise RuntimeError(f"{error_prefix} endpoint returned invalid JSON: {compact}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{error_prefix} endpoint returned a non-object JSON payload")
    return parsed
