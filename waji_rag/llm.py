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
from waji_rag.model_call_logging import append_model_api_request_log


@dataclass(slots=True)
class ModelCallResult:
    """One model call result with lightweight diagnostics."""

    text: str
    debug: dict[str, object]


class ModelProviderError(RuntimeError):
    """Raised when a model provider request or response fails."""


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
            raise ModelProviderError("chat endpoint returned no choices")
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


def judge_work_order_relevance(
    *,
    query: str,
    work_order_hit: dict[str, object],
    linked_parts: list[dict[str, object]],
    config: LLMConfig,
) -> ModelCallResult:
    """Ask the LLM whether one retrieved work order is truly relevant."""

    client = OpenAICompatibleChatClient(config)
    payload = json.dumps(
        {
            "query": query,
            "work_order": compact_json_payload(work_order_hit),
            "linked_parts": [compact_json_payload(item) for item in linked_parts],
        },
        ensure_ascii=False,
        indent=2,
    )
    messages = [
        {
            "role": "system",
            "content": (
                "你是挖掘机售后维修 RAG 的证据筛选器。你的任务是判断一个历史工单是否能支持当前问题。"
                "必须谨慎：相同异常词但部件或系统明显不同，应判为 unrelated；只有故障现象、部件、处理过程有直接关联时才判 high 或 medium。"
                "只返回 JSON，不要使用 Markdown。"
            ),
        },
        {
            "role": "user",
            "content": (
                "请按下面 schema 返回 JSON：\n"
                "{\n"
                '  "work_order_id": "...",\n'
                '  "related": true,\n'
                '  "relevance_level": "high|medium|low|unrelated|unknown",\n'
                '  "matched_reason": "...",\n'
                '  "repair_actions": ["..."],\n'
                '  "usable_parts": [{"name": "...", "code": "...", "quantity": "..."}],\n'
                '  "source_path": "..."\n'
                "}\n\n"
                f"输入证据：\n{payload}"
            ),
        },
    ]
    return client.complete(messages, service="harness_work_order_filter", error_prefix="harness work-order filter")


def select_manual_titles(
    *,
    query: str,
    manual_hits: list[dict[str, object]],
    config: LLMConfig,
) -> ModelCallResult:
    """Ask the LLM to select truly relevant manual titles from retrieved hits."""

    client = OpenAICompatibleChatClient(config)
    candidates = [
        {
            "doc_id": item.get("doc_id"),
            "title": item.get("title"),
            "doc_type": item.get("doc_type"),
            "source_path": item.get("source_path"),
            "score": item.get("score"),
        }
        for item in manual_hits
    ]
    payload = json.dumps({"query": query, "manual_candidates": candidates}, ensure_ascii=False, indent=2)
    messages = [
        {
            "role": "system",
            "content": (
                "你是挖掘机维修手册召回结果筛选器。只根据标题、doc_id 和路径判断是否真实相关。"
                "同样叫异响、漏油、动作慢，但部件明显不一致的手册应 rejected。只返回 JSON，不要使用 Markdown。"
            ),
        },
        {
            "role": "user",
            "content": (
                "请返回 JSON：\n"
                "{\n"
                '  "selected": [{"doc_id": "...", "relevance_level": "high|medium", "reason": "..."}],\n'
                '  "rejected": [{"doc_id": "...", "reason": "..."}]\n'
                "}\n\n"
                f"输入候选：\n{payload}"
            ),
        },
    ]
    return client.complete(messages, service="harness_manual_filter", error_prefix="harness manual filter")


def extract_answer_facts(
    *,
    query: str,
    selected_evidence: dict[str, object],
    selected_parts: list[dict[str, object]],
    config: LLMConfig,
) -> ModelCallResult:
    """Ask the LLM to normalize selected evidence into answer facts."""

    client = OpenAICompatibleChatClient(config)
    payload = json.dumps(
        {
            "query": query,
            "selected_evidence": compact_json_payload(selected_evidence, max_chars=18_000),
            "selected_parts": [compact_json_payload(item) for item in selected_parts],
        },
        ensure_ascii=False,
        indent=2,
    )
    messages = [
        {
            "role": "system",
            "content": (
                "你是挖掘机维修证据事实整理器。只能从输入证据抽取、归并事实，不要编造备件编码。"
                "有明确物料编码的备件只能来自历史工单结构化备件。手册或正文里仅提到名称但无编码的，放入 uncoded_possible_parts。"
                "只返回 JSON，不要使用 Markdown。"
            ),
        },
        {
            "role": "user",
            "content": (
                "请返回 JSON：\n"
                "{\n"
                '  "fault_code_facts": [],\n'
                '  "work_order_groups": [],\n'
                '  "manual_summaries": [],\n'
                '  "coded_parts": [],\n'
                '  "uncoded_possible_parts": []\n'
                "}\n\n"
                f"输入证据：\n{payload}"
            ),
        },
    ]
    return client.complete(messages, service="harness_fact_extraction", error_prefix="harness fact extraction")


def generate_harness_answer(
    *,
    query: str,
    final_context: dict[str, object],
    config: LLMConfig,
) -> ModelCallResult:
    """Generate the final answer from harness-approved context."""

    client = OpenAICompatibleChatClient(config)
    payload = json.dumps({"query": query, "final_context": compact_json_payload(final_context, max_chars=22_000)}, ensure_ascii=False, indent=2)
    messages = [
        {
            "role": "system",
            "content": (
                "你是挖掘机售后维修诊断助手。只能基于 final_context 回答，不要编造故障、工单、地址或备件编码。"
                "必须按固定顺序输出四段：1. 故障码匹配结果；2. 历史工单经验；3. 指导手册补充；4. 本次可能所需备件。"
                "备件部分先用表格列出有明确编码的备件，再用文字补充无编码的可能备件。"
            ),
        },
        {
            "role": "user",
            "content": (
                "请根据下面上下文生成中文答案。若某一类证据为空，要明确写“未召回”。"
                "历史工单备件必须提示：来自历史维修记录，未经过当前设备适配校验。\n\n"
                f"{payload}"
            ),
        },
    ]
    return client.complete(messages, service="harness_answer", error_prefix="harness answer")


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


def build_harness_fallback_answer(*, query: str, final_context: dict[str, object]) -> str:
    """Build the fixed-section answer from harness context without calling an LLM."""

    facts = final_context.get("facts") if isinstance(final_context.get("facts"), dict) else {}
    fault_code_facts = list_payload(facts.get("fault_code_facts") if isinstance(facts, dict) else None)
    work_order_groups = list_payload(facts.get("work_order_groups") if isinstance(facts, dict) else None)
    manual_summaries = list_payload(facts.get("manual_summaries") if isinstance(facts, dict) else None)
    coded_parts = list_payload(facts.get("coded_parts") if isinstance(facts, dict) else None)
    uncoded_parts = list_payload(facts.get("uncoded_possible_parts") if isinstance(facts, dict) else None)

    lines = [f"问题：{query}", "", "## 1. 故障码匹配结果"]
    if fault_code_facts:
        for item in fault_code_facts:
            lines.append(f"- {item.get('title') or item.get('doc_id')}: {item.get('summary') or '已召回故障码手册证据'}")
            if item.get("source_path"):
                lines.append(f"  原文地址：{item.get('source_path')}")
    else:
        lines.append("- 未召回故障码精确匹配结果。")

    lines.extend(["", "## 2. 历史工单经验"])
    if work_order_groups:
        for group in work_order_groups:
            lines.append(f"- {group.get('summary') or group.get('repair_action') or '相似历史处理方式'}")
            source_orders = group.get("source_work_orders")
            if isinstance(source_orders, list) and source_orders:
                lines.append(f"  支持工单：{', '.join(str(item) for item in source_orders)}")
    else:
        lines.append("- 未保留与当前问题直接相关的历史工单。")

    lines.extend(["", "## 3. 指导手册补充"])
    if manual_summaries:
        for item in manual_summaries:
            lines.append(f"- {item.get('title') or item.get('doc_id')}: {item.get('summary') or '请参考召回手册条目'}")
            if item.get("source_path"):
                lines.append(f"  原文地址：{item.get('source_path')}")
    else:
        lines.append("- 未保留相关指导手册。")

    lines.extend(["", "## 4. 本次可能所需备件"])
    if coded_parts:
        lines.append("| 备件名称 | 备件编码 | 数量 | 来源工单 |")
        lines.append("| --- | --- | --- | --- |")
        for part in coded_parts:
            lines.append(
                "| "
                f"{part.get('name') or part.get('part_name') or '未知'} | "
                f"{part.get('code') or part.get('part_code') or '未提供'} | "
                f"{part.get('quantity') or '未提供'} | "
                f"{part.get('source_work_order_id') or part.get('work_order_id') or '未知'} |"
            )
        lines.append("")
        lines.append("以上明确编码备件来自历史维修记录，未经过当前设备适配校验。")
    else:
        lines.append("- 未召回带明确编码的备件。")
    if uncoded_parts:
        lines.append("- 无编码但可作为排查方向的可能备件：" + "、".join(str(item.get("name") or item.get("part_name") or item) for item in uncoded_parts))
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


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from plain text or a fenced Markdown response."""

    candidate = strip_json_fence(text.strip())
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        parsed = json.loads(extract_json_slice(candidate))
    if not isinstance(parsed, dict):
        raise ValueError("model response JSON is not an object")
    return parsed


def strip_json_fence(text: str) -> str:
    """Remove a single Markdown JSON fence if the model returned one."""

    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else text


def extract_json_slice(text: str) -> str:
    """Extract the outermost JSON object slice from mixed model text."""

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("model response does not contain a JSON object")
    return text[start : end + 1]


def compact_json_payload(value: object, *, max_chars: int = 6000) -> object:
    """Return a JSON-like payload clipped to keep harness prompts bounded."""

    if isinstance(value, dict):
        return {str(key): compact_json_payload(item, max_chars=max_chars) for key, item in value.items()}
    if isinstance(value, list):
        return [compact_json_payload(item, max_chars=max_chars) for item in value]
    if isinstance(value, str):
        return value[:max_chars]
    return value


def list_payload(value: object) -> list[dict[str, object]]:
    """Return dict items from a JSON-like list."""

    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


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
        raise ModelProviderError(f"{error_prefix} HTTP {exc.code}: {body[:500]}") from exc
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
        raise ModelProviderError(f"{error_prefix} request failed: {exc}") from exc

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
        raise ModelProviderError(f"{error_prefix} endpoint returned invalid JSON: {compact}") from exc
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
        raise ModelProviderError(f"{error_prefix} endpoint returned a non-object JSON payload")
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
