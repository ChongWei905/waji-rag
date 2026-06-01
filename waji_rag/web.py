"""A local web UI for PostgreSQL-backed RAG debugging."""

from __future__ import annotations

import base64
import csv
import html
import hashlib
import json
import os
import posixpath
import threading
import time
import traceback
import uuid
import zipfile
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

from waji_rag import __version__
from waji_rag.config import (
    DEFAULT_DASHSCOPE_BASE_URL,
    DEFAULT_DASHSCOPE_RERANK_BASE_URL,
    DEFAULT_DOCARBOR_ENV_PATH,
    DEFAULT_MODEL_API_REQUEST_LOG_PATH,
    DEFAULT_OPENAI_BASE_URL,
    load_config,
    redact_secrets,
)
from waji_rag.pg_index import (
    DatabaseOptions,
    IngestPaused,
    PgEmbeddingBackfill,
    PgEmbeddingOptions,
    PgIngestBuilder,
    PgIngestOptions,
    PgPipelineOptions,
    PgSchemaManager,
    PgSearchOptions,
    clear_application_data,
    connect,
    create_task_schema,
    format_embedding_report_summary,
    format_ingest_report_summary,
    format_search_summary,
    json_param,
    redact_database_url,
    run_pg_pipeline,
    run_pg_search,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_WORK_ORDER_DIR = PROJECT_ROOT / "examples" / "demo_data" / "work_orders"
DEMO_MANUAL_DIR = PROJECT_ROOT / "examples" / "demo_data" / "manuals"
DEFAULT_QUERY = (
    "用户报修机器风扇皮带异响，请回答有可能是哪些故障导致的，如何解决，"
    "相应故障需要更换备件的详细信息（备件的编号及名称，备件编码，备件数量）"
)
DEFAULT_WEB_CONFIG_PATH = PROJECT_ROOT / "web_config.json"
DEFAULT_SHARED_CONFIG_PATH = PROJECT_ROOT / ".git" / "info" / "waji-rag-shared-config.json"
BATCH_EVAL_SHARE_ID_LENGTH = 12
_TASK_SCHEMA_LOCK = threading.Lock()
_TASK_SCHEMA_DATABASES: set[str] = set()
_TASK_DB_RETRY_SQLSTATES = {"40P01", "40001", "55P03"}
_TASK_DB_MAX_ATTEMPTS = 5
APP_CONFIG_SECTION_KEYS = ("retrieval", "embedding", "rerank", "llm", "answer")


INDEX_HTML = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Waji RAG Debug</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f5f2;
      --ink: #202124;
      --muted: #66685f;
      --line: #d8d5cb;
      --panel: #ffffff;
      --soft: #f7f7f3;
      --accent: #0f766e;
      --accent-strong: #115e59;
      --danger: #9f1239;
      --ok: #166534;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }}
    header {{
      border-bottom: 1px solid var(--line);
      background: #fff;
      padding: 14px 22px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
    }}
    h1 {{
      font-size: 18px;
      line-height: 1.2;
      margin: 0;
      font-weight: 700;
      letter-spacing: 0;
    }}
    main {{
      width: min(1440px, calc(100vw - 32px));
      margin: 18px auto 32px;
      display: grid;
      grid-template-columns: minmax(360px, 430px) 1fr;
      gap: 16px;
    }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }}
    section + section {{ margin-top: 16px; }}
    h2 {{
      font-size: 15px;
      margin: 0 0 14px;
      letter-spacing: 0;
    }}
    h3 {{
      font-size: 14px;
      margin: 16px 0 8px;
      letter-spacing: 0;
    }}
    label {{
      display: block;
      font-size: 13px;
      color: var(--muted);
      margin: 12px 0 6px;
    }}
    input, textarea, select {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
      font: inherit;
      background: #fff;
    }}
    input, select {{ height: 38px; }}
    textarea {{
      min-height: 92px;
      resize: vertical;
      line-height: 1.5;
    }}
    .row {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }}
    .checkline {{
      display: flex;
      align-items: center;
      gap: 8px;
      margin-top: 12px;
      color: var(--muted);
      font-size: 13px;
    }}
    .checkline input {{
      width: 16px;
      height: 16px;
      padding: 0;
      margin: 0;
    }}
    button {{
      border: 0;
      border-radius: 6px;
      background: var(--accent);
      color: #fff;
      min-height: 40px;
      padding: 0 14px;
      font-weight: 700;
      cursor: pointer;
      margin-top: 14px;
    }}
    button.secondary {{ background: #303030; }}
    button:hover {{ background: var(--accent-strong); }}
    button.secondary:hover {{ background: #1f1f1f; }}
    button:disabled {{
      opacity: .55;
      cursor: wait;
    }}
    .actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}
    .status {{
      min-height: 34px;
      padding: 9px 10px;
      border-radius: 6px;
      background: #f4f4ef;
      border: 1px solid var(--line);
      font-size: 13px;
      color: var(--muted);
      margin-bottom: 12px;
    }}
    .status.error {{
      color: var(--danger);
      border-color: #fecdd3;
      background: #fff1f2;
    }}
    .status.success {{
      color: var(--ok);
      border-color: #bbf7d0;
      background: #f0fdf4;
    }}
    .workflow-status {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      margin-top: 12px;
    }}
    .workflow-step {{
      min-height: 42px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
      background: var(--soft);
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }}
    .workflow-step.active {{
      border-color: #5eead4;
      background: #f0fdfa;
      color: var(--accent-strong);
    }}
    .workflow-step.done {{
      border-color: #bbf7d0;
      background: #f0fdf4;
      color: var(--ok);
    }}
    .tabs {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 12px;
    }}
    .tab {{
      height: 34px;
      margin: 0;
      background: #ecebe4;
      color: var(--ink);
      font-weight: 650;
    }}
    .tab.active {{
      background: var(--accent);
      color: #fff;
    }}
    .panel {{ display: none; }}
    .panel.active {{ display: block; }}
    pre {{
      margin: 0;
      min-height: 560px;
      max-height: 76vh;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: #fbfbf8;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      line-height: 1.5;
    }}
    .answer {{
      min-height: 220px;
      white-space: pre-wrap;
      line-height: 1.65;
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--soft);
    }}
    .grid-list {{
      display: grid;
      gap: 10px;
    }}
    .item {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--soft);
      padding: 12px;
    }}
    .item-title {{
      font-weight: 750;
      margin-bottom: 6px;
    }}
    .item-meta {{
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
      word-break: break-word;
    }}
    .item-body {{
      margin-top: 8px;
      line-height: 1.55;
      white-space: pre-wrap;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      min-height: 22px;
      padding: 0 8px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--muted);
      font-size: 12px;
      margin: 2px 4px 2px 0;
    }}
    .pill.ok {{ color: var(--ok); }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      background: #fff;
      border: 1px solid var(--line);
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 8px;
      text-align: left;
      vertical-align: top;
      word-break: break-word;
    }}
    th {{ background: var(--soft); }}
    .meta {{
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }}
    @media (max-width: 980px) {{
      main {{ grid-template-columns: 1fr; }}
      .row {{ grid-template-columns: 1fr; }}
      .workflow-status {{ grid-template-columns: 1fr 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Waji RAG Debug</h1>
    <div class="meta" id="version">loading</div>
  </header>
  <main>
    <div>
      <section>
        <h2>运行配置</h2>
        <label for="databaseUrl">Database URL</label>
        <input id="databaseUrl" value="postgresql://waji:waji@127.0.0.1:55432/waji_rag">
        <label for="envFile">Env 文件</label>
        <input id="envFile" placeholder="{html.escape(DEFAULT_DOCARBOR_ENV_PATH)}">
        <div class="actions">
          <button id="previewConfigBtn" class="secondary">预览配置</button>
          <button id="doctorBtn" class="secondary">环境检查</button>
          <button id="docArborEnvBtn" class="secondary">填入 DocArbor Env</button>
        </div>
      </section>
      <section>
        <h2>全流程</h2>
        <div class="actions">
          <button id="loadDemoBtn" class="secondary">加载 Demo 配置</button>
          <button id="runFullFlowBtn">一键跑全流程</button>
        </div>
        <div class="workflow-status">
          <div id="stepConfig" class="workflow-step">配置</div>
          <div id="stepInit" class="workflow-step">初始化 PG</div>
          <div id="stepIngest" class="workflow-step">入库</div>
          <div id="stepAsk" class="workflow-step">问答</div>
        </div>
      </section>
      <section>
        <h2>模型配置</h2>
        <label class="checkline" for="enableEmbedding">
          <input id="enableEmbedding" type="checkbox">
          <span>启用 embedding / hybrid</span>
        </label>
        <label for="embeddingModel">Embedding 模型</label>
        <input id="embeddingModel" value="text-embedding-v4">
        <div class="row">
          <div>
            <label for="embeddingDimensions">向量维度</label>
            <input id="embeddingDimensions" type="number" min="1" value="1024">
          </div>
          <div>
            <label for="embeddingBatchSize">批量大小</label>
            <input id="embeddingBatchSize" type="number" min="1" value="10">
          </div>
        </div>
        <label for="embeddingBaseUrl">Embedding Base URL</label>
        <input id="embeddingBaseUrl" value="{DEFAULT_DASHSCOPE_BASE_URL}">
        <label class="checkline" for="enableRerank">
          <input id="enableRerank" type="checkbox">
          <span>启用 rerank</span>
        </label>
        <label for="rerankModel">Rerank 模型</label>
        <input id="rerankModel" value="qwen3-rerank">
        <label for="rerankBaseUrl">Rerank Base URL</label>
        <input id="rerankBaseUrl" value="{DEFAULT_DASHSCOPE_RERANK_BASE_URL}">
        <label class="checkline" for="enableLlm">
          <input id="enableLlm" type="checkbox">
          <span>启用 LLM 答案生成</span>
        </label>
        <label for="llmModel">LLM 模型</label>
        <input id="llmModel" value="qwen3.5-plus">
        <label for="llmBaseUrl">LLM Base URL</label>
        <input id="llmBaseUrl" value="{DEFAULT_DASHSCOPE_BASE_URL}">
      </section>
      <section>
        <h2>入库</h2>
        <label for="workOrderDir">工单 TXT 目录</label>
        <input id="workOrderDir" value="{html.escape(str(DEMO_WORK_ORDER_DIR))}" placeholder="D:\\waji\\data\\work_orders">
        <label for="manualDir">手册 HTML/MD 目录</label>
        <input id="manualDir" value="{html.escape(str(DEMO_MANUAL_DIR))}" placeholder="D:\\waji\\data\\manuals">
        <div class="row">
          <div>
            <label for="workOrderLimit">工单上限</label>
            <input id="workOrderLimit" type="number" min="0" placeholder="留空全量">
          </div>
          <div>
            <label for="manualLimit">手册上限</label>
            <input id="manualLimit" type="number" min="0" placeholder="留空全量">
          </div>
        </div>
        <div class="row">
          <div>
            <label for="maxManualChars">手册块字符数</label>
            <input id="maxManualChars" type="number" min="200" value="1800">
          </div>
          <label class="checkline" for="ingestReset">
            <input id="ingestReset" type="checkbox" checked>
            <span>入库前重建</span>
          </label>
        </div>
        <div class="actions">
          <button id="initDbBtn">初始化 PG</button>
          <button id="ingestDbBtn">执行入库</button>
        </div>
      </section>
      <section>
        <h2>问答</h2>
        <label for="query">用户问题</label>
        <textarea id="query">{html.escape(DEFAULT_QUERY)}</textarea>
        <div class="row">
          <div>
            <label for="topK">每路 Top K</label>
            <input id="topK" type="number" min="1" value="1">
          </div>
          <div>
            <label for="evidenceTopK">答案证据数</label>
            <input id="evidenceTopK" type="number" min="1" value="4">
          </div>
        </div>
        <div class="actions">
          <button id="searchDbBtn" class="secondary">只检索</button>
          <button id="askDbBtn">端到端问答</button>
        </div>
      </section>
    </div>
    <section>
      <h2>运行结果</h2>
      <div id="status" class="status">ready</div>
      <div class="tabs">
        <button class="tab active" data-tab="answerPanel">答案</button>
        <button class="tab" data-tab="tracePanel">阶段日志</button>
        <button class="tab" data-tab="evidencePanel">召回证据</button>
        <button class="tab" data-tab="partsPanel">备件候选</button>
        <button class="tab" data-tab="rawPanel">原始 JSON</button>
      </div>
      <div id="answerPanel" class="panel active"><div id="answer" class="answer"></div></div>
      <div id="tracePanel" class="panel"><div id="trace" class="grid-list"></div></div>
      <div id="evidencePanel" class="panel"><div id="evidence" class="grid-list"></div></div>
      <div id="partsPanel" class="panel"><div id="parts"></div></div>
      <div id="rawPanel" class="panel"><pre id="output">{{}}</pre></div>
    </section>
  </main>
  <script>
    const $ = (id) => document.getElementById(id);
    const demoWorkOrderDir = {json.dumps(str(DEMO_WORK_ORDER_DIR), ensure_ascii=False)};
    const demoManualDir = {json.dumps(str(DEMO_MANUAL_DIR), ensure_ascii=False)};
    const docArborEnvPath = {json.dumps(DEFAULT_DOCARBOR_ENV_PATH, ensure_ascii=False)};
    const defaultQuery = {json.dumps(DEFAULT_QUERY, ensure_ascii=False)};

    async function postJson(url, payload) {{
      const response = await fetch(url, {{
        method: "POST",
        headers: {{"Content-Type": "application/json"}},
        body: JSON.stringify(payload)
      }});
      const data = await response.json();
      if (!response.ok) {{
        throw new Error(data.error || response.statusText);
      }}
      return data;
    }}

    async function getJson(url) {{
      const response = await fetch(url);
      const data = await response.json();
      if (!response.ok) {{
        throw new Error(data.error || response.statusText);
      }}
      return data;
    }}

    function configOverrides() {{
      return {{
        embedding: {{
          enabled: $("enableEmbedding").checked,
          provider: "dashscope",
          model: $("embeddingModel").value.trim(),
          base_url: $("embeddingBaseUrl").value.trim(),
          dimensions: $("embeddingDimensions").value ? Number($("embeddingDimensions").value) : null,
          batch_size: $("embeddingBatchSize").value ? Number($("embeddingBatchSize").value) : 10
        }},
        rerank: {{
          enabled: $("enableRerank").checked,
          provider: "dashscope",
          model: $("rerankModel").value.trim(),
          base_url: $("rerankBaseUrl").value.trim(),
          top_n: $("evidenceTopK").value ? Number($("evidenceTopK").value) : 8
        }},
        llm: {{
          enabled: $("enableLlm").checked,
          provider: "dashscope",
          model: $("llmModel").value.trim(),
          base_url: $("llmBaseUrl").value.trim(),
          max_tokens: 1400,
          temperature: 0
        }},
        answer: {{
          enabled: true,
          evidence_top_k: $("evidenceTopK").value ? Number($("evidenceTopK").value) : 8,
          include_debug: true
        }}
      }};
    }}

    function commonPayload() {{
      return {{
        database_url: $("databaseUrl").value.trim() || null,
        env_file: $("envFile").value.trim() || null,
        config_overrides: configOverrides()
      }};
    }}

    function ingestPayload() {{
      return {{
        ...commonPayload(),
        work_order_dir: $("workOrderDir").value.trim() || null,
        manual_dir: $("manualDir").value.trim() || null,
        reset: $("ingestReset").checked,
        resume: $("ingestResume").checked,
        work_order_limit: $("workOrderLimit").value ? Number($("workOrderLimit").value) : null,
        manual_limit: $("manualLimit").value ? Number($("manualLimit").value) : null,
        max_manual_chars: $("maxManualChars").value ? Number($("maxManualChars").value) : 1800
      }};
    }}

    function queryPayload() {{
      return {{
        ...commonPayload(),
        query: $("query").value.trim(),
        top_k: $("topK").value ? Number($("topK").value) : 5,
        debug: true
      }};
    }}

    function applyDemoDefaults() {{
      $("workOrderDir").value = demoWorkOrderDir;
      $("manualDir").value = demoManualDir;
      $("query").value = defaultQuery;
      $("topK").value = "1";
      $("evidenceTopK").value = "4";
      $("workOrderCandidateTopK").value = "50";
      $("workOrderMinRelativeScore").value = "0.45";
      $("workOrderMaxHits").value = "10";
      $("manualCandidateTopK").value = "30";
      $("manualMinRelativeScore").value = "0.55";
      $("manualMaxHits").value = "5";
      $("workOrderLimit").value = "";
      $("manualLimit").value = "";
      $("ingestReset").checked = true;
      $("enableEmbedding").checked = false;
      $("enableRerank").checked = false;
      $("enableLlm").checked = false;
      setWorkflowStep("config", "done");
      setWorkflowStep("init", "");
      setWorkflowStep("ingest", "");
      setWorkflowStep("ask", "");
      $("status").className = "status";
      $("status").textContent = "demo ready";
    }}

    function setWorkflowStep(step, state) {{
      const ids = {{
        config: "stepConfig",
        init: "stepInit",
        ingest: "stepIngest",
        ask: "stepAsk"
      }};
      const element = $(ids[step]);
      if (!element) return;
      element.className = "workflow-step" + (state ? " " + state : "");
    }}

    function resetWorkflowSteps() {{
      for (const step of ["config", "init", "ingest", "ask"]) setWorkflowStep(step, "");
    }}

    async function runButton(button, task) {{
      button.disabled = true;
      $("status").className = "status";
      $("status").textContent = "running";
      try {{
        renderResult(await task());
      }} catch (error) {{
        $("status").className = "status error";
        $("status").textContent = String(error);
        $("output").textContent = JSON.stringify({{error: String(error)}}, null, 2);
      }} finally {{
        button.disabled = false;
      }}
    }}

    async function runFullFlow(button) {{
      button.disabled = true;
      resetWorkflowSteps();
      setWorkflowStep("config", "active");
      $("status").className = "status";
      $("status").textContent = "checking config";
      try {{
        const preview = await postJson("/api/config-preview", commonPayload());
        setWorkflowStep("config", "done");

        setWorkflowStep("init", "active");
        $("status").textContent = "initializing PostgreSQL";
        const initResult = await postJson("/api/init-db", {{
          database_url: $("databaseUrl").value.trim() || null,
          reset: false
        }});
        setWorkflowStep("init", "done");

        setWorkflowStep("ingest", "active");
        $("status").textContent = "ingesting evidence";
        const ingestResult = await postJson("/api/ingest-db", ingestPayload());
        setWorkflowStep("ingest", "done");

        setWorkflowStep("ask", "active");
        $("status").textContent = "retrieving and answering";
        const askResult = await postJson("/api/ask-db", queryPayload());
        askResult.workflow = {{
          config: preview,
          init: initResult,
          ingest: ingestResult
        }};
        setWorkflowStep("ask", "done");
        $("status").className = "status success";
        renderResult({{...askResult, summary: "全流程完成 · " + (askResult.summary || "ok")}});
      }} catch (error) {{
        $("status").className = "status error";
        $("status").textContent = String(error);
        $("output").textContent = JSON.stringify({{error: String(error)}}, null, 2);
      }} finally {{
        button.disabled = false;
      }}
    }}

    function renderResult(data) {{
      $("status").className = "status";
      $("status").textContent = data.summary || "ok";
      $("output").textContent = JSON.stringify(data, null, 2);
      const result = data.result || data;
      const answerPayload = result.answer || {{}};
      $("answer").textContent = answerPayload.text || data.summary || JSON.stringify(data, null, 2);
      renderTrace(result.trace || []);
      renderEvidence(result);
      renderParts(result.part_candidates || (result.retrieval && result.retrieval.part_candidates) || []);
    }}

    function renderTrace(trace) {{
      $("trace").innerHTML = "";
      if (!trace.length) {{
        $("trace").innerHTML = '<div class="item">暂无阶段日志</div>';
        return;
      }}
      for (const item of trace) {{
        const div = document.createElement("div");
        div.className = "item";
        div.innerHTML = `
          <div class="item-title">${{escapeHtml(item.stage || "")}} <span class="pill ${{item.status === "ok" ? "ok" : ""}}">${{escapeHtml(item.status || "")}}</span></div>
          <div class="item-meta">${{escapeHtml(item.timestamp || "")}}</div>
          <div class="item-body">${{escapeHtml(JSON.stringify(item.details || {{}}, null, 2))}}</div>
        `;
        $("trace").appendChild(div);
      }}
    }}

    function renderEvidence(result) {{
      $("evidence").innerHTML = "";
      const retrieval = result.retrieval || result;
      const channels = retrieval.channels || {{}};
      const names = ["work_orders", "manual_typical_faults", "manual_fault_codes", "part_evidence"];
      let count = 0;
      for (const name of names) {{
        const hits = channels[name] || [];
        for (const [index, hit] of hits.entries()) {{
          count += 1;
          const div = document.createElement("div");
          div.className = "item";
          const terms = (hit.matched_terms || []).slice(0, 12).map(t => `<span class="pill">${{escapeHtml(t.term || "")}} · ${{escapeHtml(t.field || "")}}</span>`).join("");
          div.innerHTML = `
            <div class="item-title">${{escapeHtml(name)}} #${{index + 1}} · ${{escapeHtml(hit.title || "")}}</div>
            <div class="item-meta">score=${{escapeHtml(String(hit.score ?? ""))}} · doc_id=${{escapeHtml(hit.doc_id || "")}} · source=${{escapeHtml(hit.source_path || "")}}</div>
            <div>${{terms}}</div>
            <div class="item-body">${{escapeHtml(hit.body_preview || "")}}</div>
          `;
          $("evidence").appendChild(div);
        }}
      }}
      if (!count) $("evidence").innerHTML = '<div class="item">暂无召回证据</div>';
    }}

    function renderParts(parts) {{
      if (!parts.length) {{
        $("parts").innerHTML = '<div class="item">暂无备件候选</div>';
        return;
      }}
      const rows = parts.map(part => `
        <tr>
          <td>${{escapeHtml(part.work_order_id || "")}}</td>
          <td>${{escapeHtml(part.part_number_name || part.part_name || "")}}</td>
          <td>${{escapeHtml(part.part_number || "")}}</td>
          <td>${{escapeHtml(part.part_code || "")}}</td>
          <td>${{escapeHtml(part.quantity || "")}}</td>
          <td>${{escapeHtml(part.source_path || "")}}</td>
        </tr>
      `).join("");
      $("parts").innerHTML = `
        <table>
          <thead><tr><th>工单</th><th>备件编号及名称 / 名称</th><th>备件编号</th><th>备件编码</th><th>数量</th><th>来源</th></tr></thead>
          <tbody>${{rows}}</tbody>
        </table>
      `;
    }}

    function escapeHtml(value) {{
      return String(value ?? "").replace(/[&<>"']/g, ch => ({{
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
      }}[ch]));
    }}

    document.querySelectorAll(".tab").forEach(tab => {{
      tab.addEventListener("click", () => {{
        document.querySelectorAll(".tab").forEach(item => item.classList.remove("active"));
        document.querySelectorAll(".panel").forEach(item => item.classList.remove("active"));
        tab.classList.add("active");
        $(tab.dataset.tab).classList.add("active");
      }});
    }});

    $("doctorBtn").addEventListener("click", () => runButton($("doctorBtn"), () => getJson("/api/doctor")));
    $("docArborEnvBtn").addEventListener("click", () => {{
      $("envFile").value = docArborEnvPath;
      $("status").className = "status";
      $("status").textContent = "env path filled";
    }});
    $("loadDemoBtn").addEventListener("click", applyDemoDefaults);
    $("runFullFlowBtn").addEventListener("click", () => runFullFlow($("runFullFlowBtn")));
    $("previewConfigBtn").addEventListener("click", () => runButton($("previewConfigBtn"), () => postJson("/api/config-preview", commonPayload())));
    $("initDbBtn").addEventListener("click", () => runButton($("initDbBtn"), () => postJson("/api/init-db", {{
      database_url: $("databaseUrl").value.trim() || null,
      reset: false
    }})));
    $("ingestDbBtn").addEventListener("click", () => runButton($("ingestDbBtn"), () => postJson("/api/ingest-db", ingestPayload())));
    $("searchDbBtn").addEventListener("click", () => runButton($("searchDbBtn"), () => postJson("/api/search-db", queryPayload())));
    $("askDbBtn").addEventListener("click", () => runButton($("askDbBtn"), () => postJson("/api/ask-db", queryPayload())));

    getJson("/api/doctor").then(data => {{
      $("version").textContent = data.waji_rag_version + " · " + data.platform;
    }}).catch(() => {{}});
  </script>
</body>
</html>
"""


def build_redesigned_index_html() -> str:
    """Build the process-first debugging UI."""

    template = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Waji RAG Workbench</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f3f4f6;
      --surface: #ffffff;
      --soft: #f8fafc;
      --ink: #111827;
      --muted: #64748b;
      --line: #d7dde5;
      --line-strong: #b9c2cf;
      --accent: #0f766e;
      --accent-soft: #e6fffb;
      --ok: #166534;
      --warn: #92400e;
      --danger: #9f1239;
      --code: #0f172a;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      overflow-x: hidden;
    }
    button, input, select, textarea {
      font: inherit;
    }
    button {
      min-height: 36px;
      border: 1px solid transparent;
      border-radius: 6px;
      padding: 0 12px;
      color: #fff;
      background: var(--accent);
      font-weight: 700;
      cursor: pointer;
    }
    button:hover { filter: brightness(.95); }
    button:disabled { opacity: .56; cursor: wait; }
    button.secondary {
      color: var(--ink);
      background: #fff;
      border-color: var(--line-strong);
    }
    button.ghost {
      color: var(--muted);
      background: transparent;
      border-color: var(--line);
    }
    button.danger {
      color: #fff;
      background: var(--danger);
      border-color: var(--danger);
    }
    input, select, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      padding: 8px 10px;
    }
    select {
      min-height: 38px;
    }
    textarea {
      min-height: 86px;
      resize: vertical;
      line-height: 1.5;
    }
    label {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin: 12px 0 6px;
    }
    header {
      min-height: 58px;
      padding: 12px 18px;
      background: #fff;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
    }
    h1 {
      margin: 0;
      font-size: 18px;
      letter-spacing: 0;
    }
    h2 {
      margin: 0 0 10px;
      font-size: 15px;
      letter-spacing: 0;
    }
    h3 {
      margin: 0;
      font-size: 13px;
      letter-spacing: 0;
    }
    .header-actions, .actions, .query-actions {
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 8px;
    }
    .panel-title-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 10px;
    }
    .panel-title-row h2 {
      margin: 0;
    }
    .header-actions button, .query-actions button {
      white-space: nowrap;
    }
    .meta {
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }
    .page-shell {
      width: min(1800px, calc(100vw - 28px));
      margin: 14px auto 28px;
      display: grid;
      grid-template-columns: minmax(240px, 300px) minmax(0, 1fr);
      gap: 12px;
      align-items: start;
    }
    .page-shell.history-collapsed {
      grid-template-columns: minmax(0, 1fr);
    }
    .page-shell.batch-home-mode {
      grid-template-columns: minmax(0, 1fr);
    }
    .page-shell.history-collapsed .question-sidebar {
      display: none;
    }
    .page-shell.batch-home-mode .question-sidebar {
      display: none;
    }
    .page-shell:not(.history-collapsed) #openQuestionSidebarBtn {
      display: none;
    }
    main {
      min-width: 0;
    }
    .query-band {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      display: grid;
      grid-template-columns: 1fr 280px;
      gap: 14px;
      align-items: end;
    }
    .question-sidebar, .question-main-toolbar {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .question-sidebar {
      position: sticky;
      top: 12px;
      padding: 10px;
      display: grid;
      gap: 10px;
    }
    .question-sidebar-head, .question-main-toolbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }
    .question-sidebar-head h2 {
      margin: 0;
    }
    .question-sidebar-head button, .question-main-toolbar button {
      min-height: 32px;
      padding: 0 10px;
      font-size: 12px;
    }
    .question-main {
      min-width: 0;
      display: grid;
      gap: 10px;
    }
    .question-main-toolbar {
      min-height: 42px;
      padding: 6px 8px;
    }
    .current-question-title {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-size: 13px;
      font-weight: 760;
    }
    .question-tabs {
      min-width: 0;
      display: grid;
      gap: 8px;
      max-height: min(520px, calc(100vh - 260px));
      overflow-y: auto;
      padding-right: 2px;
    }
    .batch-metric-selector {
      display: grid;
      gap: 5px;
      margin-bottom: 8px;
    }
    .batch-metric-selector label {
      font-size: 12px;
      color: var(--muted);
      margin: 0;
    }
    .batch-metric-selector select {
      min-height: 34px;
    }
    .question-tab {
      width: 100%;
      min-height: 44px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--ink);
      padding: 7px 9px;
      text-align: left;
      display: grid;
      gap: 3px;
    }
    .question-tab.active {
      border-color: #5eead4;
      background: var(--accent-soft);
    }
    .question-tab.pass {
      border-color: #86efac;
      background: #f0fdf4;
    }
    .question-tab.warn {
      border-color: #facc15;
      background: #fefce8;
    }
    .question-tab.fail, .question-tab.error {
      border-color: #fecaca;
      background: #fff1f2;
    }
    .question-tab.skipped {
      border-style: dashed;
      background: var(--soft);
    }
    .question-tab.overview {
      position: sticky;
      top: 0;
      z-index: 2;
      border-color: var(--line-strong);
      background: #fff;
      box-shadow: 0 8px 14px rgba(15, 23, 42, .08);
    }
    .question-tab.pass.active, .question-tab.warn.active, .question-tab.fail.active, .question-tab.error.active, .question-tab.skipped.active {
      box-shadow: inset 0 0 0 1px #0f766e;
    }
    .question-tab-title {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-size: 13px;
      font-weight: 760;
    }
    .question-tab-meta {
      color: var(--muted);
      font-size: 11px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .query-tools {
      display: grid;
      gap: 10px;
      align-content: end;
    }
    .query-tools .query-actions {
      justify-content: flex-start;
    }
    .qa-config-summary {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--soft);
      color: var(--muted);
      padding: 9px 10px;
      font-size: 12px;
      line-height: 1.45;
    }
    .status {
      min-height: 36px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--soft);
      color: var(--muted);
      padding: 9px 10px;
      font-size: 13px;
      margin-top: 10px;
    }
    .status.success {
      color: var(--ok);
      background: #f0fdf4;
      border-color: #bbf7d0;
    }
    .status.error {
      color: var(--danger);
      background: #fff1f2;
      border-color: #fecdd3;
    }
    .view-tabs {
      display: flex;
      gap: 8px;
      margin-top: 12px;
      flex-wrap: wrap;
    }
    .view-tab {
      color: var(--ink);
      background: #fff;
      border-color: var(--line-strong);
    }
    .view-tab.active {
      color: #fff;
      background: var(--accent);
      border-color: var(--accent);
    }
    .view {
      display: none;
    }
    .view.active {
      display: block;
    }
    .build-dashboard {
      display: grid;
      grid-template-columns: minmax(0, .9fr) minmax(320px, 1.1fr);
      gap: 12px;
    }
    .progress-track {
      height: 12px;
      border: 1px solid var(--line);
      border-radius: 999px;
      overflow: hidden;
      background: #fff;
      margin: 12px 0;
    }
    .progress-bar {
      width: 0%;
      height: 100%;
      background: var(--accent);
      transition: width .2s ease;
    }
    .stat-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
    }
    .stat-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fff;
      min-width: 0;
    }
    .stat-value {
      font-weight: 800;
      font-size: 20px;
      line-height: 1.2;
    }
    .doc-info {
      display: grid;
      gap: 8px;
    }
    .failure-panel {
      margin-top: 12px;
      border: 1px solid #fed7aa;
      border-radius: 8px;
      background: #fffbeb;
      padding: 10px;
      display: grid;
      gap: 8px;
    }
    .failure-panel.hidden {
      display: none;
    }
    .failure-row {
      border: 1px solid #fde68a;
      border-radius: 6px;
      background: #fff;
      padding: 8px;
      min-width: 0;
    }
    .failure-path {
      font-weight: 740;
      font-size: 13px;
      overflow-wrap: anywhere;
    }
    .failure-error {
      margin-top: 4px;
      color: var(--danger);
      font-size: 12px;
      line-height: 1.45;
      overflow-wrap: anywhere;
    }
    .panel.hidden {
      display: none;
    }
    .hidden {
      display: none !important;
    }
    .workspace {
      margin-top: 14px;
      display: grid;
      grid-template-columns: minmax(220px, 260px) minmax(0, 1fr);
      gap: 14px;
      align-items: start;
    }
    .workspace.hidden {
      display: none;
    }
    .workspace.batch-row-mode {
      grid-template-columns: minmax(0, 1fr);
    }
    .workspace.batch-row-mode .stage-rail {
      display: none;
    }
    .workspace.batch-row-mode .content-grid {
      grid-column: 1 / -1;
    }
    .stage-rail, .panel {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-width: 0;
    }
    .stage-rail {
      position: sticky;
      top: 12px;
      overflow: hidden;
    }
    .task-list {
      display: grid;
      gap: 8px;
      max-height: min(620px, calc(100vh - 250px));
      overflow: auto;
      padding-right: 2px;
    }
    .task-card {
      width: 100%;
      min-width: 0;
      min-height: 72px;
      text-align: left;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--ink);
      padding: 10px;
      display: grid;
      gap: 5px;
      overflow: hidden;
    }
    .task-card.active {
      border-color: #5eead4;
      background: var(--accent-soft);
    }
    .task-card.failed {
      border-color: #fecdd3;
      background: #fff1f2;
    }
    .task-card.running {
      border-color: #bfdbfe;
      background: #eff6ff;
    }
    .task-card.completed_with_errors {
      border-color: #fed7aa;
      background: #fffbeb;
    }
    .task-card.stopped {
      border-color: #fed7aa;
      background: #fffbeb;
    }
    .task-line {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      min-width: 0;
    }
    .task-title {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-size: 13px;
      font-weight: 760;
    }
    .task-subtitle {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
      max-height: 32px;
      overflow: hidden;
      overflow-wrap: anywhere;
    }
    .stage-list {
      display: grid;
      gap: 8px;
    }
    .stage-node {
      width: 100%;
      min-width: 0;
      min-height: 64px;
      text-align: left;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--ink);
      padding: 10px;
      display: grid;
      gap: 4px;
      overflow: hidden;
    }
    .stage-node.active {
      border-color: #5eead4;
      background: var(--accent-soft);
    }
    .stage-node.selected {
      outline: 2px solid #0f766e;
      outline-offset: 2px;
    }
    .stage-node.done {
      border-color: #bbf7d0;
      background: #f0fdf4;
    }
    .stage-node.fallback, .stage-node.skipped, .stage-node.filtered {
      border-color: #fed7aa;
      background: #fffbeb;
    }
    .stage-node.error {
      border-color: #fecdd3;
      background: #fff1f2;
    }
    .stage-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      font-weight: 760;
      font-size: 13px;
      min-width: 0;
    }
    .stage-title span:first-child {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .stage-note {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
      max-height: 34px;
      overflow: hidden;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    .pill {
      display: inline-flex;
      min-height: 22px;
      align-items: center;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 0 8px;
      color: var(--muted);
      background: #fff;
      font-size: 12px;
      white-space: nowrap;
    }
    .pill.ok { color: var(--ok); border-color: #bbf7d0; }
    .pill.warn { color: var(--warn); border-color: #fed7aa; }
    .content-grid {
      display: grid;
      gap: 14px;
      min-width: 0;
    }
    .answer-layout {
      display: grid;
      grid-template-columns: minmax(0, 1.35fr) minmax(320px, .65fr);
      gap: 12px;
    }
    .answer-box, .json-box {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--soft);
      padding: 12px;
      min-height: 220px;
      white-space: pre-wrap;
      line-height: 1.62;
      overflow: auto;
    }
    .answer-box.markdown-body {
      white-space: normal;
    }
    .markdown-body h1, .markdown-body h2, .markdown-body h3, .markdown-body h4 {
      margin: 10px 0 6px;
      line-height: 1.35;
    }
    .markdown-body h1 { font-size: 20px; }
    .markdown-body h2 { font-size: 17px; }
    .markdown-body h3, .markdown-body h4 { font-size: 15px; }
    .markdown-body p {
      margin: 0 0 9px;
    }
    .markdown-body ul, .markdown-body ol {
      margin: 0 0 10px 20px;
      padding: 0;
    }
    .markdown-body li {
      margin: 3px 0;
    }
    .markdown-body blockquote {
      margin: 8px 0;
      padding: 6px 10px;
      border-left: 3px solid var(--line-strong);
      background: #fff;
      color: var(--muted);
      border-radius: 6px;
    }
    .markdown-body pre {
      margin: 8px 0 10px;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      overflow: auto;
      white-space: pre;
    }
    .markdown-body code {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      font-size: 12px;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 5px;
      padding: 1px 4px;
    }
    .markdown-body pre code {
      border: 0;
      padding: 0;
      background: transparent;
    }
    .markdown-body table {
      width: 100%;
      border-collapse: collapse;
      margin: 8px 0 12px;
      background: #fff;
      font-size: 13px;
    }
    .markdown-body th, .markdown-body td {
      border: 1px solid var(--line);
      padding: 7px 8px;
      text-align: left;
      vertical-align: top;
    }
    .markdown-body th {
      background: #f8fafc;
      font-weight: 760;
    }
    .markdown-body a {
      color: var(--accent);
      text-decoration: none;
      overflow-wrap: anywhere;
    }
    .markdown-body a:hover {
      text-decoration: underline;
    }
    .part-box {
      display: grid;
      gap: 8px;
    }
    .part-row, .evidence-row {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 10px;
    }
    .part-evidence-hit {
      background: #fff;
    }
    .part-evidence-fields {
      display: grid;
      gap: 8px;
    }
    .part-field {
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--soft);
      padding: 8px;
    }
    .part-field-value {
      margin-top: 3px;
      font-size: 13px;
      font-weight: 760;
      overflow-wrap: anywhere;
    }
    .row-title {
      font-weight: 760;
      margin-bottom: 5px;
    }
    .row-meta {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
      word-break: break-word;
    }
    .retrieval-board {
      display: grid;
      grid-template-columns: repeat(4, minmax(220px, 1fr));
      gap: 10px;
    }
    .route-column {
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      overflow: hidden;
    }
    .route-head {
      padding: 10px;
      background: var(--soft);
      border-bottom: 1px solid var(--line);
    }
    .route-head .row-meta {
      margin-top: 4px;
    }
    .route-body {
      display: grid;
      gap: 8px;
      padding: 10px;
      max-height: 520px;
      overflow: auto;
    }
    .hit {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px;
      background: #fff;
    }
    .hit-title {
      font-weight: 740;
      font-size: 13px;
      margin-bottom: 5px;
    }
    .hit-preview {
      margin-top: 7px;
      color: #334155;
      font-size: 12px;
      line-height: 1.5;
      white-space: pre-wrap;
    }
    .term-list {
      margin-top: 6px;
      display: flex;
      flex-wrap: wrap;
      gap: 4px;
    }
    .inspector {
      display: grid;
      grid-template-columns: minmax(0, .9fr) minmax(0, 1.1fr);
      gap: 12px;
    }
    .batch-retry-panel {
      grid-column: 1 / -1;
      margin-top: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--soft);
      padding: 10px;
      display: grid;
      gap: 10px;
    }
    .batch-retry-controls {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
    }
    .batch-retry-panel label {
      margin-top: 0;
    }
    .batch-retry-actions {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      flex-wrap: wrap;
    }
    .batch-comparison-card {
      border-top: 1px solid var(--line);
      padding-top: 10px;
      display: grid;
      gap: 8px;
    }
    .batch-comparison-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }
    .batch-comparison-grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) minmax(220px, .7fr);
      grid-template-areas:
        "expected-parts retrieved-parts summary"
        "expected-work-orders retrieved-work-orders summary";
      gap: 8px;
      align-items: stretch;
    }
    .batch-comparison-field {
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 9px;
      display: grid;
      gap: 6px;
    }
    .batch-comparison-expected-parts {
      grid-area: expected-parts;
    }
    .batch-comparison-retrieved-parts {
      grid-area: retrieved-parts;
    }
    .batch-comparison-expected-work-orders {
      grid-area: expected-work-orders;
    }
    .batch-comparison-retrieved-work-orders {
      grid-area: retrieved-work-orders;
    }
    .batch-comparison-summary {
      grid-area: summary;
      align-content: start;
      background: #fbfdff;
    }
    .metric-conclusion {
      display: grid;
      gap: 5px;
      padding-top: 8px;
      border-top: 1px solid var(--line);
    }
    .batch-answer-preview {
      max-height: 260px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 9px;
      font-size: 13px;
    }
    .batch-comparison-field .empty {
      padding: 10px;
    }
    .match-status {
      border-radius: 8px;
      padding: 9px;
      font-weight: 780;
    }
    .match-status.pass {
      color: var(--ok);
      background: #f0fdf4;
      border: 1px solid #bbf7d0;
    }
    .match-status.warn {
      color: #a16207;
      background: #fefce8;
      border: 1px solid #fde68a;
    }
    .match-status.fail {
      color: var(--danger);
      background: #fff1f2;
      border: 1px solid #fecdd3;
    }
    .json-box {
      min-height: 360px;
      max-height: 640px;
      margin: 0;
      color: #dbeafe;
      background: var(--code);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      word-break: break-word;
    }
    .empty {
      color: var(--muted);
      border: 1px dashed var(--line-strong);
      border-radius: 8px;
      padding: 18px;
      text-align: center;
      background: var(--soft);
    }
    .modal-backdrop {
      position: fixed;
      inset: 0;
      display: none;
      align-items: center;
      justify-content: center;
      background: rgba(15, 23, 42, .42);
      z-index: 20;
      padding: 18px;
    }
    .modal-backdrop.open {
      display: flex;
    }
    .modal {
      width: min(860px, 100%);
      max-height: min(760px, calc(100vh - 36px));
      display: flex;
      flex-direction: column;
      overflow: hidden;
      border-radius: 8px;
      border: 1px solid var(--line);
      background: #fff;
      box-shadow: 0 22px 60px rgba(15, 23, 42, .24);
    }
    .history-modal {
      width: min(760px, 100%);
    }
    .batch-eval-modal {
      width: min(1180px, 100%);
    }
    .qa-config-modal {
      width: min(780px, 100%);
    }
    .modal-head, .modal-foot {
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    .modal-head {
      flex: 0 0 auto;
      background: #fff;
      position: relative;
      z-index: 1;
    }
    .modal-foot {
      flex: 0 0 auto;
      background: #fff;
      border-top: 1px solid var(--line);
      border-bottom: 0;
      justify-content: flex-end;
    }
    .modal-body {
      padding: 16px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px 14px;
    }
    .modal-body, .history-body, .batch-eval-body {
      min-height: 0;
      overflow: auto;
    }
    .modal-body .full {
      grid-column: 1 / -1;
    }
    .history-body {
      padding: 16px;
      display: grid;
      gap: 12px;
    }
    .batch-eval-body {
      padding: 16px;
      display: grid;
      gap: 14px;
    }
    .batch-home-layout {
      display: grid;
      grid-template-columns: minmax(0, 1.15fr) minmax(320px, .85fr);
      gap: 12px;
      margin-top: 14px;
    }
    .batch-detail-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
    }
    .batch-eval-run-list {
      display: grid;
      gap: 8px;
      max-height: min(560px, calc(100vh - 320px));
      overflow: auto;
      padding-right: 2px;
    }
    .batch-eval-run-card {
      width: 100%;
      text-align: left;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--ink);
      padding: 10px;
      display: grid;
      gap: 5px;
    }
    .batch-eval-run-card.active {
      border-color: #5eead4;
      background: var(--accent-soft);
    }
    .batch-eval-run-card.completed_with_errors, .batch-eval-run-card.stopped {
      border-color: #fed7aa;
      background: #fffbeb;
    }
    .batch-eval-run-card.failed {
      border-color: #fecdd3;
      background: #fff1f2;
    }
    .batch-eval-controls {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }
    .batch-eval-toolbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      flex-wrap: wrap;
    }
    .batch-eval-results {
      display: grid;
      gap: 8px;
      max-height: min(430px, calc(100vh - 340px));
      overflow: auto;
      padding-right: 2px;
    }
    .eval-row {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 10px;
      display: grid;
      gap: 8px;
    }
    .eval-row.pass {
      border-color: #86efac;
      background: #f0fdf4;
    }
    .eval-row.warn {
      border-color: #facc15;
      background: #fefce8;
    }
    .eval-row.fail, .eval-row.error {
      border-color: #fecaca;
      background: #fff1f2;
    }
    .eval-row.skipped {
      border-style: dashed;
      background: var(--soft);
    }
    .eval-row-head {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
    }
    .eval-row-grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 8px;
    }
    .eval-field {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      background: rgba(255, 255, 255, .72);
      min-width: 0;
      overflow-wrap: anywhere;
    }
    .checkline {
      min-height: 38px;
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 13px;
      margin-top: 12px;
    }
    .checkline input {
      width: 16px;
      height: 16px;
      margin: 0;
      padding: 0;
    }
    @media (max-width: 1180px) {
      .query-band, .workspace, .answer-layout, .inspector, .build-dashboard, .page-shell { grid-template-columns: 1fr; }
      .stage-rail { position: static; }
      .question-sidebar { position: static; }
      .retrieval-board { grid-template-columns: 1fr 1fr; }
    }
    @media (max-width: 760px) {
      header, .header-actions { align-items: stretch; }
      header { flex-direction: column; }
      .query-tools, .retrieval-board, .modal-body, .batch-eval-controls, .batch-retry-controls, .batch-comparison-grid, .eval-row-grid, .batch-home-layout { grid-template-columns: 1fr; }
      .batch-comparison-grid {
        grid-template-areas:
          "expected-parts"
          "retrieved-parts"
          "expected-work-orders"
          "retrieved-work-orders"
          "summary";
      }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Waji RAG Workbench</h1>
      <div id="version" class="meta">loading</div>
    </div>
    <div class="header-actions">
      <button id="runBuildBtn" class="secondary">构建</button>
      <button id="runSearchBtn" class="secondary">检索</button>
      <button id="runAnswerBtn" class="secondary">回答</button>
      <button id="runFullFlowBtn">全流程</button>
      <button id="openBatchEvalBtn" class="secondary">批量评测</button>
      <button id="openQuestionSidebarHeaderBtn" class="secondary">回答历史</button>
      <button id="openHistoryBtn" class="secondary">历史任务</button>
      <button id="openConfigBtn" class="secondary">配置</button>
      <button id="doctorBtn" class="ghost">环境检查</button>
    </div>
  </header>

  <div id="pageShell" class="page-shell">
    <aside id="questionSidebar" class="question-sidebar">
      <div class="question-sidebar-head">
        <h2 id="questionSidebarTitle">历史回答</h2>
        <button id="closeQuestionSidebarBtn" class="ghost">收起</button>
      </div>
      <div id="batchMetricSelector" class="batch-metric-selector hidden">
        <label for="batchMetricSelect">评测指标</label>
        <select id="batchMetricSelect">
          <option value="part_recall">备件召回率</option>
          <option value="work_order_recall">工单召回率</option>
        </select>
      </div>
      <div id="questionTabs" class="question-tabs"></div>
      <button id="newQuestionBtn" class="secondary">新问题</button>
    </aside>

    <main>
    <div class="view-tabs">
      <button id="buildViewBtn" class="view-tab active">索引构建</button>
      <button id="qaViewBtn" class="view-tab">检索与回答</button>
      <button id="batchViewBtn" class="view-tab">批量评测</button>
    </div>

    <div id="status" class="status">页面已载入。可以单独运行“构建 / 检索 / 回答”，也可以运行“全流程”。</div>

    <section id="buildView" class="view active">
      <div class="panel">
        <div class="panel-title-row">
          <h2>索引构建</h2>
          <div class="actions">
            <button id="resumeBuildBtn" class="secondary">继续构建</button>
            <button id="retryFailedBtn" class="secondary">重试失败条目</button>
            <button id="runEmbeddingBtn" class="secondary">补Embedding</button>
            <button id="pauseTaskBtn" class="ghost">暂停任务</button>
            <button id="clearDataBtn" class="danger">清空数据</button>
          </div>
        </div>
        <div class="build-dashboard">
          <div>
            <div class="doc-info">
              <div class="evidence-row">
                <div class="row-title">文档来源</div>
                <div id="buildSources" class="row-meta">尚未运行。</div>
              </div>
              <div class="evidence-row">
                <div class="row-title">当前进度</div>
                <div id="buildProgressText" class="row-meta">等待构建任务。</div>
                <div class="progress-track"><div id="buildProgressBar" class="progress-bar"></div></div>
                <div id="buildCurrentFile" class="row-meta"></div>
              </div>
            </div>
          </div>
          <div>
            <div id="buildStats" class="stat-grid"></div>
          </div>
        </div>
        <div id="failedItemsPanel" class="failure-panel hidden"></div>
      </div>
    </section>

    <section id="qaView" class="view">
      <div class="question-main">
        <div class="question-main-toolbar">
          <button id="openQuestionSidebarBtn" class="secondary">历史回答</button>
          <div id="currentQuestionTitle" class="current-question-title">当前问题</div>
        </div>
        <section class="query-band">
          <div>
            <label for="query">当前问题</label>
            <textarea id="query">__DEFAULT_QUERY_TEXT__</textarea>
          </div>
          <div class="query-tools">
            <div id="qaConfigSummary" class="qa-config-summary">回答参数尚未载入。</div>
            <div class="query-actions">
              <button id="openQaConfigBtn" class="secondary">回答参数</button>
              <button id="runQuestionSearchBtn" class="secondary">检索当前问题</button>
              <button id="runQuestionAnswerBtn">回答当前问题</button>
            </div>
          </div>
        </section>
      </div>
    </section>

    <section id="batchView" class="view">
      <div id="batchHomePanel" class="batch-home-layout">
        <div class="panel">
          <div class="panel-title-row">
            <h2>发起批量评测</h2>
            <div class="actions">
              <button id="runBatchEvalBtn">开始评测</button>
            </div>
          </div>
          <div class="batch-eval-body">
            <div>
              <label for="batchEvalCsv">CSV / XLSX 文件</label>
              <input id="batchEvalCsv" type="file" accept=".csv,text/csv,.xlsx,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet">
              <div id="batchEvalFileMeta" class="row-meta">尚未载入 CSV / XLSX。第一行会作为列名。</div>
            </div>
            <div class="batch-eval-controls">
              <div>
                <label for="batchEvalRunMode">评测模式</label>
                <select id="batchEvalRunMode">
                  <option value="search" selected>只检索</option>
                  <option value="answer">检索 + 回答</option>
                </select>
              </div>
              <label class="checkline" for="batchEvalWorkOrderHybrid">
                <input id="batchEvalWorkOrderHybrid" type="checkbox">
                <span>历史工单使用 hybrid</span>
              </label>
              <label class="checkline" for="batchEvalManualHybrid">
                <input id="batchEvalManualHybrid" type="checkbox">
                <span>故障手册使用 hybrid</span>
              </label>
              <div>
                <label for="batchEvalTopK">故障码 Top K</label>
                <input id="batchEvalTopK" type="number" min="1" value="1">
              </div>
              <div>
                <label for="batchEvalManualCandidateTopK">手册候选上限</label>
                <input id="batchEvalManualCandidateTopK" type="number" min="1" value="30">
              </div>
              <div>
                <label for="batchEvalManualMinRelativeScore">手册相对阈值</label>
                <input id="batchEvalManualMinRelativeScore" type="number" min="0" max="1" step="0.05" value="0.55">
              </div>
              <div>
                <label for="batchEvalManualMaxHits">手册最大返回</label>
                <input id="batchEvalManualMaxHits" type="number" min="0" value="5">
              </div>
              <div>
                <label for="batchEvalWorkOrderCandidateTopK">工单候选上限</label>
                <input id="batchEvalWorkOrderCandidateTopK" type="number" min="1" value="50">
              </div>
              <div>
                <label for="batchEvalWorkOrderMinRelativeScore">工单相对阈值</label>
                <input id="batchEvalWorkOrderMinRelativeScore" type="number" min="0" max="1" step="0.05" value="0.45">
              </div>
              <div>
                <label for="batchEvalWorkOrderMaxHits">工单最大返回</label>
                <input id="batchEvalWorkOrderMaxHits" type="number" min="0" value="10">
              </div>
              <div>
                <label for="batchEvalConcurrency">并发数</label>
                <select id="batchEvalConcurrency">
                  <option value="1">1</option>
                  <option value="4" selected>4</option>
                  <option value="8">8</option>
                  <option value="16">16</option>
                </select>
              </div>
              <div>
                <label for="batchEvalQuestionColumn">问题列</label>
                <select id="batchEvalQuestionColumn"></select>
              </div>
              <div>
                <label for="batchEvalWorkOrderIdColumn">预期工单ID列</label>
                <select id="batchEvalWorkOrderIdColumn"></select>
              </div>
              <div>
                <label for="batchEvalPartNameColumn">新件备件名称列</label>
                <select id="batchEvalPartNameColumn"></select>
              </div>
              <div>
                <label for="batchEvalPartCodeColumn">新件物料编码列</label>
                <select id="batchEvalPartCodeColumn"></select>
              </div>
              <div>
                <label for="batchEvalPartQuantityColumn">新件数量列</label>
                <select id="batchEvalPartQuantityColumn"></select>
              </div>
            </div>
            <div id="batchEvalHomeStatus" class="row-meta">载入 CSV / XLSX 并配置列、评测模式、检索参数和并发数后点击开始评测。手册召回由候选上限、相对阈值和最大返回控制；故障码 Top K 只控制故障码精确匹配返回数。</div>
          </div>
        </div>
        <div class="panel">
          <div class="panel-title-row">
            <h2>历史批量评测</h2>
            <div class="actions">
              <button id="refreshBatchEvalRunsBtn" class="secondary">刷新</button>
            </div>
          </div>
          <div id="batchEvalRunList" class="batch-eval-run-list"><div class="empty">暂无批量评测记录</div></div>
        </div>
      </div>
      <div id="batchDetailPanel" class="panel hidden">
        <div class="batch-detail-head">
          <div>
            <h2 id="batchEvalDetailTitle">批量评测详情</h2>
            <div id="batchEvalStatus" class="row-meta">选择左侧评测问题即可重现对应检索结果。</div>
          </div>
          <div class="actions">
            <button id="backToBatchHomeBtn" class="secondary">返回批量评测</button>
            <button id="stopBatchEvalBtn" class="secondary" disabled>停止</button>
            <button id="exportBatchEvalBtn" class="secondary" disabled>导出结果</button>
          </div>
        </div>
        <div class="progress-track"><div id="batchEvalProgressBar" class="progress-bar"></div></div>
        <div id="batchEvalSummary" class="stat-grid"></div>
        <div id="batchEvalResults" class="batch-eval-results"><div class="empty">暂无评测结果</div></div>
      </div>
    </section>

    <div id="workspace" class="workspace">
      <section id="batchRetryPanel" class="panel batch-retry-panel hidden"></section>

      <aside class="stage-rail">
        <h2 id="stageListTitle">构建阶段</h2>
        <div id="stageList" class="stage-list"></div>
      </aside>

      <div class="content-grid">
        <section id="answerPanel" class="panel">
          <h2>答案与备件</h2>
          <div class="answer-layout">
            <div id="answer" class="answer-box">尚未运行。</div>
            <div id="parts" class="part-box"><div class="empty">暂无备件候选</div></div>
          </div>
        </section>

        <section id="retrievalPanel" class="panel hidden">
          <h2>多路召回</h2>
          <div id="retrievalBoard" class="retrieval-board"></div>
        </section>

        <section id="inspectorPanel" class="panel">
          <h2>阶段返回</h2>
          <div class="inspector">
            <div id="stageSummary"></div>
            <pre id="stageJson" class="json-box">{}</pre>
          </div>
        </section>

        <section id="evidencePanel" class="panel hidden">
          <h2 id="evidencePanelTitle">答案生成依据</h2>
          <div id="selectedEvidence" class="part-box"><div class="empty">暂无选中证据</div></div>
        </section>
      </div>
    </div>
    </main>
  </div>

  <div id="taskHistoryModal" class="modal-backdrop">
    <div class="modal history-modal">
      <div class="modal-head">
        <h2 id="taskListTitle">历史任务清单</h2>
        <div class="actions">
          <button id="refreshTasksBtn" class="ghost">刷新</button>
          <button id="closeHistoryBtn" class="ghost">关闭</button>
        </div>
      </div>
      <div class="history-body">
        <div id="taskList" class="task-list"><div class="empty">暂无历史任务</div></div>
      </div>
    </div>
  </div>

  <div id="configModal" class="modal-backdrop">
    <div class="modal">
      <div class="modal-head">
        <h2>运行配置</h2>
        <button id="closeConfigBtn" class="ghost">关闭</button>
      </div>
      <div class="modal-body">
        <div class="full part-box">
          <div class="row-title">连接与数据源由服务端配置文件控制</div>
          <div id="serverConfigSummary" class="row-meta">页面不会保存或提交数据库、模型 API、代理、密钥、数据目录等连接项。</div>
        </div>
        <div>
          <label for="workOrderLimit">工单上限</label>
          <input id="workOrderLimit" type="number" min="0" placeholder="留空全量">
        </div>
        <div>
          <label for="manualLimit">手册上限</label>
          <input id="manualLimit" type="number" min="0" placeholder="留空全量">
        </div>
        <div>
          <label for="maxManualChars">手册块字符数</label>
          <input id="maxManualChars" type="number" min="200" value="1800">
        </div>
        <label class="checkline" for="ingestReset">
          <input id="ingestReset" type="checkbox" checked>
          <span>入库前重建</span>
        </label>
        <label class="checkline" for="ingestResume">
          <input id="ingestResume" type="checkbox" checked>
          <span>断点续跑</span>
        </label>

        <label class="checkline" for="enableEmbedding">
          <input id="enableEmbedding" type="checkbox">
          <span>启用 embedding / hybrid</span>
        </label>

        <label class="checkline" for="enableRerank">
          <input id="enableRerank" type="checkbox">
          <span>启用 rerank</span>
        </label>

        <label class="checkline" for="enableLlm">
          <input id="enableLlm" type="checkbox">
          <span>启用 LLM 答案生成</span>
        </label>
        <div class="full">
          <label for="configImportFile">页面偏好导入</label>
          <input id="configImportFile" type="file" accept="application/json,.json">
        </div>
      </div>
      <div class="modal-foot">
        <button id="loadDemoBtn" class="secondary">加载默认页面参数</button>
        <button id="exportConfigBtn" class="secondary">导出页面偏好</button>
        <button id="importConfigBtn" class="secondary">导入页面偏好</button>
        <button id="previewConfigBtn" class="secondary">预览服务端配置</button>
        <button id="saveConfigBtn">保存页面偏好</button>
      </div>
    </div>
  </div>

  <div id="qaConfigModal" class="modal-backdrop">
    <div class="modal qa-config-modal">
      <div class="modal-head">
        <h2>回答参数</h2>
        <button id="closeQaConfigBtn" class="ghost">关闭</button>
      </div>
      <div class="modal-body">
        <div class="full part-box">
          <div class="row-title">检索方式</div>
          <div class="row-meta">工单和故障手册可以分别选择是否使用 hybrid。勾选后本次检索会尝试启用 embedding；如果服务端没有可用 embedding，会自动退回 BM25。</div>
        </div>
        <label class="checkline" for="qaWorkOrderHybrid">
          <input id="qaWorkOrderHybrid" type="checkbox">
          <span>历史工单使用 hybrid</span>
        </label>
        <label class="checkline" for="qaManualHybrid">
          <input id="qaManualHybrid" type="checkbox">
          <span>故障手册使用 hybrid</span>
        </label>
        <div>
          <label for="topK">手册 / 故障码 Top K</label>
          <input id="topK" type="number" min="1" value="1">
        </div>
        <div>
          <label for="evidenceTopK">答案证据数</label>
          <input id="evidenceTopK" type="number" min="1" value="4">
        </div>
        <div>
          <label for="manualCandidateTopK">手册候选上限</label>
          <input id="manualCandidateTopK" type="number" min="1" value="30">
        </div>
        <div>
          <label for="manualMinRelativeScore">手册相对阈值</label>
          <input id="manualMinRelativeScore" type="number" min="0" max="1" step="0.05" value="0.55">
        </div>
        <div>
          <label for="manualMaxHits">手册最大返回</label>
          <input id="manualMaxHits" type="number" min="0" value="5">
        </div>
        <div>
          <label for="workOrderCandidateTopK">工单候选上限</label>
          <input id="workOrderCandidateTopK" type="number" min="1" value="50">
        </div>
        <div>
          <label for="workOrderMinRelativeScore">工单相对阈值</label>
          <input id="workOrderMinRelativeScore" type="number" min="0" max="1" step="0.05" value="0.45">
        </div>
        <div>
          <label for="workOrderMaxHits">工单最大返回</label>
          <input id="workOrderMaxHits" type="number" min="0" value="10">
        </div>
      </div>
      <div class="modal-foot">
        <button id="saveQaConfigBtn">保存回答参数</button>
      </div>
    </div>
  </div>

  <script>
    const $ = (id) => document.getElementById(id);
    const demoWorkOrderDir = __DEMO_WORK_ORDER_DIR_JSON__;
    const demoManualDir = __DEMO_MANUAL_DIR_JSON__;
    const docArborEnvPath = __DOCARBOR_ENV_PATH_JSON__;
    const defaultQuery = __DEFAULT_QUERY_JSON__;
    const dashscopeBaseUrl = __DASHSCOPE_BASE_URL_JSON__;
    const openaiBaseUrl = __OPENAI_BASE_URL_JSON__;
    const localEmbeddingBaseUrl = "http://127.0.0.1:8888/v1";
    const defaultModelApiRequestLogPath = __MODEL_API_REQUEST_LOG_PATH_JSON__;
    const savedConfigKey = "waji-rag-workbench-config-v1";
    const routeBatchEvalShareId = routeBatchEvalShareIdFromPath(window.location.pathname);
    const channels = [
      ["work_orders", "历史工单", "doc_type=work_order；字段权重优先 reported_issue，再看 solution/raw_text。"],
      ["manual_typical_faults", "典型故障手册", "doc_type=manual_typical_fault；优先 fault_title/file_name，再看正文 chunk。"],
      ["manual_fault_codes", "故障码手册", "问题出现故障码时先精确匹配 fault_code，未出现故障码则为空。"],
      ["part_evidence", "备件证据", "不独立检索备件索引；仅展示历史工单召回后关联出的备件。"]
    ];
    const buildStageOrder = [
      ["config", "配置解析", "读取页面配置、env 和模型开关"],
      ["init", "初始化 PG", "创建 PostgreSQL / pgvector 表结构"],
      ["ingest", "构建索引", "解析工单、HTML 转 Markdown、入库、建 BM25/向量"],
      ["embedding", "补Embedding", "扫描已有索引文档，为缺失向量的文档补齐 embedding"]
    ];
    const qaStageOrder = [
      ["retrieval", "多路召回", "历史工单、手册、故障码、备件证据分路召回"],
      ["work_order_filter", "工单筛选", "LLM 并发判断历史工单是否真实相关"],
      ["manual_filter", "手册筛选", "LLM 根据手册标题筛选相关指导手册"],
      ["fact_extraction", "事实整理", "归并故障码、工单处理经验、手册摘要和备件"],
      ["answer", "答案生成", "按固定维修诊断结构生成最终答复"]
    ];
    const batchEvalBaseMetricOptions = [
      ["part_recall", "备件召回率"],
      ["work_order_recall", "工单召回率"]
    ];
    const batchEvalAnswerMetricOption = ["answer_part_recall", "回答备件召回率"];
    let appState = {
      stages: {},
      selectedStage: "config",
      lastResult: null,
      currentTaskId: null,
      currentTask: null,
      tasks: [],
      buildStages: {},
      buildSelectedStage: "config",
      questionTabs: [],
      activeQuestionTabId: null,
      questionTabCounter: 0,
      questionSidebarOpen: true,
      activeView: "build",
      buildPollTimer: null,
      answerPollTimer: null,
      stageSelectionLocked: false,
      batchEvalRuns: [],
      activeBatchEvalTaskId: null,
      activeBatchEvalTask: null,
      activeBatchEvalRowNumber: null,
      activeBatchEvalRetryTaskId: null,
      activeBatchEvalReplay: null,
      batchRetry: {
        topK: 1,
        workOrderHybrid: false,
        manualHybrid: false,
        workOrderCandidateTopK: 50,
        workOrderMinRelativeScore: 0.45,
        workOrderMaxHits: 10,
        manualCandidateTopK: 30,
        manualMinRelativeScore: 0.55,
        manualMaxHits: 5,
        running: false
      },
      batchEval: {
        headers: [],
        rows: [],
        results: [],
        running: false,
        stopRequested: false,
        fileName: "",
        rowCount: 0,
        taskId: null,
        shareId: null,
        useSharedDatabase: false,
        settings: null,
        partColumns: null,
        workOrderColumn: null,
        selectedMetric: "part_recall",
        questionIndex: null,
        persistPromise: Promise.resolve(),
        persistError: null
      }
    };

    function stageOrderForView(view = appState.activeView) {
      return view === "qa" || view === "batch" ? qaStageOrder : buildStageOrder;
    }

    function initialStageForView(view = appState.activeView) {
      return view === "qa" || view === "batch" ? "retrieval" : "config";
    }

    function stageIdSetForView(view = appState.activeView) {
      return new Set(stageOrderForView(view).map(([id]) => id));
    }

    function createStageState(view = "build", selectedStage = null) {
      const stages = {};
      for (const [id] of stageOrderForView(view)) stages[id] = {status: "pending", data: null, summary: ""};
      const stageIds = stageIdSetForView(view);
      const safeSelectedStage = selectedStage && stageIds.has(selectedStage) ? selectedStage : initialStageForView(view);
      return {stages, selectedStage: safeSelectedStage};
    }

    function hydrateStageState(view, stages = {}, selectedStage = null) {
      const hydrated = createStageState(view, selectedStage);
      const stageIds = stageIdSetForView(view);
      for (const [id, value] of Object.entries(stages || {})) {
        if (!stageIds.has(id)) continue;
        hydrated.stages[id] = value || hydrated.stages[id];
      }
      if (!stageIds.has(hydrated.selectedStage)) hydrated.selectedStage = initialStageForView(view);
      return hydrated;
    }

    function makeQuestionTab(query = "") {
      const state = createStageState("qa", "retrieval");
      appState.questionTabCounter += 1;
      return {
        id: `q-${Date.now()}-${appState.questionTabCounter}`,
        query: String(query || "").trim() || defaultQuery,
        title: questionTitle(query || defaultQuery),
        stages: state.stages,
        selectedStage: state.selectedStage,
        lastResult: null,
        searchTaskId: null,
        answerTaskId: null,
        status: "draft",
        persisted: false,
        loadingResult: false,
        updatedAt: new Date().toISOString()
      };
    }

    function makeQuestionTabFromServer(item) {
      const tab = makeQuestionTab(item.query || defaultQuery);
      tab.id = item.id || tab.id;
      tab.title = item.title || questionTitle(tab.query);
      tab.searchTaskId = item.search_task_id || null;
      tab.answerTaskId = item.answer_task_id || null;
      tab.status = item.status || "persisted";
      tab.persisted = true;
      tab.updatedAt = item.updated_at || tab.updatedAt;
      return tab;
    }

    function questionTitle(query) {
      const text = String(query || "").trim().replace(/\s+/g, " ");
      return text.length > 24 ? `${text.slice(0, 24)}...` : text || "新问题";
    }

    function activeQuestionTab() {
      return appState.questionTabs.find(tab => tab.id === appState.activeQuestionTabId) || null;
    }

    function updateCurrentQuestionTitle() {
      const title = $("currentQuestionTitle");
      if (!title) return;
      const tab = activeQuestionTab();
      title.textContent = tab ? (tab.title || questionTitle(tab.query)) : "当前问题";
    }

    function ensureQuestionTab(query = $("query").value) {
      if (!appState.questionTabs.length) {
        const tab = makeQuestionTab(query || defaultQuery);
        appState.questionTabs.push(tab);
        appState.activeQuestionTabId = tab.id;
        return tab;
      }
      const current = activeQuestionTab();
      if (current) return current;
      appState.activeQuestionTabId = appState.questionTabs[0].id;
      return appState.questionTabs[0];
    }

    function ensureQuestionTabForQuery(query, options = {}) {
      const normalized = String(query || "").trim();
      let tab = appState.questionTabs.find(item => item.query.trim() === normalized);
      if (!tab) {
        tab = makeQuestionTab(normalized || defaultQuery);
        appState.questionTabs.push(tab);
      }
      activateQuestionTab(tab.id, options);
      return tab;
    }

    function syncActiveQuestionInput() {
      const tab = activeQuestionTab();
      if (!tab || document.activeElement !== $("query")) return;
      if (tab.searchTaskId || tab.answerTaskId) return;
      tab.query = $("query").value.trim();
      tab.title = questionTitle(tab.query);
      tab.status = "draft";
      tab.updatedAt = new Date().toISOString();
      renderQuestionTabs();
      updateCurrentQuestionTitle();
    }

    function syncActiveQuestionState() {
      if (appState.activeView !== "qa") return;
      const tab = activeQuestionTab();
      if (!tab) return;
      tab.stages = appState.stages;
      tab.selectedStage = appState.selectedStage;
      tab.lastResult = appState.lastResult;
      tab.updatedAt = new Date().toISOString();
      renderQuestionTabs();
    }

    function activateQuestionTab(tabId, options = {}) {
      const tab = appState.questionTabs.find(item => item.id === tabId);
      if (!tab) return;
      if (appState.activeView === "build") {
        appState.buildStages = appState.stages;
        appState.buildSelectedStage = appState.selectedStage;
      } else {
        syncActiveQuestionState();
      }
      const tabState = hydrateStageState("qa", tab.stages, tab.selectedStage || "retrieval");
      tab.stages = tabState.stages;
      tab.selectedStage = tabState.selectedStage;
      appState.activeQuestionTabId = tab.id;
      appState.activeView = "qa";
      $("query").value = tab.query;
      appState.stages = tab.stages;
      appState.selectedStage = tab.selectedStage;
      appState.lastResult = tab.lastResult || null;
      appState.currentTaskId = tab.answerTaskId || tab.searchTaskId || appState.currentTaskId;
      renderQuestionTabs();
      updateCurrentQuestionTitle();
      renderStages();
      renderStageInspector();
      renderQuestionResult(tab);
      if (options.loadPersisted !== false) loadPersistedQuestionResult(tab);
      saveConfigToLocalStorage();
    }

    function renderQuestionTabs() {
      if (appState.activeView === "batch") {
        renderBatchEvalQuestionTabs();
        return;
      }
      const container = $("questionTabs");
      if (!container) return;
      if (!appState.questionTabs.length) {
        container.innerHTML = '<div class="empty">暂无问题</div>';
        updateCurrentQuestionTitle();
        return;
      }
      container.innerHTML = "";
      for (const tab of questionTabsNewestFirst()) {
        const button = document.createElement("button");
        button.className = `question-tab ${tab.id === appState.activeQuestionTabId ? "active" : ""}`;
        button.innerHTML = `
          <div class="question-tab-title">${escapeHtml(tab.title || questionTitle(tab.query))}</div>
          <div class="question-tab-meta">${escapeHtml(questionTabMeta(tab))}</div>
        `;
        button.addEventListener("click", () => activateQuestionTab(tab.id));
        container.appendChild(button);
      }
      updateCurrentQuestionTitle();
    }

    function questionTabsNewestFirst() {
      return [...appState.questionTabs].sort((left, right) => questionTabTime(right) - questionTabTime(left));
    }

    function questionTabTime(tab) {
      const value = Date.parse(tab && tab.updatedAt ? tab.updatedAt : "");
      return Number.isFinite(value) ? value : 0;
    }

    function setQuestionSidebar(open, options = {}) {
      appState.questionSidebarOpen = Boolean(open);
      const shell = $("pageShell");
      if (shell) shell.classList.toggle("history-collapsed", !appState.questionSidebarOpen);
      const headerButton = $("openQuestionSidebarHeaderBtn");
      if (headerButton) headerButton.style.display = appState.questionSidebarOpen ? "none" : "";
      if (options.save !== false) saveConfigToLocalStorage();
    }

    function updateSidebarChrome() {
      const isBatch = appState.activeView === "batch";
      $("questionSidebarTitle").textContent = isBatch ? "本次评测问题" : "历史回答";
      $("openQuestionSidebarHeaderBtn").textContent = isBatch ? "评测问题" : "回答历史";
      $("openQuestionSidebarBtn").textContent = isBatch ? "评测问题" : "历史回答";
      $("newQuestionBtn").classList.toggle("hidden", isBatch);
      const metricSelector = $("batchMetricSelector");
      if (metricSelector) metricSelector.classList.toggle("hidden", !isBatch);
      renderBatchMetricSelector();
    }

    function renderBatchMetricSelector(settings = appState.activeBatchEvalTaskId ? appState.batchEval.settings : $("batchEvalRunMode").value) {
      const select = $("batchMetricSelect");
      if (!select) return;
      const options = batchEvalMetricOptionsForSettings(settings);
      const metric = selectedBatchEvalMetric(settings);
      select.innerHTML = options.map(([value, label]) => `<option value="${escapeHtml(value)}">${escapeHtml(label)}</option>`).join("");
      if (select.value !== metric) select.value = metric;
    }

    function batchEvalMetricOptionsForSettings(settings = appState.batchEval.settings) {
      const options = [...batchEvalBaseMetricOptions];
      if (batchEvalRunModeFromSettings(settings) === "answer") options.push(batchEvalAnswerMetricOption);
      return options;
    }

    function batchEvalRunModeFromSettings(settings = appState.batchEval.settings) {
      if (typeof settings === "string") return settings === "answer" ? "answer" : "search";
      const mode = settings && (settings.runMode || settings.run_mode);
      if (mode === "answer") return "answer";
      if (mode === "search") return "search";
      const selector = $("batchEvalRunMode");
      return selector && selector.value === "answer" ? "answer" : "search";
    }

    function batchEvalRunModeFromItem(item) {
      if (!item) return batchEvalRunModeFromSettings();
      if (item.runMode === "answer" || item.run_mode === "answer") return "answer";
      if (item.runMode === "search" || item.run_mode === "search") return "search";
      if (item.answerText || item.answer_text) return "answer";
      return batchEvalRunModeFromSettings(item.settings || item.batch_settings || appState.batchEval.settings);
    }

    function batchEvalRunModeLabel(mode = batchEvalRunModeFromSettings()) {
      return batchEvalRunModeFromSettings(mode) === "answer" ? "检索+回答" : "只检索";
    }

    function questionTabMeta(tab) {
      const bits = [];
      if (tab.searchTaskId) bits.push(`检索 #${tab.searchTaskId}`);
      if (tab.answerTaskId) bits.push(`回答 #${tab.answerTaskId}`);
      bits.push(tab.status || "draft");
      return bits.join(" · ");
    }

    function questionStatusFromTask(task, fallback = "persisted") {
      if (task && task.status === "completed") {
        return task.task_type === "search" ? "searched" : "answered";
      }
      return (task && task.status) || fallback;
    }

    function renderQuestionResult(tab) {
      const result = tab && tab.lastResult ? tab.lastResult : null;
      if (!result) {
        renderAnswer({});
        renderParts([]);
        renderRetrievalBoard({channels: {}, mode: "", top_k: ""});
        renderSelectedEvidence([]);
        return;
      }
      if (result.answer || result.retrieval) {
        renderAnswer(result.answer || {});
        renderParts(result.part_candidates || (result.retrieval && result.retrieval.part_candidates) || []);
        renderRetrievalBoard(result.retrieval || result);
        renderSelectedEvidence(result.selected_evidence || []);
      } else {
        renderRetrievalBoard(result);
        renderParts(result.part_candidates || []);
        $("answer").textContent = "已完成检索。请查看“多路召回”和“阶段返回”。";
        renderSelectedEvidence([]);
      }
    }

    async function loadPersistedQuestionResult(tab) {
      if (!tab || tab.loadingResult || tab.lastResult) return;
      const taskId = tab.answerTaskId || tab.searchTaskId;
      if (!taskId || ["searching", "answering"].includes(tab.status)) return;
      tab.loadingResult = true;
      try {
        const data = await postJson("/api/task", taskPayload({task_id: taskId}));
        const task = data.task;
        if (!task) return;
        const target = appState.questionTabs.find(item => item.id === tab.id);
        if (!target) return;
        target.status = questionStatusFromTask(task, target.status);
        target.lastResult = task.result && task.result.result ? task.result.result : (task.result || null);
        target.updatedAt = task.updated_at || target.updatedAt;
        if (appState.activeView === "qa" && appState.activeQuestionTabId === target.id) {
          appState.currentTaskId = task.id;
          if (task.task_type === "search") {
            renderSearchResult(task.result || {});
          } else {
            renderPipelineResult(task.result || {});
            if (["running", "pause_requested"].includes(task.status)) {
              startAnswerPolling(task.id, target.id);
            }
          }
          syncActiveQuestionState();
        }
      } catch (error) {
        setStatus(`加载历史回答失败：${error}`, "error");
      } finally {
        tab.loadingResult = false;
      }
    }

    function createNewQuestionTab() {
      const tab = makeQuestionTab(defaultQuery);
      tab.query = "";
      tab.title = "新问题";
      tab.status = "draft";
      appState.questionTabs.push(tab);
      activateQuestionTab(tab.id);
      $("query").focus();
      saveConfigToLocalStorage();
    }

    function configOverrides(options = {}) {
      const retrievalOverrides = options.retrieval && typeof options.retrieval === "object" ? options.retrieval : {};
      const hybridRequested = retrievalOverrides.work_order_mode === "hybrid" || retrievalOverrides.manual_mode === "hybrid";
      const autoEmbeddingForHybrid = Boolean((options.queryRuntime || options.autoEmbeddingForHybrid) && hybridRequested);
      return {
        retrieval: {
          work_order_candidate_top_k: $("workOrderCandidateTopK").value ? Number($("workOrderCandidateTopK").value) : 50,
          work_order_min_relative_score: $("workOrderMinRelativeScore").value ? Number($("workOrderMinRelativeScore").value) : 0.45,
          work_order_max_hits: $("workOrderMaxHits").value ? Number($("workOrderMaxHits").value) : 10,
          manual_candidate_top_k: $("manualCandidateTopK").value ? Number($("manualCandidateTopK").value) : 30,
          manual_min_relative_score: $("manualMinRelativeScore").value ? Number($("manualMinRelativeScore").value) : 0.55,
          manual_max_hits: $("manualMaxHits").value ? Number($("manualMaxHits").value) : 5,
          ...retrievalOverrides
        },
        embedding: {
          enabled: $("enableEmbedding").checked || autoEmbeddingForHybrid
        },
        rerank: {
          enabled: $("enableRerank").checked,
          top_n: $("evidenceTopK").value ? Number($("evidenceTopK").value) : 8
        },
        llm: {
          enabled: $("enableLlm").checked
        },
        answer: {
          enabled: true,
          evidence_top_k: $("evidenceTopK").value ? Number($("evidenceTopK").value) : 8,
          include_debug: true
        }
      };
    }

    function qaRetrievalOverrides(overrides = {}) {
      return {
        work_order_mode: $("qaWorkOrderHybrid").checked ? "hybrid" : "bm25",
        manual_mode: $("qaManualHybrid").checked ? "hybrid" : "bm25",
        ...overrides
      };
    }

    function updateQaConfigSummary() {
      const workOrderMode = $("qaWorkOrderHybrid").checked ? "hybrid" : "BM25";
      const manualMode = $("qaManualHybrid").checked ? "hybrid" : "BM25";
      $("qaConfigSummary").textContent = [
        `工单=${workOrderMode}`,
        `手册=${manualMode}`,
        `工单候选=${$("workOrderCandidateTopK").value || 50}`,
        `手册候选=${$("manualCandidateTopK").value || 30}`,
        `答案证据=${$("evidenceTopK").value || 4}`
      ].join(" · ");
    }

    function commonPayload(options = {}) {
      return {
        config_overrides: configOverrides(options)
      };
    }

    function taskPayload(extra = {}) {
      return {
        ...extra
      };
    }

    function sharedTaskPayload(extra = {}) {
      return {
        use_shared_database: true,
        ...extra
      };
    }

    function parseCsv(value) {
      return String(value || "").split(",").map(item => item.trim()).filter(Boolean);
    }

    function routeBatchEvalShareIdFromPath(pathname) {
      const raw = String(pathname || "").replace(/^\/+|\/+$/g, "");
      if (!raw || raw.includes("/") || raw.toLowerCase().startsWith("api")) return "";
      return /^[A-Za-z0-9_-]{6,64}$/.test(raw) ? raw : "";
    }

    function parseCsvFileText(text) {
      const rows = [];
      let row = [];
      let cell = "";
      let quoted = false;
      const source = String(text || "").replace(/^\uFEFF/, "");
      for (let index = 0; index < source.length; index += 1) {
        const ch = source[index];
        const next = source[index + 1];
        if (ch === '"') {
          if (quoted && next === '"') {
            cell += '"';
            index += 1;
          } else {
            quoted = !quoted;
          }
        } else if (ch === "," && !quoted) {
          row.push(cell);
          cell = "";
        } else if ((ch === "\n" || ch === "\r") && !quoted) {
          if (ch === "\r" && next === "\n") index += 1;
          row.push(cell);
          rows.push(row);
          row = [];
          cell = "";
        } else {
          cell += ch;
        }
      }
      if (cell || row.length) {
        row.push(cell);
        rows.push(row);
      }
      const nonEmptyRows = rows.filter(item => item.some(value => String(value || "").trim()));
      if (!nonEmptyRows.length) throw new Error("CSV 内容为空");
      const headers = nonEmptyRows[0].map((value, index) => String(value || `列${index + 1}`).trim() || `列${index + 1}`);
      const records = nonEmptyRows.slice(1).map(rowValues => headers.map((_header, index) => rowValues[index] ?? ""));
      return {headers, rows: records};
    }

    async function loadBatchEvalCsv() {
      const file = $("batchEvalCsv").files && $("batchEvalCsv").files[0];
      if (!file) return;
      try {
        const parsed = await parseBatchEvalFile(file);
        appState.batchEval.headers = parsed.headers;
        appState.batchEval.rows = parsed.rows;
        appState.batchEval.results = [];
        appState.batchEval.fileName = file.name;
        appState.batchEval.rowCount = parsed.rows.length;
        appState.batchEval.taskId = null;
        appState.batchEval.shareId = null;
        appState.batchEval.useSharedDatabase = false;
        renderBatchEvalColumns();
        renderBatchEvalResults();
        $("batchEvalFileMeta").textContent = `${file.name} · ${parsed.rows.length} 行数据 · ${parsed.headers.length} 列`;
        $("batchEvalHomeStatus").textContent = `${parsed.format || "表格"} 已载入，请确认列映射后开始评测。`;
        $("batchEvalStatus").textContent = "准备发起新的批量评测。";
        $("exportBatchEvalBtn").disabled = true;
      } catch (error) {
        setStatus(String(error), "error");
        $("batchEvalHomeStatus").textContent = String(error);
      }
    }

    async function parseBatchEvalFile(file) {
      const name = String(file && file.name ? file.name : "").toLowerCase();
      const type = String(file && file.type ? file.type : "").toLowerCase();
      if (name.endsWith(".xlsx") || type.includes("spreadsheetml.sheet")) {
        const response = await postJson("/api/parse-table", {
          filename: file.name,
          data_base64: await fileToBase64(file)
        });
        return {
          headers: response.headers || [],
          rows: response.rows || [],
          format: response.format || "XLSX"
        };
      }
      return {...parseCsvFileText(await file.text()), format: "CSV"};
    }

    function fileToBase64(file) {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
          const dataUrl = String(reader.result || "");
          const commaIndex = dataUrl.indexOf(",");
          resolve(commaIndex >= 0 ? dataUrl.slice(commaIndex + 1) : dataUrl);
        };
        reader.onerror = () => reject(reader.error || new Error("读取文件失败"));
        reader.readAsDataURL(file);
      });
    }

    function renderBatchEvalColumns() {
      const headers = appState.batchEval.headers || [];
      setColumnOptions("batchEvalQuestionColumn", headers, guessColumnIndex(headers, ["问题", "query", "报修", "故障"]), false);
      setColumnOptions("batchEvalWorkOrderIdColumn", headers, guessColumnIndex(headers, ["预期工单", "目标工单", "工单ID", "工单id", "工单编号", "work_order_id", "filename", "文件名"]), true);
      setColumnOptions("batchEvalPartNameColumn", headers, guessColumnIndex(headers, ["新件备件名称", "备件名称", "名称", "part_name"]), true);
      setColumnOptions("batchEvalPartCodeColumn", headers, guessColumnIndex(headers, ["新件物料编码", "物料编码", "备件编码", "part_code", "编码"]), true);
      setColumnOptions("batchEvalPartQuantityColumn", headers, guessColumnIndex(headers, ["新件数量", "备件数量", "数量", "quantity"]), true);
    }

    function setColumnOptions(id, headers, selectedIndex, allowBlank) {
      const select = $(id);
      const blankOption = allowBlank ? '<option value="">不使用</option>' : "";
      select.innerHTML = blankOption + headers.map((header, index) => (
        `<option value="${index}">${escapeHtml(index + 1)} · ${escapeHtml(header)}</option>`
      )).join("");
      if (selectedIndex >= 0) select.value = String(selectedIndex);
    }

    function guessColumnIndex(headers, keywords) {
      const normalizedHeaders = headers.map(header => normalizeEvalText(header));
      for (const keyword of keywords) {
        const normalizedKeyword = normalizeEvalText(keyword);
        const found = normalizedHeaders.findIndex(header => header.includes(normalizedKeyword));
        if (found >= 0) return found;
      }
      return -1;
    }

    async function refreshBatchEvalRuns(options = {}) {
      try {
        const data = await postJson("/api/batch-evals", taskPayload({limit: 120}));
        appState.batchEvalRuns = data.batch_evals || [];
        renderBatchEvalRuns();
        return data;
      } catch (error) {
        if (!options.quiet) setStatus(String(error), "error");
        appState.batchEvalRuns = [];
        renderBatchEvalRuns();
        return {batch_evals: []};
      }
    }

    function renderBatchEvalRuns() {
      const container = $("batchEvalRunList");
      if (!container) return;
      if (!appState.batchEvalRuns.length) {
        container.innerHTML = '<div class="empty">暂无批量评测记录</div>';
        return;
      }
      container.innerHTML = appState.batchEvalRuns.map(task => {
        const counts = task.counts || {};
        const metricLabel = batchEvalMetricLabel(counts.selected_metric || "part_recall");
        const active = Number(task.id) === Number(appState.activeBatchEvalTaskId);
        const shareId = batchEvalShareId(task);
        const sharePath = shareId ? `/${shareId}` : "生成中";
        const runModeLabel = batchEvalRunModeLabel((task.result && (task.result.run_mode || task.result.settings)) || "search");
        return `
          <button class="batch-eval-run-card ${escapeHtml(task.status || "")} ${active ? "active" : ""}" data-task-id="${escapeHtml(task.id)}">
            <div class="task-line">
              <span class="task-title">#${escapeHtml(task.id)} ${escapeHtml(batchEvalTaskTitle(task))}</span>
              <span class="pill ${task.status === "completed" ? "ok" : task.status === "failed" || task.status === "completed_with_errors" || task.status === "stopped" ? "warn" : ""}">${escapeHtml(task.status || "")}</span>
            </div>
            <div class="task-subtitle">${escapeHtml(runModeLabel)} · ${escapeHtml(metricLabel)} · 正确 ${escapeHtml(counts.pass ?? 0)} · 失败 ${escapeHtml(counts.fail ?? 0)} · 错误 ${escapeHtml(counts.error ?? 0)} · 总数 ${escapeHtml(counts.total ?? "-")}</div>
            <div class="row-meta">分享路径：${escapeHtml(sharePath)}</div>
            <div class="row-meta">${escapeHtml(compactTime(task.updated_at || task.created_at))}</div>
          </button>
        `;
      }).join("");
      for (const button of container.querySelectorAll("[data-task-id]")) {
        button.addEventListener("click", () => loadBatchEvalRun(Number(button.dataset.taskId)));
      }
    }

    function batchEvalTaskTitle(task) {
      const result = task && task.result ? task.result : {};
      return result.file_name || task.file_name || task.query || "批量评测";
    }

    function batchEvalShareId(task) {
      const result = task && task.result ? task.result : {};
      return String((result && result.share_id) || (task && task.share_id) || "").trim();
    }

    function updateBatchEvalBrowserRoute(task, options = {}) {
      const shareId = batchEvalShareId(task);
      if (!shareId || options.push === false) return;
      const targetPath = `/${shareId}`;
      if (window.location.pathname !== targetPath) {
        const method = options.replace ? "replaceState" : "pushState";
        window.history[method]({batchEvalShareId: shareId}, "", targetPath);
      }
    }

    function resetWorkbenchBrowserRoute(options = {}) {
      if (window.location.pathname === "/") return;
      const method = options.replace ? "replaceState" : "pushState";
      window.history[method]({}, "", "/");
    }

    async function loadBatchEvalRun(taskId) {
      try {
        appState.batchEval.useSharedDatabase = false;
        const data = await postJson("/api/task", taskPayload({task_id: taskId}));
        applyBatchEvalTask(data.task);
        switchView("batch");
        activateBatchEvalOverview();
        updateBatchEvalBrowserRoute(data.task);
        setStatus(`已载入批量评测 #${taskId}`, "success");
      } catch (error) {
        setStatus(String(error), "error");
      }
    }

    async function loadBatchEvalShareRoute(shareId = routeBatchEvalShareId) {
      if (!shareId) return false;
      try {
        const data = await postJson("/api/batch-eval-share", {share_id: shareId});
        applyBatchEvalTask(data.task);
        appState.batchEval.useSharedDatabase = true;
        switchView("batch");
        activateBatchEvalOverview();
        setStatus(`已通过分享链接载入批量评测 /${shareId}`, "success");
        return true;
      } catch (error) {
        setStatus(`分享链接加载失败：${error}`, "error");
        return false;
      }
    }

    function applyBatchEvalTask(task) {
      if (!task) return;
      const result = task.result || {};
      appState.activeBatchEvalTaskId = task.id;
      appState.activeBatchEvalTask = task;
      appState.activeBatchEvalRowNumber = null;
      appState.activeBatchEvalRetryTaskId = null;
      appState.activeBatchEvalReplay = null;
      appState.batchEval.taskId = task.id;
      appState.batchEval.fileName = result.file_name || task.query || "";
      appState.batchEval.headers = result.headers || [];
      appState.batchEval.rows = [];
      appState.batchEval.rowCount = Number(result.row_count || result.total || (result.rows || []).length || 0);
      appState.batchEval.results = Array.isArray(result.rows) ? result.rows : [];
      appState.batchEval.shareId = result.share_id || task.share_id || null;
      appState.batchEval.settings = result.settings || (result.run_mode ? {runMode: result.run_mode} : null);
      if (appState.batchEval.settings && !appState.batchEval.settings.runMode) {
        appState.batchEval.settings = {...appState.batchEval.settings, runMode: result.run_mode || "search"};
      }
      appState.batchEval.partColumns = result.part_columns || null;
      appState.batchEval.workOrderColumn = result.work_order_column ?? null;
      appState.batchEval.selectedMetric = result.selected_metric || appState.batchEval.selectedMetric || "part_recall";
      appState.batchEval.questionIndex = result.question_column ?? null;
      syncBatchRetryDefaults(appState.batchEval.settings);
      renderBatchEvalPage();
    }

    function syncBatchRetryDefaults(settings) {
      const defaults = batchRetryDefaults(settings);
      appState.batchRetry = {
        ...appState.batchRetry,
        ...defaults,
        running: false
      };
    }

    function batchRetryDefaults(settings = appState.batchEval.settings) {
      const retrieval = (settings && settings.retrieval) || {};
      return {
        topK: Number(settings && settings.topK ? settings.topK : 1),
        workOrderCandidateTopK: Number(retrieval.work_order_candidate_top_k ?? 50),
        workOrderMinRelativeScore: Number(retrieval.work_order_min_relative_score ?? 0.45),
        workOrderMaxHits: Number(retrieval.work_order_max_hits ?? 10),
        manualCandidateTopK: Number(retrieval.manual_candidate_top_k ?? 30),
        manualMinRelativeScore: Number(retrieval.manual_min_relative_score ?? 0.55),
        manualMaxHits: Number(retrieval.manual_max_hits ?? 5),
        workOrderHybrid: retrieval.work_order_mode === "hybrid" || Boolean(settings && settings.workOrderHybrid === true),
        manualHybrid: retrieval.manual_mode === "hybrid" || Boolean(settings && settings.manualHybrid === true)
      };
    }

    function renderBatchEvalPage() {
      const detailOpen = appState.activeView === "batch" && Boolean(appState.activeBatchEvalTaskId);
      const overviewOpen = detailOpen && appState.activeBatchEvalRowNumber === null;
      $("batchHomePanel").classList.toggle("hidden", detailOpen);
      $("batchDetailPanel").classList.toggle("hidden", !overviewOpen);
      renderShellMode();
      updateSidebarChrome();
      renderBatchRetryPanel();
      renderBatchEvalRuns();
      renderBatchEvalResults();
      renderBatchEvalQuestionTabs();
      const task = appState.activeBatchEvalTask || {};
      $("batchEvalDetailTitle").textContent = detailOpen ? `批量评测 #${task.id || appState.activeBatchEvalTaskId} · ${batchEvalTaskTitle(task)}` : "批量评测详情";
      $("exportBatchEvalBtn").disabled = !appState.batchEval.results.length;
    }

    function openBatchEvalHome(options = {}) {
      appState.activeBatchEvalTaskId = null;
      appState.activeBatchEvalTask = null;
      appState.activeBatchEvalRowNumber = null;
      appState.activeBatchEvalRetryTaskId = null;
      appState.activeBatchEvalReplay = null;
      appState.batchEval.useSharedDatabase = false;
      switchView("batch");
      if (options.updateRoute !== false) resetWorkbenchBrowserRoute();
      refreshBatchEvalRuns({quiet: true});
    }

    function renderBatchEvalQuestionTabs() {
      const container = $("questionTabs");
      if (!container) return;
      if (appState.activeView !== "batch") return;
      renderBatchMetricSelector();
      const rows = orderedBatchEvalResults();
      const metric = selectedBatchEvalMetric();
      const metricLabel = batchEvalMetricLabel(metric);
      if (!appState.activeBatchEvalTaskId) {
        container.innerHTML = '<div class="empty">选择或发起一个批量评测后，这里会显示该批次的问题。</div>';
        return;
      }
      if (!rows.length) {
        container.innerHTML = '<div class="empty">本次评测暂无问题结果</div>';
        return;
      }
      const overviewActive = appState.activeBatchEvalRowNumber === null;
      const counts = batchEvalCounts(rows, metric);
      const overview = `
        <button class="question-tab overview ${overviewActive ? "active" : ""}" data-batch-overview="1">
          <div class="question-tab-title">整体进展</div>
          <div class="question-tab-meta">${escapeHtml(metricLabel)} · ${escapeHtml(counts.done)} / ${escapeHtml(counts.total)} · 正确 ${escapeHtml(counts.pass)} · 失败 ${escapeHtml(counts.fail)}</div>
        </button>
      `;
      container.innerHTML = overview + rows.map(item => {
        const status = batchEvalMetricStatus(item, metric);
        return `
        <button class="question-tab ${escapeHtml(status)} ${Number(item.rowNumber) === Number(appState.activeBatchEvalRowNumber) ? "active" : ""}" data-row-number="${escapeHtml(item.rowNumber)}">
          <div class="question-tab-title">#${escapeHtml(item.rowNumber)} ${escapeHtml(batchEvalStatusText(status))} · ${escapeHtml(questionTitle(item.question || ""))}</div>
          <div class="question-tab-meta">${escapeHtml(item.taskId ? `${batchEvalRunModeFromItem(item) === "answer" ? "回答" : "检索"} #${item.taskId}` : batchEvalReason(item, metric) || "无检索任务")}</div>
        </button>
      `;
      }).join("");
      const overviewButton = container.querySelector("[data-batch-overview]");
      if (overviewButton) overviewButton.addEventListener("click", activateBatchEvalOverview);
      for (const button of container.querySelectorAll("[data-row-number]")) {
        button.addEventListener("click", () => activateBatchEvalRow(Number(button.dataset.rowNumber)));
      }
    }

    function activateBatchEvalOverview() {
      appState.activeBatchEvalRowNumber = null;
      appState.activeBatchEvalRetryTaskId = null;
      appState.activeBatchEvalReplay = null;
      renderBatchEvalPage();
      setStatus("已切换到批量评测整体进展", "success");
    }

    async function activateBatchEvalRow(rowNumber) {
      const item = orderedBatchEvalResults().find(row => Number(row.rowNumber) === Number(rowNumber));
      if (!item) return;
      appState.activeBatchEvalRowNumber = item.rowNumber;
      appState.activeBatchEvalRetryTaskId = null;
      appState.activeBatchEvalReplay = null;
      syncBatchRetryDefaults(appState.batchEval.settings);
      $("query").value = item.question || "";
      resetStages("batch", "retrieval");
      renderBatchEvalPage();
      const metric = selectedBatchEvalMetric();
      const metricStatus = batchEvalMetricStatus(item, metric);
      const metricStatusKind = metricStatus === "pass" ? "success" : metricStatus === "fail" || metricStatus === "error" ? "error" : "";
      if (!item.taskId) {
        renderRetrievalBoard({channels: {}, mode: "", top_k: ""});
        renderParts([]);
        renderSelectedEvidence([]);
        $("answer").textContent = item.message || "该行没有可重现的检索任务。";
        setStage("retrieval", item.status === "skipped" ? "skipped" : "error", item, batchEvalReason(item, metric) || "无原始召回任务");
        $("batchEvalStatus").textContent = `第 ${item.rowNumber} 行 · ${batchEvalMetricLabel(metric)} ${batchEvalStatusText(metricStatus)} · 无原始召回任务，可按当前参数重试检索。`;
        setStatus(`已打开批量评测第 ${item.rowNumber} 行`, metricStatusKind);
        return;
      }
      try {
        const payload = appState.batchEval.useSharedDatabase
          ? sharedTaskPayload({task_id: item.taskId})
          : taskPayload({task_id: item.taskId});
        const data = await postJson("/api/task", payload);
        const task = data.task;
        if (!task) throw new Error(`task not found: ${item.taskId}`);
        appState.currentTaskId = task.id;
        renderBatchOriginalRetrieval(item, task.result && task.result.result ? task.result.result : task.result || {});
        $("batchEvalStatus").textContent = `第 ${item.rowNumber} 行 · ${batchEvalMetricLabel(metric)} ${batchEvalStatusText(metricStatus)} · 当前展示批量评测时的原始${batchEvalRunModeFromItem(item) === "answer" ? "回答" : "召回"}；点击重试按钮才会重新检索。`;
        setStatus(`已打开批量评测第 ${item.rowNumber} 行原始${batchEvalRunModeFromItem(item) === "answer" ? "回答" : "召回"}`, metricStatusKind);
      } catch (error) {
        renderRetrievalBoard({channels: {}, mode: "", top_k: ""});
        renderParts([]);
        renderSelectedEvidence([]);
        setStage("retrieval", "error", {error: String(error), row: item}, "重现检索失败");
        setStatus(String(error), "error");
      }
    }

    function renderBatchOriginalRetrieval(item, retrieval) {
      const result = retrieval || {};
      const retrieved = result.retrieval || result;
      appState.lastResult = result || {};
      resetStages("batch", "retrieval");
      setStage("retrieval", "done", retrieved, formatRetrievalSummary(retrieved), {select: false});
      if (result.answer_harness && result.answer_harness.work_order_filter) {
        setStage("work_order_filter", result.answer_harness.work_order_filter.status || "done", result.answer_harness.work_order_filter, workOrderFilterSummary(result.answer_harness.work_order_filter), {select: false});
      }
      if (result.answer_harness && result.answer_harness.manual_filter) {
        setStage("manual_filter", result.answer_harness.manual_filter.status || "done", result.answer_harness.manual_filter, manualFilterSummary(result.answer_harness.manual_filter), {select: false});
      }
      if (result.answer_harness && result.answer_harness.facts) {
        setStage("fact_extraction", result.answer_harness.facts.status || "done", result.answer_harness.facts, factSummary(result.answer_harness.facts), {select: false});
      }
      if (result.answer) {
        setStage("answer", result.answer.status || "done", result.answer, answerSummary(result.answer), {select: false});
        renderAnswer(result.answer);
      } else {
        $("answer").classList.remove("markdown-body");
        $("answer").textContent = `当前展示第 ${item.rowNumber} 行原始召回。`;
      }
      selectStage("retrieval");
      renderStages();
      renderStageInspector();
      renderRetrievalBoard(retrieved);
      renderParts(answerPartsForDisplay(result, retrieved));
      renderSelectedEvidence([]);
    }

    function batchEvalStatusText(status) {
      return {
        pass: "正确",
        warn: "工单未命中",
        fail: "失败",
        error: "错误",
        skipped: "跳过"
      }[status] || status || "未知";
    }

    async function runBatchEval() {
      if (appState.batchEval.running) return;
      const rows = appState.batchEval.rows || [];
      const questionIndex = Number($("batchEvalQuestionColumn").value);
      if (!rows.length) {
        setStatus("请先载入 CSV 文件", "error");
        return;
      }
      if (!Number.isInteger(questionIndex) || questionIndex < 0) {
        setStatus("请选择问题列", "error");
        return;
      }
      const settings = batchEvalSettings();
      if (!settings.ok) {
        setStatus(settings.error, "error");
        return;
      }
      const partColumns = {
        name: optionalColumnIndex("batchEvalPartNameColumn"),
        code: optionalColumnIndex("batchEvalPartCodeColumn"),
        quantity: optionalColumnIndex("batchEvalPartQuantityColumn")
      };
      const workOrderColumn = optionalColumnIndex("batchEvalWorkOrderIdColumn");
      appState.batchEval.settings = settings;
      if (!batchEvalMetricOptionsForSettings(settings).some(([value]) => value === appState.batchEval.selectedMetric)) {
        appState.batchEval.selectedMetric = "part_recall";
      }
      appState.batchEval.partColumns = partColumns;
      appState.batchEval.workOrderColumn = workOrderColumn;
      appState.batchEval.questionIndex = questionIndex;
      appState.batchEval.rowCount = rows.length;
      syncBatchRetryDefaults(settings);
      appState.batchEval.persistPromise = Promise.resolve();
      appState.batchEval.persistError = null;
      appState.batchEval.running = true;
      appState.batchEval.stopRequested = false;
      appState.batchEval.results = [];
      $("runBatchEvalBtn").disabled = true;
      $("stopBatchEvalBtn").disabled = false;
      $("exportBatchEvalBtn").disabled = true;
      try {
        const created = await createBatchEvalTask(settings, partColumns, questionIndex, workOrderColumn);
        appState.batchEval.taskId = created.task_id;
        appState.batchEval.shareId = (created.result && created.result.share_id) || created.share_id || null;
        appState.batchEval.useSharedDatabase = false;
        appState.activeBatchEvalTaskId = created.task_id;
        appState.activeBatchEvalRowNumber = null;
        appState.activeBatchEvalRetryTaskId = null;
        appState.activeBatchEvalReplay = null;
        appState.activeBatchEvalTask = {
          id: created.task_id,
          task_type: "batch_eval",
          status: "running",
          query: created.query,
          result: batchEvalResultPayload("running")
        };
        switchView("batch");
        renderBatchEvalPage();
        updateBatchEvalBrowserRoute(appState.activeBatchEvalTask);
      } catch (error) {
        appState.batchEval.running = false;
        $("runBatchEvalBtn").disabled = false;
        $("stopBatchEvalBtn").disabled = true;
        setStatus(`批量评测创建失败：${error}`, "error");
        return;
      }
      let nextIndex = 0;
      let completed = 0;
      const workerCount = Math.min(settings.concurrency, Math.max(rows.length, 1));
      const nextWorkIndex = () => {
        if (appState.batchEval.stopRequested || nextIndex >= rows.length) return null;
        const current = nextIndex;
        nextIndex += 1;
        return current;
      };
      const runWorker = async () => {
        while (true) {
          const index = nextWorkIndex();
          if (index === null) return;
          const item = await runBatchEvalRow(index, rows[index], questionIndex, partColumns, workOrderColumn, settings);
          appState.batchEval.results.push(item);
          completed += 1;
          $("batchEvalStatus").textContent = `评测中：${completed} / ${rows.length} · ${batchEvalRunModeLabel(settings.runMode)} · 并发 ${workerCount}`;
          renderBatchEvalResults(completed, rows.length);
          renderBatchEvalQuestionTabs();
          scheduleBatchEvalPersist("running");
        }
      };
      try {
        await Promise.all(Array.from({length: workerCount}, () => runWorker()));
        const finalStatus = appState.batchEval.stopRequested ? "stopped" : appState.batchEval.results.some(item => item.status === "error") ? "completed_with_errors" : "completed";
        $("batchEvalStatus").textContent = finalStatus === "stopped" ? "评测已停止" : "评测完成";
        await scheduleBatchEvalPersist(finalStatus);
        await refreshBatchEvalRuns({quiet: true});
      } finally {
        appState.batchEval.running = false;
        appState.batchEval.stopRequested = false;
        $("runBatchEvalBtn").disabled = false;
        $("stopBatchEvalBtn").disabled = true;
        $("exportBatchEvalBtn").disabled = !appState.batchEval.results.length;
        renderBatchEvalPage();
      }
    }

    async function createBatchEvalTask(settings, partColumns, questionIndex, workOrderColumn) {
      const payload = {
        ...commonPayload({retrieval: settings.retrieval, autoEmbeddingForHybrid: true}),
        batch_eval: {
          file_name: appState.batchEval.fileName || "未命名表格",
          row_count: appState.batchEval.rows.length,
          headers: appState.batchEval.headers,
          settings,
          selected_metric: selectedBatchEvalMetric(),
          question_column: questionIndex,
          work_order_column: workOrderColumn,
          part_columns: partColumns
        }
      };
      return postJson("/api/create-batch-eval", payload);
    }

    function scheduleBatchEvalPersist(status) {
      if (!appState.batchEval.taskId) return Promise.resolve();
      const taskId = appState.batchEval.taskId;
      appState.batchEval.persistPromise = (appState.batchEval.persistPromise || Promise.resolve())
        .catch(() => {})
        .then(() => postJson("/api/update-batch-eval", {
          ...taskPayload({task_id: taskId}),
          status,
          result: batchEvalResultPayload(status)
        }))
        .catch(error => {
          appState.batchEval.persistError = String(error);
          setStatus(`批量评测进度保存失败：${error}`, "error");
        });
      return appState.batchEval.persistPromise;
    }

    function batchEvalResultPayload(status) {
      const rows = orderedBatchEvalResults();
      const counts = batchEvalCounts(rows);
      return {
        task_id: appState.batchEval.taskId,
        status,
        file_name: appState.batchEval.fileName || "未命名表格",
        share_id: appState.batchEval.shareId || null,
        run_mode: batchEvalRunModeFromSettings(appState.batchEval.settings),
        headers: appState.batchEval.headers || [],
        row_count: appState.batchEval.rowCount || appState.batchEval.rows.length || rows.length,
        settings: appState.batchEval.settings || null,
        selected_metric: selectedBatchEvalMetric(),
        question_column: appState.batchEval.questionIndex,
        work_order_column: appState.batchEval.workOrderColumn,
        part_columns: appState.batchEval.partColumns || null,
        counts,
        rows,
        updated_at: new Date().toISOString()
      };
    }

    function batchEvalCounts(results = appState.batchEval.results, metric = selectedBatchEvalMetric()) {
      const rows = results || [];
      const workOrderRequired = rows.filter(item => item.expectedWorkOrderId).length;
      const statuses = rows.map(item => batchEvalMetricStatus(item, metric));
      return {
        total: appState.batchEval.rowCount || appState.batchEval.rows.length || rows.length,
        done: rows.length,
        pass: statuses.filter(status => status === "pass").length,
        warn: 0,
        fail: statuses.filter(status => status === "fail").length,
        error: statuses.filter(status => status === "error").length,
        skipped: statuses.filter(status => status === "skipped").length,
        work_order_required: workOrderRequired,
        work_order_pass: rows.filter(item => item.match && item.match.work_order && item.match.work_order.required && item.match.work_order.correct).length,
        part_recall_pass: rows.filter(item => batchEvalMetricCorrect(item, "part_recall")).length,
        answer_part_recall_pass: rows.filter(item => batchEvalMetricCorrect(item, "answer_part_recall")).length,
        selected_metric: metric,
        metric_counts: batchEvalAllMetricCounts(rows)
      };
    }

    function selectedBatchEvalMetric(settings = appState.activeBatchEvalTaskId ? appState.batchEval.settings : $("batchEvalRunMode").value) {
      const metric = appState.batchEval && appState.batchEval.selectedMetric ? appState.batchEval.selectedMetric : "part_recall";
      return batchEvalMetricOptionsForSettings(settings).some(([value]) => value === metric) ? metric : "part_recall";
    }

    function batchEvalMetricLabel(metric = selectedBatchEvalMetric()) {
      const found = [...batchEvalBaseMetricOptions, batchEvalAnswerMetricOption].find(([value]) => value === metric);
      return found ? found[1] : "备件召回率";
    }

    function batchEvalMetricStatus(item, metric = selectedBatchEvalMetric()) {
      if (!item) return "fail";
      if (item.status === "error" || item.status === "skipped") return item.status;
      if (metric === "answer_part_recall" && !batchEvalAnswerMetricAvailable(item)) return "skipped";
      return batchEvalMetricCorrect(item, metric) ? "pass" : "fail";
    }

    function batchEvalMetricCorrect(item, metric = selectedBatchEvalMetric()) {
      const match = item && item.match ? item.match : {};
      if (metric === "work_order_recall") return Boolean(match.work_order ? match.work_order.correct : false);
      if (metric === "answer_part_recall") return batchEvalAnswerMetricAvailable(item) && Boolean(match.answer_part_recall_correct);
      return Boolean(match.part_recall_correct ?? (Array.isArray(match.missing) && match.missing.length === 0));
    }

    function batchEvalAnswerMetricAvailable(item) {
      const match = item && item.match ? item.match : {};
      return batchEvalRunModeFromItem(item) === "answer" && Boolean(match.answer_part_recall);
    }

    function batchEvalAllMetricCounts(rows) {
      const payload = {};
      for (const [metric] of batchEvalMetricOptionsForSettings()) {
        const statuses = (rows || []).map(item => batchEvalMetricStatus(item, metric));
        payload[metric] = {
          pass: statuses.filter(status => status === "pass").length,
          fail: statuses.filter(status => status === "fail").length,
          error: statuses.filter(status => status === "error").length,
          skipped: statuses.filter(status => status === "skipped").length
        };
      }
      return payload;
    }

    function batchEvalSettings() {
      const runMode = $("batchEvalRunMode").value === "answer" ? "answer" : "search";
      const workOrderHybrid = $("batchEvalWorkOrderHybrid").checked;
      const manualHybrid = $("batchEvalManualHybrid").checked;
      const topK = numberInputValue("batchEvalTopK", 1);
      const workOrderCandidateTopK = numberInputValue("batchEvalWorkOrderCandidateTopK", 50);
      const workOrderMinRelativeScore = numberInputValue("batchEvalWorkOrderMinRelativeScore", 0.45);
      const workOrderMaxHits = numberInputValue("batchEvalWorkOrderMaxHits", 10);
      const manualCandidateTopK = numberInputValue("batchEvalManualCandidateTopK", 30);
      const manualMinRelativeScore = numberInputValue("batchEvalManualMinRelativeScore", 0.55);
      const manualMaxHits = numberInputValue("batchEvalManualMaxHits", 5);
      const concurrency = numberInputValue("batchEvalConcurrency", 4);
      if (!Number.isFinite(topK) || topK < 1) return {ok: false, error: "故障码 Top K 必须大于等于 1"};
      if (!Number.isFinite(workOrderCandidateTopK) || workOrderCandidateTopK < 1) return {ok: false, error: "工单候选上限必须大于等于 1"};
      if (!Number.isFinite(workOrderMinRelativeScore) || workOrderMinRelativeScore < 0 || workOrderMinRelativeScore > 1) {
        return {ok: false, error: "工单相对阈值必须在 0 到 1 之间"};
      }
      if (!Number.isFinite(workOrderMaxHits) || workOrderMaxHits < 0) return {ok: false, error: "工单最大返回必须大于等于 0"};
      if (!Number.isFinite(manualCandidateTopK) || manualCandidateTopK < 1) return {ok: false, error: "手册候选上限必须大于等于 1"};
      if (!Number.isFinite(manualMinRelativeScore) || manualMinRelativeScore < 0 || manualMinRelativeScore > 1) {
        return {ok: false, error: "手册相对阈值必须在 0 到 1 之间"};
      }
      if (!Number.isFinite(manualMaxHits) || manualMaxHits < 0) return {ok: false, error: "手册最大返回必须大于等于 0"};
      if (!Number.isFinite(concurrency) || concurrency < 1) return {ok: false, error: "并发数必须大于等于 1"};
      return {
        ok: true,
        runMode,
        workOrderHybrid,
        manualHybrid,
        topK: Math.floor(topK),
        concurrency: Math.min(Math.floor(concurrency), 16),
        retrieval: {
          work_order_mode: workOrderHybrid ? "hybrid" : "bm25",
          manual_mode: manualHybrid ? "hybrid" : "bm25",
          work_order_candidate_top_k: Math.floor(workOrderCandidateTopK),
          work_order_min_relative_score: workOrderMinRelativeScore,
          work_order_max_hits: Math.floor(workOrderMaxHits),
          manual_candidate_top_k: Math.floor(manualCandidateTopK),
          manual_min_relative_score: manualMinRelativeScore,
          manual_max_hits: Math.floor(manualMaxHits)
        }
      };
    }

    function numberInputValue(id, fallback) {
      const value = $(id).value;
      return value === "" ? fallback : Number(value);
    }

    async function runBatchEvalRow(index, row, questionIndex, partColumns, workOrderColumn, settings) {
      const rowNumber = index + 2;
      const question = String(row[questionIndex] || "").trim();
      const expectedParts = buildExpectedParts(row, partColumns);
      const expectedWorkOrderId = expectedWorkOrderIdFromRow(row, workOrderColumn);
      const runMode = settings.runMode === "answer" ? "answer" : "search";
      if (!question) {
        return {
          rowNumber,
          question: "",
          status: "skipped",
          runMode,
          expectedParts,
          expectedWorkOrderId,
          retrievedParts: [],
          retrievedWorkOrderIds: [],
          retrievedWorkOrders: [],
          answerText: "",
          message: "问题为空"
        };
      }
      let taskId = null;
      try {
        const payload = queryPayload(question, {topK: settings.topK, retrieval: settings.retrieval, useQaModes: false, autoEmbeddingForHybrid: true});
        const response = runMode === "answer"
          ? await runBatchAnswerTask(payload)
          : await postJson("/api/search-db", payload);
        taskId = response.task_id || null;
        const result = response.result || response;
        const retrieval = runMode === "answer" ? (result.retrieval || result) : result;
        const answer = runMode === "answer" && result.answer && typeof result.answer === "object" ? result.answer : {};
        const answerText = String(answer.text || "");
        const retrievedParts = listPartCandidates(retrieval);
        const retrievedWorkOrders = listRetrievedWorkOrders(retrieval);
        const retrievedWorkOrderIds = uniqueWorkOrderIdsFromHits(retrievedWorkOrders);
        const match = evaluateBatchEvalMatch(expectedParts, retrievedParts, expectedWorkOrderId, retrievedWorkOrderIds, answerText, runMode);
        return {
          rowNumber,
          question,
          status: batchEvalStatusFromMatch(match),
          runMode,
          taskId,
          expectedParts,
          expectedWorkOrderId,
          retrievedParts,
          retrievedWorkOrderIds,
          retrievedWorkOrders,
          answerText,
          answerStatus: answer.status || "",
          match
        };
      } catch (error) {
        return {
          rowNumber,
          question,
          status: "error",
          runMode,
          taskId,
          expectedParts,
          expectedWorkOrderId,
          retrievedParts: [],
          retrievedWorkOrderIds: [],
          retrievedWorkOrders: [],
          answerText: "",
          message: String(error)
        };
      }
    }

    async function runBatchAnswerTask(payload) {
      const started = await postJson("/api/ask-db", {...payload, async: true});
      const taskId = started.task_id;
      if (!taskId) throw new Error("回答任务未返回 task_id");
      const task = await waitForBatchAnswerTask(taskId);
      const result = task.result && typeof task.result === "object" ? task.result : {};
      return {
        task_id: task.id || taskId,
        summary: task.summary || result.summary || "",
        result: result.result || result
      };
    }

    async function waitForBatchAnswerTask(taskId) {
      while (true) {
        if (appState.batchEval.stopRequested) throw new Error("批量评测已停止，当前回答任务未等待完成");
        const data = await postJson("/api/task", taskPayload({task_id: taskId}));
        const task = data.task;
        if (!task) throw new Error(`回答任务不存在：${taskId}`);
        if (!["running", "pause_requested"].includes(task.status)) {
          if (task.status === "failed") throw new Error(task.error || task.summary || `回答任务 #${taskId} 失败`);
          return task;
        }
        await sleep(1200);
      }
    }

    function sleep(ms) {
      return new Promise(resolve => window.setTimeout(resolve, ms));
    }

    function optionalColumnIndex(id) {
      const value = $(id).value;
      if (value === "") return null;
      const index = Number(value);
      return Number.isInteger(index) && index >= 0 ? index : null;
    }

    function buildExpectedParts(row, columns) {
      const names = splitExpectedCell(columns.name === null ? "" : row[columns.name]);
      const codes = splitExpectedCell(columns.code === null ? "" : row[columns.code]);
      const quantities = splitExpectedCell(columns.quantity === null ? "" : row[columns.quantity]);
      const count = Math.max(names.length, codes.length, quantities.length);
      const expected = [];
      for (let index = 0; index < count; index += 1) {
        const item = {
          name: names[index] || "",
          code: codes[index] || "",
          quantity: quantities[index] || ""
        };
        if (item.name || item.code || item.quantity) expected.push(item);
      }
      return expected;
    }

    function expectedWorkOrderIdFromRow(row, columnIndex) {
      if (columnIndex === null || columnIndex === undefined) return "";
      return normalizeWorkOrderId(row[columnIndex]);
    }

    function splitExpectedCell(value) {
      return String(value || "").split(/[，,]/).map(item => item.trim()).filter(Boolean);
    }

    function listPartCandidates(retrieval) {
      const parts = retrieval && Array.isArray(retrieval.part_candidates) ? retrieval.part_candidates : [];
      return parts.map(part => ({
        name: String(part.part_name || part.part_number_name || ""),
        code: String(part.part_code || ""),
        quantity: String(part.quantity || ""),
        work_order_id: String(part.work_order_id || ""),
        source_path: String(part.source_path || "")
      }));
    }

    function listRetrievedWorkOrders(retrieval) {
      const channelsPayload = retrieval && retrieval.channels && typeof retrieval.channels === "object" ? retrieval.channels : {};
      const channelHits = Array.isArray(channelsPayload.work_orders) ? channelsPayload.work_orders : [];
      const topLevelHits = retrieval && Array.isArray(retrieval.work_orders) ? retrieval.work_orders : [];
      const hits = channelHits.length ? channelHits : topLevelHits;
      return hits.map((hit, index) => {
        const ids = uniqueWorkOrderIds(workOrderIdCandidatesFromHit(hit));
        return {
          rank: index + 1,
          id: ids[0] || "",
          ids,
          title: String(hit.title || ""),
          doc_id: String(hit.doc_id || ""),
          work_order_id: String(hit.work_order_id || ""),
          source_path: String(hit.source_path || ""),
          score: hit.score ?? ""
        };
      });
    }

    function workOrderIdCandidatesFromHit(hit) {
      const metadata = hit && hit.metadata && typeof hit.metadata === "object" ? hit.metadata : {};
      return [
        hit && hit.work_order_id,
        metadata.work_order_id,
        hit && hit.source_path,
        metadata.source_path,
        hit && hit.doc_id
      ].map(normalizeWorkOrderId).filter(Boolean);
    }

    function uniqueWorkOrderIdsFromHits(workOrders) {
      return uniqueWorkOrderIds((workOrders || []).flatMap(item => item.ids || item.id || []));
    }

    function uniqueWorkOrderIds(values) {
      const ids = [];
      const seen = new Set();
      for (const value of values || []) {
        const id = normalizeWorkOrderId(value);
        const key = normalizeEvalText(id);
        if (!id || seen.has(key)) continue;
        seen.add(key);
        ids.push(id);
      }
      return ids;
    }

    function normalizeWorkOrderId(value) {
      const raw = String(value || "").trim();
      if (!raw) return "";
      const withoutQuery = raw.split(/[?#]/)[0];
      const leaf = (withoutQuery.replace(/\\/g, "/").split("/").filter(Boolean).pop() || withoutQuery).trim();
      return leaf.replace(/\.[^.]+$/, "").replace(/^wo:/i, "").trim();
    }

    function evaluateBatchEvalMatch(expectedParts, retrievedParts, expectedWorkOrderId, retrievedWorkOrderIds, answerText = "", runMode = "search") {
      const recallMatch = evaluatePartSetMatch(expectedParts, retrievedParts, expectedPartMatches);
      const workOrderMatch = evaluateWorkOrderRecall(expectedWorkOrderId, retrievedWorkOrderIds);
      const answerPartRecall = evaluateAnswerPartRecall(expectedParts, answerText, runMode);
      const partRecallCorrect = recallMatch.missing.length === 0;
      const correct = partRecallCorrect && workOrderMatch.correct;
      return {
        ...recallMatch,
        correct,
        part_correct: partRecallCorrect,
        part_recall_correct: partRecallCorrect,
        answer_part_recall_correct: answerPartRecall.correct,
        part_recall: {
          ...recallMatch,
          correct: partRecallCorrect
        },
        answer_part_recall: answerPartRecall,
        work_order: workOrderMatch,
        metric_results: {
          part_recall: partRecallCorrect,
          work_order_recall: workOrderMatch.correct,
          answer_part_recall: answerPartRecall.correct
        },
        yellow_error: partRecallCorrect && workOrderMatch.required && !workOrderMatch.correct
      };
    }

    function evaluateAnswerPartRecall(expectedParts, answerText, runMode = "search") {
      const expected = Array.isArray(expectedParts) ? expectedParts : [];
      const available = runMode === "answer";
      if (!available) {
        return {available: false, required: expected.length > 0, correct: true, matched: [], missing: [], answer_text_preview: ""};
      }
      const answerKey = normalizeEvalText(answerText);
      const matched = [];
      const missing = [];
      for (const part of expected) {
        if (answerContainsExpectedPart(answerKey, part)) {
          matched.push(part);
        } else {
          missing.push(part);
        }
      }
      return {
        available: true,
        required: expected.length > 0,
        correct: missing.length === 0,
        matched,
        missing,
        answer_text_preview: String(answerText || "").slice(0, 500)
      };
    }

    function answerContainsExpectedPart(answerKey, expectedPart) {
      const name = normalizeEvalText(expectedPart && expectedPart.name);
      const code = normalizeEvalText(expectedPart && expectedPart.code);
      if (code && answerKey.includes(code)) return true;
      if (name && answerKey.includes(name)) return true;
      return !name && !code;
    }

    function evaluateWorkOrderRecall(expectedWorkOrderId, retrievedWorkOrderIds) {
      const expected = normalizeWorkOrderId(expectedWorkOrderId);
      const retrieved = uniqueWorkOrderIds(retrievedWorkOrderIds || []);
      if (!expected) {
        return {
          required: false,
          correct: true,
          expected: "",
          matched: "",
          retrieved
        };
      }
      const expectedKey = normalizeEvalText(expected);
      const matched = retrieved.find(id => normalizeEvalText(id) === expectedKey) || "";
      return {
        required: true,
        correct: Boolean(matched),
        expected,
        matched,
        retrieved
      };
    }

    function batchEvalStatusFromMatch(match) {
      if (match.correct) return "pass";
      if (match.yellow_error) return "warn";
      return "fail";
    }

    function evaluatePartSetMatch(expectedParts, retrievedParts, matches) {
      const matched = [];
      const missing = [];
      const usedCandidateIndexes = new Set();
      for (const expected of expectedParts) {
        const matchedIndex = retrievedParts.findIndex((candidate, index) => (
          !usedCandidateIndexes.has(index) && matches(expected, candidate)
        ));
        if (matchedIndex >= 0) {
          usedCandidateIndexes.add(matchedIndex);
          matched.push({expected, actual: retrievedParts[matchedIndex]});
        } else {
          missing.push(expected);
        }
      }
      const unexpected = retrievedParts.filter((_, index) => !usedCandidateIndexes.has(index));
      return {
        correct: missing.length === 0 && unexpected.length === 0,
        matched,
        missing,
        unexpected,
        unexpected_count: unexpected.length
      };
    }

    function expectedPartMatches(expected, actual) {
      return partNameMatches(expected.name, actual.name)
        && partCodeMatches(expected.code, actual.code)
        && partQuantityMatches(expected.quantity, actual.quantity);
    }

    function partNameMatches(expected, actual) {
      if (!String(expected || "").trim()) return true;
      const left = normalizeEvalText(expected);
      const right = normalizeEvalText(actual);
      return Boolean(left && right && (right.includes(left) || left.includes(right)));
    }

    function partCodeMatches(expected, actual) {
      if (!String(expected || "").trim()) return true;
      return normalizeEvalText(expected) === normalizeEvalText(actual);
    }

    function partQuantityMatches(expected, actual) {
      if (!String(expected || "").trim()) return true;
      const leftNumber = Number(String(expected).trim());
      const rightNumber = Number(String(actual).trim());
      if (Number.isFinite(leftNumber) && Number.isFinite(rightNumber)) return leftNumber === rightNumber;
      return normalizeEvalText(expected) === normalizeEvalText(actual);
    }

    function normalizeEvalText(value) {
      return String(value || "").trim().toLowerCase().replace(/\s+/g, "");
    }

    function renderBatchEvalResults(done = appState.batchEval.results.length, total = appState.batchEval.rows.length) {
      const results = appState.batchEval.results || [];
      const metric = selectedBatchEvalMetric();
      const counts = batchEvalCounts(results, metric);
      const metricLabel = batchEvalMetricLabel(metric);
      const runModeLabel = batchEvalRunModeLabel(batchEvalRunModeFromSettings());
      const safeTotal = total || counts.total || results.length;
      const percent = safeTotal ? Math.round((done / safeTotal) * 100) : 0;
      $("batchEvalProgressBar").style.width = `${Math.min(Math.max(percent, 0), 100)}%`;
      $("batchEvalSummary").innerHTML = [
        ["评测模式", runModeLabel],
        ["当前指标", metricLabel],
        ["总行数", counts.total],
        ["已评测", counts.done],
        ["正确", counts.pass],
        ["失败", counts.fail],
        ["错误/跳过", `${counts.error} / ${counts.skipped}`],
      ].map(([label, value]) => `
        <div class="stat-card">
          <div class="stat-value">${escapeHtml(value)}</div>
          <div class="row-meta">${escapeHtml(label)}</div>
        </div>
      `).join("");
      if (!results.length) {
        $("batchEvalResults").innerHTML = '<div class="empty">暂无评测结果</div>';
        return;
      }
      $("batchEvalResults").innerHTML = orderedBatchEvalResults(results).map(renderBatchEvalRow).join("");
    }

    function renderBatchEvalRow(item) {
      const metric = selectedBatchEvalMetric();
      const status = batchEvalMetricStatus(item, metric);
      const statusText = batchEvalStatusText(status);
      const expected = item.expectedParts && item.expectedParts.length ? item.expectedParts.map(formatExpectedPart).join("<br>") : "期望不召回备件";
      const actual = item.retrievedParts && item.retrievedParts.length ? item.retrievedParts.map(formatRetrievedPart).join("<br>") : "未召回备件";
      const expectedWorkOrder = item.expectedWorkOrderId ? escapeHtml(item.expectedWorkOrderId) : "未配置预期工单";
      const actualWorkOrders = item.retrievedWorkOrderIds && item.retrievedWorkOrderIds.length
        ? item.retrievedWorkOrderIds.map(id => escapeHtml(id)).join("<br>")
        : "未召回历史工单";
      const reason = batchEvalReason(item, metric);
      const runModeLabel = batchEvalRunModeLabel(batchEvalRunModeFromItem(item));
      return `
        <div class="eval-row ${escapeHtml(status)}">
          <div class="eval-row-head">
            <div class="row-title">#${escapeHtml(item.rowNumber)} ${escapeHtml(statusText)}</div>
            <div class="row-meta">${escapeHtml(runModeLabel)} · ${escapeHtml(batchEvalMetricLabel(metric))}${item.taskId ? ` · task #${escapeHtml(item.taskId)}` : ""}</div>
          </div>
          <div class="row-meta">${escapeHtml(item.question || "")}</div>
          <div class="eval-row-grid">
            <div class="eval-field"><div class="row-meta">真值</div>${expected}</div>
            <div class="eval-field"><div class="row-meta">召回备件证据</div>${actual}</div>
            <div class="eval-field"><div class="row-meta">预期工单</div>${expectedWorkOrder}</div>
            <div class="eval-field"><div class="row-meta">历史工单召回</div>${actualWorkOrders}</div>
          </div>
          ${reason ? `<div class="row-meta">${escapeHtml(reason)}</div>` : ""}
        </div>
      `;
    }

    function orderedBatchEvalResults(results = appState.batchEval.results) {
      return [...(results || [])].sort((left, right) => Number(left.rowNumber || 0) - Number(right.rowNumber || 0));
    }

    function batchEvalReason(item, metric = selectedBatchEvalMetric()) {
      if (item.message) return item.message;
      const match = item.match || {};
      const lines = [];
      const workOrder = match.work_order || {};
      if (metric === "work_order_recall") {
        if (workOrder.required && !workOrder.correct) {
          lines.push(`未召回预期工单：${workOrder.expected || item.expectedWorkOrderId || ""}`);
        }
        return lines.join("；");
      }
      if (metric === "answer_part_recall") {
        const answerMatch = match.answer_part_recall || {};
        if (!batchEvalAnswerMetricAvailable(item)) return "当前评测模式未生成最终回答";
        if (Array.isArray(answerMatch.missing) && answerMatch.missing.length) {
          lines.push(`最终回答未包含：${answerMatch.missing.map(formatExpectedPartText).join("；")}`);
        }
        return lines.join("；") || "最终回答已包含全部预期备件";
      }
      const partMatch = match.part_recall || match;
      if (Array.isArray(partMatch.missing) && partMatch.missing.length) {
        lines.push(`未命中备件：${partMatch.missing.map(formatExpectedPartText).join("；")}`);
      }
      if (!lines.length && metric === "part_recall") return "预期备件均已召回";
      return lines.join("；");
    }

    function formatExpectedPart(part) {
      return escapeHtml(formatExpectedPartText(part));
    }

    function formatExpectedPartText(part) {
      return [
        part.name ? `证据=${part.name}` : "",
        part.code ? `编码=${part.code}` : "",
        part.quantity ? `数量=${part.quantity}` : "",
      ].filter(Boolean).join("，") || "空";
    }

    function formatRetrievedPart(part) {
      return escapeHtml(formatRetrievedPartText(part));
    }

    function formatRetrievedPartText(part) {
      return [
        part.name ? `证据=${part.name}` : "",
        part.code ? `编码=${part.code}` : "",
        part.quantity ? `数量=${part.quantity}` : "",
        part.work_order_id ? `工单=${part.work_order_id}` : "",
      ].filter(Boolean).join("，") || "空";
    }

    function formatRetrievedWorkOrderText(item) {
      const ids = item && item.ids && item.ids.length ? item.ids.join("/") : item && item.id ? item.id : "";
      return [
        item && item.rank ? `#${item.rank}` : "",
        ids ? `工单=${ids}` : "",
        item && item.score !== "" && item.score !== undefined ? `score=${item.score}` : "",
        item && item.title ? `标题=${item.title}` : "",
        item && item.source_path ? `来源=${item.source_path}` : "",
      ].filter(Boolean).join("，") || "空";
    }

    function exportBatchEvalResults() {
      const metric = selectedBatchEvalMetric();
      const rows = [[
        "row",
        "run_mode",
        "selected_metric",
        "selected_metric_status",
        "question",
        "expected_work_order_id",
        "retrieved_work_order_ids",
        "part_recall_match",
        "work_order_recall_match",
        "answer_part_recall_match",
        "expected_parts",
        "retrieved_parts",
        "final_answer",
        "message"
      ]];
      for (const item of orderedBatchEvalResults()) {
        const match = item.match || {};
        const answerMetricStatus = batchEvalMetricStatus(item, "answer_part_recall");
        rows.push([
          item.rowNumber,
          batchEvalRunModeFromItem(item),
          metric,
          batchEvalMetricStatus(item, metric),
          item.question || "",
          item.expectedWorkOrderId || "",
          (item.retrievedWorkOrderIds || []).join(" | "),
          batchEvalMetricCorrect(item, "part_recall") ? "pass" : "fail",
          batchEvalMetricCorrect(item, "work_order_recall") ? "pass" : "fail",
          answerMetricStatus === "skipped" ? "" : answerMetricStatus,
          (item.expectedParts || []).map(formatExpectedPartText).join(" | ") || "期望不召回备件",
          (item.retrievedParts || []).map(formatRetrievedPartText).join(" | "),
          item.answerText || item.answer_text || "",
          batchEvalReason(item, metric),
        ]);
      }
      const csv = rows.map(row => row.map(csvCell).join(",")).join("\n");
      const blob = new Blob(["\uFEFF", csv], {type: "text/csv;charset=utf-8"});
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `waji-rag-batch-eval-${new Date().toISOString().slice(0, 10)}.csv`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    }

    function csvCell(value) {
      const text = String(value ?? "");
      return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
    }

    function currentConfigSnapshot() {
      return {
        version: 1,
        saved_at: new Date().toISOString(),
        ui: {
          active_view: appState.activeView,
          query: $("query").value,
          question_sidebar_open: appState.questionSidebarOpen,
          top_k: $("topK").value,
          evidence_top_k: $("evidenceTopK").value,
          qa_work_order_hybrid: $("qaWorkOrderHybrid").checked,
          qa_manual_hybrid: $("qaManualHybrid").checked,
          work_order_candidate_top_k: $("workOrderCandidateTopK").value,
          work_order_min_relative_score: $("workOrderMinRelativeScore").value,
          work_order_max_hits: $("workOrderMaxHits").value,
          manual_candidate_top_k: $("manualCandidateTopK").value,
          manual_min_relative_score: $("manualMinRelativeScore").value,
          manual_max_hits: $("manualMaxHits").value
        },
        batch_eval: {
          run_mode: $("batchEvalRunMode").value,
          work_order_hybrid: $("batchEvalWorkOrderHybrid").checked,
          manual_hybrid: $("batchEvalManualHybrid").checked,
          work_order_mode: $("batchEvalWorkOrderHybrid").checked ? "hybrid" : "bm25",
          manual_mode: $("batchEvalManualHybrid").checked ? "hybrid" : "bm25",
          top_k: $("batchEvalTopK").value,
          fault_code_top_k: $("batchEvalTopK").value,
          work_order_candidate_top_k: $("batchEvalWorkOrderCandidateTopK").value,
          work_order_min_relative_score: $("batchEvalWorkOrderMinRelativeScore").value,
          work_order_max_hits: $("batchEvalWorkOrderMaxHits").value,
          manual_candidate_top_k: $("batchEvalManualCandidateTopK").value,
          manual_min_relative_score: $("batchEvalManualMinRelativeScore").value,
          manual_max_hits: $("batchEvalManualMaxHits").value,
          concurrency: $("batchEvalConcurrency").value,
          selected_metric: selectedBatchEvalMetric()
        },
        work_order_limit: $("workOrderLimit").value,
        manual_limit: $("manualLimit").value,
        max_manual_chars: $("maxManualChars").value,
        ingest_reset: $("ingestReset").checked,
        ingest_resume: $("ingestResume").checked,
        embedding: {
          enabled: $("enableEmbedding").checked
        },
        rerank: {
          enabled: $("enableRerank").checked
        },
        llm: {
          enabled: $("enableLlm").checked
        }
      };
    }

    function applyConfigSnapshot(config, options = {}) {
      if (!config || typeof config !== "object") throw new Error("配置文件格式不正确");
      const ui = config.ui || {};
      const batchEval = config.batch_eval || {};
      setInputValue("workOrderLimit", config.work_order_limit);
      setInputValue("manualLimit", config.manual_limit);
      setInputValue("maxManualChars", config.max_manual_chars);
      setCheckboxValue("ingestReset", config.ingest_reset);
      setCheckboxValue("ingestResume", config.ingest_resume);
      setInputValue("query", ui.query);
      setInputValue("topK", ui.top_k);
      setInputValue("evidenceTopK", ui.evidence_top_k);
      const retrieval = config.retrieval || {};
      setCheckboxValue("qaWorkOrderHybrid", ui.qa_work_order_hybrid ?? retrieval.work_order_mode === "hybrid");
      setCheckboxValue("qaManualHybrid", ui.qa_manual_hybrid ?? retrieval.manual_mode === "hybrid");
      setInputValue("workOrderCandidateTopK", ui.work_order_candidate_top_k ?? retrieval.work_order_candidate_top_k);
      setInputValue("workOrderMinRelativeScore", ui.work_order_min_relative_score ?? retrieval.work_order_min_relative_score);
      setInputValue("workOrderMaxHits", ui.work_order_max_hits ?? retrieval.work_order_max_hits);
      setInputValue("manualCandidateTopK", ui.manual_candidate_top_k ?? retrieval.manual_candidate_top_k);
      setInputValue("manualMinRelativeScore", ui.manual_min_relative_score ?? retrieval.manual_min_relative_score);
      setInputValue("manualMaxHits", ui.manual_max_hits ?? retrieval.manual_max_hits);
      setInputValue("batchEvalRunMode", batchEval.run_mode ?? ui.batch_eval_run_mode);
      const batchEvalRetrieval = batchEval.retrieval && typeof batchEval.retrieval === "object" ? batchEval.retrieval : {};
      const batchEvalWorkOrderMode = batchEval.work_order_mode ?? batchEvalRetrieval.work_order_mode ?? ui.batch_eval_work_order_mode;
      const batchEvalManualMode = batchEval.manual_mode ?? batchEvalRetrieval.manual_mode ?? ui.batch_eval_manual_mode;
      setCheckboxValue("batchEvalWorkOrderHybrid", batchEval.work_order_hybrid ?? ui.batch_eval_work_order_hybrid ?? (batchEvalWorkOrderMode === "hybrid"));
      setCheckboxValue("batchEvalManualHybrid", batchEval.manual_hybrid ?? ui.batch_eval_manual_hybrid ?? (batchEvalManualMode === "hybrid"));
      setInputValue("batchEvalTopK", batchEval.fault_code_top_k ?? batchEval.top_k ?? ui.batch_eval_top_k);
      setInputValue("batchEvalWorkOrderCandidateTopK", batchEval.work_order_candidate_top_k ?? ui.batch_eval_work_order_candidate_top_k);
      setInputValue("batchEvalWorkOrderMinRelativeScore", batchEval.work_order_min_relative_score ?? ui.batch_eval_work_order_min_relative_score);
      setInputValue("batchEvalWorkOrderMaxHits", batchEval.work_order_max_hits ?? ui.batch_eval_work_order_max_hits);
      setInputValue("batchEvalManualCandidateTopK", batchEval.manual_candidate_top_k ?? ui.batch_eval_manual_candidate_top_k);
      setInputValue("batchEvalManualMinRelativeScore", batchEval.manual_min_relative_score ?? ui.batch_eval_manual_min_relative_score);
      setInputValue("batchEvalManualMaxHits", batchEval.manual_max_hits ?? ui.batch_eval_manual_max_hits);
      setInputValue("batchEvalConcurrency", batchEval.concurrency ?? ui.batch_eval_concurrency);
      if (batchEval.selected_metric) appState.batchEval.selectedMetric = batchEval.selected_metric;
      if (!batchEvalMetricOptionsForSettings($("batchEvalRunMode").value).some(([value]) => value === appState.batchEval.selectedMetric)) {
        appState.batchEval.selectedMetric = "part_recall";
      }
      if (ui.question_sidebar_open !== undefined) {
        setQuestionSidebar(Boolean(ui.question_sidebar_open), {save: false});
      }

      const embedding = config.embedding || {};
      setCheckboxValue("enableEmbedding", embedding.enabled);

      const rerank = config.rerank || {};
      setCheckboxValue("enableRerank", rerank.enabled);

      const llm = config.llm || {};
      setCheckboxValue("enableLlm", llm.enabled);

      if (ui.active_view) switchView(["qa", "batch"].includes(ui.active_view) ? ui.active_view : "build");
      updateQaConfigSummary();
      if (!options.silent) setStatus("配置已导入", "success");
    }

    function setInputValue(id, value) {
      if (value === undefined || value === null) return;
      const element = $(id);
      if (!element) return;
      element.value = String(value);
    }

    function setCheckboxValue(id, value) {
      if (value === undefined || value === null) return;
      const element = $(id);
      if (!element) return;
      element.checked = Boolean(value);
    }

    function saveConfigToLocalStorage() {
      try {
        localStorage.setItem(savedConfigKey, JSON.stringify(currentConfigSnapshot()));
        return true;
      } catch (error) {
        setStatus(`配置保存失败：${error}`, "error");
        return false;
      }
    }

    function restoreConfigFromLocalStorage() {
      try {
        const raw = localStorage.getItem(savedConfigKey);
        if (!raw) return false;
        applyConfigSnapshot(JSON.parse(raw), {silent: true});
        setStatus("已恢复上次保存的配置", "success");
        return true;
      } catch (error) {
        setStatus(`配置恢复失败：${error}`, "error");
        return false;
      }
    }

    function exportConfig() {
      const blob = new Blob([JSON.stringify(currentConfigSnapshot(), null, 2)], {type: "application/json"});
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `waji-rag-config-${new Date().toISOString().slice(0, 10)}.json`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
      saveConfigToLocalStorage();
      setStatus("页面偏好已导出", "success");
    }

    async function importConfig() {
      const file = $("configImportFile").files && $("configImportFile").files[0];
      if (!file) {
        setStatus("请先选择一个 JSON 配置文件", "error");
        return;
      }
      const text = await file.text();
      applyConfigSnapshot(JSON.parse(text));
      saveConfigToLocalStorage();
      setStatus("页面偏好已导入", "success");
    }

    async function restoreSharedConfigFromServer() {
      try {
        const data = await getJson("/api/shared-config");
        renderServerConfigSummary(data || {});
        if (!data || !data.config) return false;
        applyConfigSnapshot(data.config, {silent: true});
        setStatus("已读取服务端配置", "success");
        return true;
      } catch (error) {
        setStatus(`服务端配置读取失败：${error}`, "error");
        return false;
      }
    }

    function renderServerConfigSummary(data) {
      const target = $("serverConfigSummary");
      if (!target) return;
      const config = data && data.config ? data.config : {};
      const path = data && data.path ? data.path : "未配置";
      const database = data && data.database ? data.database : (config.database_url ? "<已配置>" : "使用环境变量或默认值");
      const workOrderDir = config.work_order_dir || "未配置";
      const manualDir = config.manual_dir || "未配置";
      const envFile = config.env_file || "未配置";
      target.innerHTML = `
        配置文件：${escapeHtml(path)}<br>
        数据库：${escapeHtml(database)}<br>
        Env：${escapeHtml(envFile)}<br>
        工单目录：${escapeHtml(workOrderDir)}<br>
        手册目录：${escapeHtml(manualDir)}
      `;
    }

    function bindAutoSave() {
      const ids = [
        "workOrderLimit", "manualLimit", "maxManualChars",
        "ingestReset", "ingestResume", "query", "topK", "evidenceTopK",
        "qaWorkOrderHybrid", "qaManualHybrid",
        "workOrderCandidateTopK", "workOrderMinRelativeScore", "workOrderMaxHits",
        "manualCandidateTopK", "manualMinRelativeScore", "manualMaxHits",
        "batchEvalRunMode", "batchEvalWorkOrderHybrid", "batchEvalManualHybrid",
        "batchEvalTopK", "batchEvalWorkOrderCandidateTopK", "batchEvalWorkOrderMinRelativeScore",
        "batchEvalWorkOrderMaxHits", "batchEvalManualCandidateTopK", "batchEvalManualMinRelativeScore",
        "batchEvalManualMaxHits", "batchEvalConcurrency",
        "enableEmbedding", "enableRerank", "enableLlm"
      ];
      for (const id of ids) {
        const element = $(id);
        if (!element) continue;
        element.addEventListener("change", () => {
          updateQaConfigSummary();
          saveConfigToLocalStorage();
        });
        element.addEventListener("input", () => {
          updateQaConfigSummary();
          saveConfigToLocalStorage();
        });
      }
    }

    function ingestPayload() {
      return {
        ...commonPayload(),
        reset: $("ingestReset").checked,
        resume: $("ingestResume").checked,
        work_order_limit: $("workOrderLimit").value ? Number($("workOrderLimit").value) : null,
        manual_limit: $("manualLimit").value ? Number($("manualLimit").value) : null,
        max_manual_chars: $("maxManualChars").value ? Number($("maxManualChars").value) : 1800
      };
    }

    function queryPayload(query = $("query").value, options = {}) {
      const requestedTopK = options.topK !== undefined ? Number(options.topK) : ($("topK").value ? Number($("topK").value) : 5);
      const useQaModes = options.useQaModes !== false;
      const retrievalOverrides = useQaModes ? qaRetrievalOverrides(options.retrieval) : (options.retrieval || {});
      return {
        ...commonPayload({retrieval: retrievalOverrides, queryRuntime: useQaModes, autoEmbeddingForHybrid: options.autoEmbeddingForHybrid}),
        query: String(query || "").trim(),
        top_k: Number.isFinite(requestedTopK) && requestedTopK > 0 ? requestedTopK : 5,
        debug: true
      };
    }

    async function postJson(url, payload) {
      const response = await fetch(url, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
      });
      const data = await response.json();
      if (!response.ok) {
        const trace = data.traceback ? `\n${String(data.traceback).split("\n").slice(-8).join("\n")}` : "";
        throw new Error(`${data.error || response.statusText}${trace}`);
      }
      return data;
    }

    async function getJson(url) {
      const response = await fetch(url);
      const data = await response.json();
      if (!response.ok) {
        const trace = data.traceback ? `\n${String(data.traceback).split("\n").slice(-8).join("\n")}` : "";
        throw new Error(`${data.error || response.statusText}${trace}`);
      }
      return data;
    }

    function setStatus(text, kind = "") {
      $("status").className = "status" + (kind ? " " + kind : "");
      $("status").textContent = text;
    }

    function renderShellMode() {
      const isBatchHome = appState.activeView === "batch" && !appState.activeBatchEvalTaskId;
      const isBatchOverview = appState.activeView === "batch" && Boolean(appState.activeBatchEvalTaskId) && appState.activeBatchEvalRowNumber === null;
      const isBatchRow = isBatchEvalRowMode();
      $("pageShell").classList.toggle("batch-home-mode", isBatchHome);
      $("workspace").classList.toggle("hidden", isBatchHome || isBatchOverview);
      $("workspace").classList.toggle("batch-row-mode", isBatchRow);
    }

    function isBatchEvalRowMode() {
      return appState.activeView === "batch" && Boolean(appState.activeBatchEvalTaskId) && appState.activeBatchEvalRowNumber !== null;
    }

    function switchView(view) {
      if (appState.activeView === "build") {
        appState.buildStages = appState.stages;
        appState.buildSelectedStage = appState.selectedStage;
      } else {
        syncActiveQuestionState();
      }
      appState.activeView = view;
      if (view === "qa") {
        const tab = ensureQuestionTab($("query").value || defaultQuery);
        const tabState = hydrateStageState("qa", tab.stages, tab.selectedStage || "retrieval");
        tab.stages = tabState.stages;
        tab.selectedStage = tabState.selectedStage;
        appState.activeQuestionTabId = tab.id;
        appState.stages = tab.stages;
        appState.selectedStage = tab.selectedStage;
        appState.lastResult = tab.lastResult || null;
        $("query").value = tab.query;
        renderQuestionTabs();
      } else if (view === "batch") {
        const batchState = hydrateStageState("batch", appState.stages, appState.selectedStage || "retrieval");
        appState.stages = batchState.stages;
        appState.selectedStage = batchState.selectedStage;
        renderBatchEvalPage();
        refreshBatchEvalRuns({quiet: true});
      } else {
        const buildState = hydrateStageState("build", appState.buildStages, appState.buildSelectedStage || "config");
        appState.buildStages = buildState.stages;
        appState.buildSelectedStage = buildState.selectedStage;
        appState.stages = buildState.stages;
        appState.selectedStage = buildState.selectedStage;
      }
      $("buildView").classList.toggle("active", view === "build");
      $("qaView").classList.toggle("active", view === "qa");
      $("batchView").classList.toggle("active", view === "batch");
      $("buildViewBtn").classList.toggle("active", view === "build");
      $("qaViewBtn").classList.toggle("active", view === "qa");
      $("batchViewBtn").classList.toggle("active", view === "batch");
      renderShellMode();
      updateSidebarChrome();
      renderTaskList();
      renderStages();
      renderStageInspector();
      saveConfigToLocalStorage();
    }

    function selectStage(id, options = {}) {
      if (!stageIdSetForView(appState.activeView).has(id)) return;
      appState.selectedStage = id;
      if (options.manual) appState.stageSelectionLocked = true;
      if (appState.activeView === "build") {
        appState.buildSelectedStage = appState.selectedStage;
      }
    }

    function setStage(id, status, data = null, summary = "", options = {}) {
      if (!stageIdSetForView(appState.activeView).has(id)) return;
      appState.stages[id] = {status, data, summary};
      if (options.select !== false) selectStage(id);
      if (appState.activeView === "build") appState.buildStages = appState.stages;
      renderStages();
      renderStageInspector();
      syncActiveQuestionState();
    }

    function resetStages(view = appState.activeView, selectedStage = null, options = {}) {
      const previousLock = appState.stageSelectionLocked;
      const state = createStageState(view, selectedStage);
      appState.stages = state.stages;
      appState.selectedStage = state.selectedStage;
      appState.stageSelectionLocked = options.preserveStageLock ? previousLock : false;
      if (view === "build") {
        appState.buildStages = state.stages;
        appState.buildSelectedStage = state.selectedStage;
      } else if (view === "qa") {
        syncActiveQuestionState();
      }
      renderStages();
      renderStageInspector();
    }

    function renderStages() {
      $("stageListTitle").textContent = appState.activeView === "build" ? "构建阶段" : "回答阶段";
      $("stageList").innerHTML = "";
      for (const [id, title, note] of stageOrderForView(appState.activeView)) {
        const state = appState.stages[id] || {status: "pending", summary: ""};
        const button = document.createElement("button");
        button.className = `stage-node ${state.status || "pending"} ${appState.selectedStage === id ? "selected" : ""}`;
        button.innerHTML = `
          <div class="stage-title">
            <span>${escapeHtml(title)}</span>
            <span class="pill ${state.status === "done" ? "ok" : state.status === "fallback" || state.status === "skipped" || state.status === "filtered" ? "warn" : ""}">${escapeHtml(state.status || "pending")}</span>
          </div>
          <div class="stage-note">${escapeHtml(state.summary || note)}</div>
        `;
        button.addEventListener("click", () => {
          selectStage(id, {manual: true});
          syncActiveQuestionState();
          renderStages();
          renderStageInspector();
        });
        $("stageList").appendChild(button);
      }
    }

    function renderStageInspector() {
      const order = stageOrderForView(appState.activeView);
      const [stageId, title, note] = order.find(([id]) => id === appState.selectedStage) || order[0];
      const state = appState.stages[stageId] || {status: "pending", data: null, summary: ""};
      $("buildView").classList.toggle("active", appState.activeView === "build");
      $("qaView").classList.toggle("active", appState.activeView === "qa");
      $("batchView").classList.toggle("active", appState.activeView === "batch");
      $("buildViewBtn").classList.toggle("active", appState.activeView === "build");
      $("qaViewBtn").classList.toggle("active", appState.activeView === "qa");
      $("batchViewBtn").classList.toggle("active", appState.activeView === "batch");
      renderShellMode();
      renderBatchRetryPanel();
      renderVisiblePanels(stageId);
      $("stageSummary").innerHTML = `
        <div class="evidence-row">
          <div class="row-title">${escapeHtml(title)}</div>
          <div class="row-meta">状态：${escapeHtml(state.status || "pending")}</div>
          <div class="row-meta">${escapeHtml(state.summary || note)}</div>
        </div>
      `;
      $("stageJson").textContent = JSON.stringify(state.data || {}, null, 2);
    }

    function renderVisiblePanels(stageId) {
      const diagnosticView = appState.activeView === "qa" || appState.activeView === "batch";
      const batchRow = isBatchEvalRowMode();
      const batchAnswerRow = batchRow && batchEvalRunModeFromItem(activeBatchEvalRow()) === "answer";
      const showAnswer = diagnosticView && stageId === "answer" && (!batchRow || batchAnswerRow);
      const showRetrieval = diagnosticView && (stageId === "retrieval" || (batchRow && !showAnswer));
      const showEvidence = diagnosticView && !batchRow && ["work_order_filter", "manual_filter", "fact_extraction"].includes(stageId);
      const showInspector = !batchRow && !showAnswer && !showRetrieval && !showEvidence;
      $("answerPanel").classList.toggle("hidden", !showAnswer);
      $("retrievalPanel").classList.toggle("hidden", !showRetrieval);
      $("evidencePanel").classList.toggle("hidden", !showEvidence);
      $("inspectorPanel").classList.toggle("hidden", !showInspector);
      if (showEvidence) {
        renderHarnessStage(stageId);
      }
    }

    function activeBatchEvalRow() {
      return orderedBatchEvalResults().find(row => Number(row.rowNumber) === Number(appState.activeBatchEvalRowNumber)) || null;
    }

    function renderBatchRetryPanel() {
      const panel = $("batchRetryPanel");
      if (!panel) return;
      const html = renderBatchRetryControls();
      panel.classList.toggle("hidden", !html);
      panel.innerHTML = html;
      bindBatchRetryControls();
    }

    function renderBatchRetryControls() {
      const item = activeBatchEvalRow();
      if (appState.activeView !== "batch" || !item) return "";
      const values = batchRetryStateValues();
      const retryTask = appState.activeBatchEvalRetryTaskId ? ` · 最近重试 task #${appState.activeBatchEvalRetryTaskId}` : "";
      const disabled = appState.batchRetry.running ? "disabled" : "";
      return `
          <div>
            <div class="row-title">单题调参重试</div>
            <div class="row-meta">默认显示本次批量评测使用的原始参数；只有点击按钮才会重跑当前问题检索，不会覆盖原始对错结论${escapeHtml(retryTask)}。</div>
          </div>
        <div class="batch-retry-controls">
          <label class="checkline" for="batchRetryWorkOrderHybrid">
            <input id="batchRetryWorkOrderHybrid" type="checkbox" ${values.workOrderHybrid ? "checked" : ""} ${disabled}>
            <span>历史工单使用 hybrid</span>
          </label>
          <label class="checkline" for="batchRetryManualHybrid">
            <input id="batchRetryManualHybrid" type="checkbox" ${values.manualHybrid ? "checked" : ""} ${disabled}>
            <span>故障手册使用 hybrid</span>
          </label>
          <div>
            <label for="batchRetryTopK">故障码 Top K</label>
            <input id="batchRetryTopK" type="number" min="1" value="${escapeHtml(values.topK)}" ${disabled}>
          </div>
          <div>
            <label for="batchRetryManualCandidateTopK">手册候选上限</label>
            <input id="batchRetryManualCandidateTopK" type="number" min="1" value="${escapeHtml(values.manualCandidateTopK)}" ${disabled}>
          </div>
          <div>
            <label for="batchRetryManualMinRelativeScore">手册相对阈值</label>
            <input id="batchRetryManualMinRelativeScore" type="number" min="0" max="1" step="0.05" value="${escapeHtml(values.manualMinRelativeScore)}" ${disabled}>
          </div>
          <div>
            <label for="batchRetryManualMaxHits">手册最大返回</label>
            <input id="batchRetryManualMaxHits" type="number" min="0" value="${escapeHtml(values.manualMaxHits)}" ${disabled}>
          </div>
          <div>
            <label for="batchRetryWorkOrderCandidateTopK">工单候选上限</label>
            <input id="batchRetryWorkOrderCandidateTopK" type="number" min="1" value="${escapeHtml(values.workOrderCandidateTopK)}" ${disabled}>
          </div>
          <div>
            <label for="batchRetryWorkOrderMinRelativeScore">工单相对阈值</label>
            <input id="batchRetryWorkOrderMinRelativeScore" type="number" min="0" max="1" step="0.05" value="${escapeHtml(values.workOrderMinRelativeScore)}" ${disabled}>
          </div>
          <div>
            <label for="batchRetryWorkOrderMaxHits">工单最大返回</label>
            <input id="batchRetryWorkOrderMaxHits" type="number" min="0" value="${escapeHtml(values.workOrderMaxHits)}" ${disabled}>
          </div>
        </div>
        <div class="batch-retry-actions">
          <div class="row-meta">${escapeHtml(item.question || "")}</div>
          <button id="retryBatchQuestionBtn" class="secondary" ${disabled}>按当前参数重试检索</button>
        </div>
        ${renderBatchEvalComparison(item)}
      `;
    }

    function setBatchEvalReplayFromRetrieval(item, retrieval, source) {
      const expectedParts = Array.isArray(item.expectedParts) ? item.expectedParts : [];
      const expectedWorkOrderId = item.expectedWorkOrderId || "";
      const retrievedParts = listPartCandidates(retrieval || {});
      const retrievedWorkOrders = listRetrievedWorkOrders(retrieval || {});
      const retrievedWorkOrderIds = uniqueWorkOrderIdsFromHits(retrievedWorkOrders);
      appState.activeBatchEvalReplay = {
        rowNumber: item.rowNumber,
        source,
        runMode: "search",
        expectedParts,
        expectedWorkOrderId,
        retrievedParts,
        retrievedWorkOrderIds,
        retrievedWorkOrders,
        answerText: "",
        match: evaluateBatchEvalMatch(expectedParts, retrievedParts, expectedWorkOrderId, retrievedWorkOrderIds, "", "search")
      };
    }

    function currentBatchEvalComparison(item) {
      const replay = appState.activeBatchEvalReplay;
      if (replay && Number(replay.rowNumber) === Number(item.rowNumber)) return replay;
      const expectedParts = Array.isArray(item.expectedParts) ? item.expectedParts : [];
      const retrievedParts = Array.isArray(item.retrievedParts) ? item.retrievedParts : [];
      const expectedWorkOrderId = item.expectedWorkOrderId || "";
      const retrievedWorkOrderIds = Array.isArray(item.retrievedWorkOrderIds) ? item.retrievedWorkOrderIds : [];
      const retrievedWorkOrders = Array.isArray(item.retrievedWorkOrders) ? item.retrievedWorkOrders : [];
      const runMode = batchEvalRunModeFromItem(item);
      const answerText = String(item.answerText || item.answer_text || "");
      const match = item.match && (runMode !== "answer" || item.match.answer_part_recall)
        ? item.match
        : evaluateBatchEvalMatch(expectedParts, retrievedParts, expectedWorkOrderId, retrievedWorkOrderIds, answerText, runMode);
      return {
        rowNumber: item.rowNumber,
        source: "原始评测",
        runMode,
        expectedParts,
        expectedWorkOrderId,
        retrievedParts,
        retrievedWorkOrderIds,
        retrievedWorkOrders,
        answerText,
        match
      };
    }

    function renderBatchEvalComparison(item) {
      const comparison = currentBatchEvalComparison(item);
      const expected = comparison.expectedParts.length
        ? comparison.expectedParts.map(part => `<div class="row-meta">${escapeHtml(formatExpectedPartText(part))}</div>`).join("")
        : '<div class="empty">期望不召回备件</div>';
      const actual = comparison.retrievedParts.length
        ? comparison.retrievedParts.map(part => `<div class="row-meta">${formatRetrievedPart(part)}</div>`).join("")
        : '<div class="empty">未召回备件</div>';
      const expectedWorkOrder = comparison.expectedWorkOrderId
        ? `<div class="row-meta">${escapeHtml(comparison.expectedWorkOrderId)}</div>`
        : '<div class="empty">未配置预期工单</div>';
      const actualWorkOrders = comparison.retrievedWorkOrders && comparison.retrievedWorkOrders.length
        ? comparison.retrievedWorkOrders.map(item => `<div class="row-meta">${escapeHtml(formatRetrievedWorkOrderText(item))}</div>`).join("")
        : comparison.retrievedWorkOrderIds && comparison.retrievedWorkOrderIds.length
          ? comparison.retrievedWorkOrderIds.map(id => `<div class="row-meta">${escapeHtml(id)}</div>`).join("")
          : '<div class="empty">未召回历史工单</div>';
      return `
        <div class="batch-comparison-card">
          <div class="batch-comparison-head">
            <div>
              <div class="row-title">预期答案 / 当前召回对比</div>
              <div class="row-meta">当前召回来源：${escapeHtml(comparison.source || "原始评测")}</div>
            </div>
          </div>
          <div class="batch-comparison-grid">
            <div class="batch-comparison-field batch-comparison-expected-parts">
              <div class="row-title">预期备件</div>
              ${expected}
            </div>
            <div class="batch-comparison-field batch-comparison-retrieved-parts">
              <div class="row-title">当前召回备件</div>
              ${actual}
            </div>
            <div class="batch-comparison-field batch-comparison-expected-work-orders">
              <div class="row-title">预期工单</div>
              ${expectedWorkOrder}
            </div>
            <div class="batch-comparison-field batch-comparison-retrieved-work-orders">
              <div class="row-title">当前召回工单</div>
              ${actualWorkOrders}
            </div>
            <div class="batch-comparison-field batch-comparison-summary">
              <div class="row-title">指标结论</div>
              ${renderBatchEvalMetricConclusions(comparison.match, comparison.runMode)}
              ${comparison.answerText ? `
                <div class="metric-conclusion">
                  <div class="row-title">最终回答</div>
                  <div class="batch-answer-preview markdown-body">${renderMarkdown(comparison.answerText)}</div>
                </div>
              ` : ""}
            </div>
          </div>
        </div>
      `;
    }

    function renderBatchEvalMetricConclusions(match, runMode = "search") {
      return batchEvalMetricOptionsForSettings(runMode).map(([metric]) => `
        <div class="metric-conclusion">
          ${renderBatchEvalMatchSummary(match, metric, runMode)}
        </div>
      `).join("");
    }

    function renderBatchEvalMatchSummary(match, metric = selectedBatchEvalMetric(), runMode = "search") {
      const metricPassed = batchEvalMetricCorrect({match, runMode}, metric);
      const label = batchEvalMetricLabel(metric);
      const workOrder = match.work_order || {};
      const lines = [];
      if (metric === "work_order_recall") {
        if (!workOrder.required) {
          lines.push("未配置预期工单，默认通过");
        } else {
          lines.push(workOrder.correct ? `已召回预期工单 ${workOrder.matched || workOrder.expected || ""}` : `未召回预期工单 ${workOrder.expected || ""}`);
        }
      } else if (metric === "answer_part_recall") {
        const answerMatch = match.answer_part_recall || {};
        if (!answerMatch.available) {
          lines.push("当前评测未生成最终回答");
        } else if (!answerMatch.required) {
          lines.push("未配置预期备件，默认通过");
        } else {
          const matched = Array.isArray(answerMatch.matched) ? answerMatch.matched : [];
          const missing = Array.isArray(answerMatch.missing) ? answerMatch.missing : [];
          if (matched.length) lines.push(`回答已包含：${matched.map(formatExpectedPartText).join("；")}`);
          if (missing.length) lines.push(`回答未包含：${missing.map(formatExpectedPartText).join("；")}`);
        }
      } else {
        const partMatch = match.part_recall || match;
        const missing = Array.isArray(partMatch.missing) ? partMatch.missing : [];
        const matched = Array.isArray(partMatch.matched) ? partMatch.matched : [];
        if (matched.length) lines.push(`命中：${matched.map(item => formatExpectedPartText(item.expected)).join("；")}`);
        if (missing.length) lines.push(`未命中：${missing.map(formatExpectedPartText).join("；")}`);
        if (!matched.length && !missing.length) {
          lines.push("未配置预期备件，默认通过");
        }
      }
      return `
        <div class="match-status ${metricPassed ? "pass" : "fail"}">${escapeHtml(label)}：${metricPassed ? "正确" : "失败"}</div>
        <div class="row-meta">${escapeHtml(lines.join("；") || "无可展示的差异")}</div>
      `;
    }

    function batchRetryValues() {
      const defaults = batchRetryDefaults();
      return {
        topK: numericValueFromElement("batchRetryTopK", appState.batchRetry.topK ?? defaults.topK),
        workOrderHybrid: Boolean($("batchRetryWorkOrderHybrid") ? $("batchRetryWorkOrderHybrid").checked : appState.batchRetry.workOrderHybrid ?? defaults.workOrderHybrid),
        manualHybrid: Boolean($("batchRetryManualHybrid") ? $("batchRetryManualHybrid").checked : appState.batchRetry.manualHybrid ?? defaults.manualHybrid),
        workOrderCandidateTopK: numericValueFromElement(
          "batchRetryWorkOrderCandidateTopK",
          appState.batchRetry.workOrderCandidateTopK ?? defaults.workOrderCandidateTopK
        ),
        workOrderMinRelativeScore: numericValueFromElement(
          "batchRetryWorkOrderMinRelativeScore",
          appState.batchRetry.workOrderMinRelativeScore ?? defaults.workOrderMinRelativeScore
        ),
        workOrderMaxHits: numericValueFromElement(
          "batchRetryWorkOrderMaxHits",
          appState.batchRetry.workOrderMaxHits ?? defaults.workOrderMaxHits
        ),
        manualCandidateTopK: numericValueFromElement(
          "batchRetryManualCandidateTopK",
          appState.batchRetry.manualCandidateTopK ?? defaults.manualCandidateTopK
        ),
        manualMinRelativeScore: numericValueFromElement(
          "batchRetryManualMinRelativeScore",
          appState.batchRetry.manualMinRelativeScore ?? defaults.manualMinRelativeScore
        ),
        manualMaxHits: numericValueFromElement(
          "batchRetryManualMaxHits",
          appState.batchRetry.manualMaxHits ?? defaults.manualMaxHits
        )
      };
    }

    function batchRetryStateValues() {
      const defaults = batchRetryDefaults();
      return {
        topK: appState.batchRetry.topK ?? defaults.topK,
        workOrderHybrid: appState.batchRetry.workOrderHybrid ?? defaults.workOrderHybrid,
        manualHybrid: appState.batchRetry.manualHybrid ?? defaults.manualHybrid,
        workOrderCandidateTopK: appState.batchRetry.workOrderCandidateTopK ?? defaults.workOrderCandidateTopK,
        workOrderMinRelativeScore: appState.batchRetry.workOrderMinRelativeScore ?? defaults.workOrderMinRelativeScore,
        workOrderMaxHits: appState.batchRetry.workOrderMaxHits ?? defaults.workOrderMaxHits,
        manualCandidateTopK: appState.batchRetry.manualCandidateTopK ?? defaults.manualCandidateTopK,
        manualMinRelativeScore: appState.batchRetry.manualMinRelativeScore ?? defaults.manualMinRelativeScore,
        manualMaxHits: appState.batchRetry.manualMaxHits ?? defaults.manualMaxHits
      };
    }

    function numericValueFromElement(id, fallback) {
      const element = $(id);
      if (!element || element.value === "") return Number(fallback);
      return Number(element.value);
    }

    function bindBatchRetryControls() {
      const button = $("retryBatchQuestionBtn");
      if (!button) return;
      const ids = [
        "batchRetryWorkOrderHybrid",
        "batchRetryManualHybrid",
        "batchRetryTopK",
        "batchRetryManualCandidateTopK",
        "batchRetryManualMinRelativeScore",
        "batchRetryManualMaxHits",
        "batchRetryWorkOrderCandidateTopK",
        "batchRetryWorkOrderMinRelativeScore",
        "batchRetryWorkOrderMaxHits"
      ];
      for (const id of ids) {
        const element = $(id);
        if (!element) continue;
        element.addEventListener("input", () => {
          appState.batchRetry = {...appState.batchRetry, ...batchRetryValues()};
        });
      }
      button.addEventListener("click", retryActiveBatchQuestion);
    }

    function validateBatchRetryValues(values) {
      if (!Number.isFinite(values.topK) || values.topK < 1) return {ok: false, error: "故障码 Top K 必须大于等于 1"};
      if (!Number.isFinite(values.workOrderCandidateTopK) || values.workOrderCandidateTopK < 1) return {ok: false, error: "工单候选上限必须大于等于 1"};
      if (!Number.isFinite(values.workOrderMinRelativeScore) || values.workOrderMinRelativeScore < 0 || values.workOrderMinRelativeScore > 1) {
        return {ok: false, error: "工单相对阈值必须在 0 到 1 之间"};
      }
      if (!Number.isFinite(values.workOrderMaxHits) || values.workOrderMaxHits < 0) return {ok: false, error: "工单最大返回必须大于等于 0"};
      if (!Number.isFinite(values.manualCandidateTopK) || values.manualCandidateTopK < 1) return {ok: false, error: "手册候选上限必须大于等于 1"};
      if (!Number.isFinite(values.manualMinRelativeScore) || values.manualMinRelativeScore < 0 || values.manualMinRelativeScore > 1) {
        return {ok: false, error: "手册相对阈值必须在 0 到 1 之间"};
      }
      if (!Number.isFinite(values.manualMaxHits) || values.manualMaxHits < 0) return {ok: false, error: "手册最大返回必须大于等于 0"};
      return {
        ok: true,
        topK: Math.floor(values.topK),
        retrieval: {
          work_order_mode: values.workOrderHybrid ? "hybrid" : "bm25",
          manual_mode: values.manualHybrid ? "hybrid" : "bm25",
          work_order_candidate_top_k: Math.floor(values.workOrderCandidateTopK),
          work_order_min_relative_score: values.workOrderMinRelativeScore,
          work_order_max_hits: Math.floor(values.workOrderMaxHits),
          manual_candidate_top_k: Math.floor(values.manualCandidateTopK),
          manual_min_relative_score: values.manualMinRelativeScore,
          manual_max_hits: Math.floor(values.manualMaxHits)
        }
      };
    }

    async function retryActiveBatchQuestion() {
      const item = activeBatchEvalRow();
      if (!item || appState.batchRetry.running) return;
      const values = batchRetryValues();
      const settings = validateBatchRetryValues(values);
      if (!settings.ok) {
        setStatus(settings.error, "error");
        return;
      }
      appState.batchRetry = {
        ...appState.batchRetry,
        ...values,
        topK: settings.topK,
        workOrderHybrid: settings.retrieval.work_order_mode === "hybrid",
        manualHybrid: settings.retrieval.manual_mode === "hybrid",
        workOrderCandidateTopK: settings.retrieval.work_order_candidate_top_k,
        workOrderMinRelativeScore: settings.retrieval.work_order_min_relative_score,
        workOrderMaxHits: settings.retrieval.work_order_max_hits,
        manualCandidateTopK: settings.retrieval.manual_candidate_top_k,
        manualMinRelativeScore: settings.retrieval.manual_min_relative_score,
        manualMaxHits: settings.retrieval.manual_max_hits,
        running: true
      };
      renderStageInspector();
      try {
        const payload = queryPayload(item.question, {topK: settings.topK, retrieval: settings.retrieval, useQaModes: false, autoEmbeddingForHybrid: true});
        setStatus(`正在重试批量评测第 ${item.rowNumber} 行`);
        setStage("retrieval", "active", payload, "按当前参数重试检索");
        const response = await postJson("/api/search-db", payload);
        appState.activeBatchEvalRetryTaskId = response.task_id || null;
        setBatchEvalReplayFromRetrieval(item, response.result || response, "本次重试");
        renderSearchResult(response);
        setStatus(`第 ${item.rowNumber} 行重试完成`, "success");
      } catch (error) {
        setStage("retrieval", "error", {error: String(error), row: item, settings}, "单题重试失败");
        setStatus(String(error), "error");
      } finally {
        appState.batchRetry.running = false;
        renderStageInspector();
      }
    }

    function renderBuildProgress(payload = {}) {
      const progress = payload.progress || {};
      const report = payload.report || {};
      const counts = progress.counts || report || {};
      const timings = progress.timing_seconds || report.timing_seconds || {};
      const percent = progress.percent ?? (report.elapsed_seconds !== undefined ? 100 : 0);
      $("buildProgressBar").style.width = `${Math.max(0, Math.min(100, Number(percent) || 0))}%`;
      $("buildProgressText").textContent = progress.message || payload.summary || "等待构建任务。";
      $("buildCurrentFile").textContent = progress.current_file ? `当前文件：${progress.current_file}` : "";
      $("buildSources").innerHTML = `
        工单目录：${escapeHtml(report.work_order_dir || "由服务端配置文件提供")}<br>
        手册目录：${escapeHtml(report.manual_dir || "由服务端配置文件提供")}<br>
        文件进度：${escapeHtml(progress.processed_files ?? "-")} / ${escapeHtml(progress.total_files ?? "-")}
      `;
      renderBuildStats(counts, progress, timings);
      renderBuildFailures(payload);
    }

    function renderBuildStats(counts = {}, progress = {}, timings = {}) {
      const failedValue = Array.isArray(counts.failed_items) ? counts.failed_items.length : (counts.failed_items ?? progress.failed_count ?? 0);
      const stats = [
        ["工单文件", counts.work_order_files ?? 0],
        ["已解析工单", counts.work_orders ?? 0],
        ["备件记录", counts.part_records ?? 0],
        ["手册文件", counts.manual_files ?? 0],
        ["手册块", counts.manual_chunks ?? 0],
        ["索引文档", counts.total_documents ?? 0],
        ["词项行", counts.term_rows ?? 0],
        ["向量", counts.embeddings ?? 0],
        ["跳过文件", counts.skipped_files ?? 0],
        ["失败", failedValue],
        ["解析耗时", formatSeconds(timings.parse_seconds)],
        ["BM25耗时", formatSeconds(timings.bm25_seconds)],
        ["Embedding耗时", formatSeconds(timings.embedding_seconds)],
        ["PG写入耗时", formatSeconds(timings.pg_write_seconds)]
      ];
      $("buildStats").innerHTML = stats.map(([label, value]) => `
        <div class="stat-card">
          <div class="stat-value">${escapeHtml(value)}</div>
          <div class="row-meta">${escapeHtml(label)}</div>
        </div>
      `).join("");
    }

    function renderBuildFailures(payload = {}) {
      const report = payload.report || {};
      const progress = payload.progress || {};
      const failedItems = Array.isArray(report.failed_items) ? report.failed_items : (Array.isArray(progress.recent_failures) ? progress.recent_failures : []);
      const warnings = Array.isArray(report.warnings) ? report.warnings : (Array.isArray(progress.warnings) ? progress.warnings : []);
      const panel = $("failedItemsPanel");
      if (!failedItems.length && !warnings.length) {
        panel.classList.add("hidden");
        panel.innerHTML = "";
        return;
      }
      const failedHtml = failedItems.length ? failedItems.slice(0, 20).map((item, index) => `
        <div class="failure-row">
          <div class="row-meta">#${index + 1} · ${escapeHtml(item.stage || "unknown")}</div>
          <div class="failure-path">${escapeHtml(item.input || "")}</div>
          <div class="failure-error">${escapeHtml(item.error || "")}</div>
        </div>
      `).join("") : '<div class="empty">暂无失败条目</div>';
      const warningHtml = warnings.length ? `
        <div class="row-title">Warnings</div>
        ${warnings.slice(-8).map(item => `<div class="row-meta">${escapeHtml(item)}</div>`).join("")}
      ` : "";
      panel.classList.remove("hidden");
      panel.innerHTML = `
        <div class="task-line">
          <div class="row-title">失败条目</div>
          <div class="row-meta">${escapeHtml(failedItems.length)} 个可查看</div>
        </div>
        ${failedHtml}
        ${warningHtml}
      `;
    }

    async function refreshTasks(options = {}) {
      try {
        const data = await postJson("/api/tasks", taskPayload({limit: 40}));
        appState.tasks = data.tasks || [];
        renderTaskList();
        return data;
      } catch (error) {
        if (!options.quiet) setStatus(String(error), "error");
        appState.tasks = [];
        renderTaskList();
        return {tasks: []};
      }
    }

    async function refreshQuestionTabsFromServer(options = {}) {
      try {
        const data = await postJson("/api/question-tabs", taskPayload({limit: 120}));
        applyServerQuestionTabs(data.question_tabs || []);
        return data;
      } catch (error) {
        if (!options.quiet) setStatus(String(error), "error");
        return {question_tabs: []};
      }
    }

    function applyServerQuestionTabs(serverTabs) {
      const previousActiveId = appState.activeQuestionTabId;
      const previousActiveQuery = activeQuestionTab() ? activeQuestionTab().query.trim() : "";
      const existingById = new Map(appState.questionTabs.map(tab => [tab.id, tab]));
      const existingByQuery = new Map(appState.questionTabs.map(tab => [tab.query.trim(), tab]));
      const nextTabs = [];
      const seenQueries = new Set();
      for (const item of serverTabs) {
        const query = String(item.query || "").trim();
        if (!query || seenQueries.has(query)) continue;
        const existing = existingById.get(item.id) || existingByQuery.get(query);
        const tab = makeQuestionTabFromServer(item);
        if (existing) {
          tab.stages = existing.stages;
          tab.selectedStage = existing.selectedStage;
          if (existing.answerTaskId === tab.answerTaskId && existing.searchTaskId === tab.searchTaskId) {
            tab.lastResult = existing.lastResult;
          }
        }
        nextTabs.push(tab);
        seenQueries.add(query);
      }
      if (nextTabs.length) {
        appState.questionTabs = nextTabs;
      } else if (!appState.questionTabs.length) {
        appState.questionTabs = [makeQuestionTab($("query").value || defaultQuery)];
      }
      const activeById = appState.questionTabs.find(tab => tab.id === previousActiveId);
      const activeByQuery = appState.questionTabs.find(tab => tab.query.trim() === previousActiveQuery);
      const activeTab = activeById || activeByQuery || appState.questionTabs[0];
      appState.activeQuestionTabId = activeTab ? activeTab.id : null;
      if (appState.activeView === "qa" && activeTab) {
        const tabState = hydrateStageState("qa", activeTab.stages, activeTab.selectedStage || "retrieval");
        activeTab.stages = tabState.stages;
        activeTab.selectedStage = tabState.selectedStage;
        appState.stages = activeTab.stages;
        appState.selectedStage = activeTab.selectedStage;
        appState.lastResult = activeTab.lastResult || null;
        $("query").value = activeTab.query;
        renderQuestionResult(activeTab);
        loadPersistedQuestionResult(activeTab);
      }
      renderQuestionTabs();
    }

    function renderTaskList() {
      $("taskListTitle").textContent = "历史任务清单";
      if (!appState.tasks.length) {
        $("taskList").innerHTML = '<div class="empty">暂无历史任务</div>';
        return;
      }
      $("taskList").innerHTML = "";
      for (const task of appState.tasks) {
        const button = document.createElement("button");
        button.className = `task-card ${task.status || ""} ${appState.currentTaskId === task.id ? "active" : ""}`;
        button.innerHTML = `
          <div class="task-line">
            <span class="task-title">#${escapeHtml(task.id)} ${escapeHtml(taskTypeLabel(task.task_type))}</span>
            <span class="pill ${task.status === "completed" ? "ok" : task.status === "failed" || task.status === "completed_with_errors" || task.status === "stopped" ? "warn" : ""}">${escapeHtml(task.status || "")}</span>
          </div>
          <div class="task-subtitle">${escapeHtml(task.query || task.summary || "知识库构建任务")}</div>
          <div class="row-meta">${escapeHtml(compactTime(task.created_at))}</div>
        `;
        button.addEventListener("click", () => loadTask(task.id));
        $("taskList").appendChild(button);
      }
    }

    async function loadTask(taskId) {
      try {
        const data = await postJson("/api/task", taskPayload({task_id: taskId}));
        renderStoredTask(data.task);
        $("taskHistoryModal").classList.remove("open");
      } catch (error) {
        setStatus(String(error), "error");
      }
    }

    function renderStoredTask(task) {
      if (!task) return;
      appState.currentTaskId = task.id;
      appState.currentTask = task;
      renderTaskList();
      if (task.task_type === "build" || task.task_type === "build_retry" || task.task_type === "embedding") {
        switchView("build");
        resetStages("build", "config");
        setStage("config", "done", task.request || {}, "已载入任务请求");
        renderBuildTaskResult(task);
      } else if (task.task_type === "batch_eval") {
        applyBatchEvalTask(task);
        switchView("batch");
      } else if (task.task_type === "search") {
        const tab = ensureQuestionTabForQuery(task.query || defaultQuery, {loadPersisted: false});
        tab.searchTaskId = task.id;
        tab.status = task.status || "searched";
        appState.currentTaskId = task.id;
        renderSearchResult(task.result || {});
        syncActiveQuestionState();
      } else {
        const tab = ensureQuestionTabForQuery(task.query || defaultQuery, {loadPersisted: false});
        tab.answerTaskId = task.id;
        tab.status = task.status || "answered";
        appState.currentTaskId = task.id;
        renderPipelineResult(task.result || {});
        if (["running", "pause_requested"].includes(task.status)) {
          tab.status = "answering";
          startAnswerPolling(task.id, tab.id);
        }
        syncActiveQuestionState();
      }
      if (task.status === "failed") {
        const failedStage = task.task_type === "build" || task.task_type === "build_retry" ? "ingest" : task.task_type === "embedding" ? "embedding" : task.task_type === "search" ? "retrieval" : "answer";
        setStage(failedStage, "error", task, task.error || "任务失败");
      }
      setStatus(`已载入任务 #${task.id} · ${taskTypeLabel(task.task_type)} · ${task.status}`, task.status === "failed" ? "error" : "success");
    }

    function renderBuildTaskResult(task) {
      const status = ["running", "pause_requested"].includes(task.status) ? "active" : task.status === "failed" ? "error" : task.status === "completed_with_errors" ? "fallback" : task.status === "paused" ? "fallback" : "done";
      const stageId = task.task_type === "embedding" ? "embedding" : "ingest";
      appState.currentTask = task;
      renderBuildProgress(task.result || {});
      setStage(stageId, status, task.result || {}, task.summary || "构建任务已完成");
      $("answer").textContent = task.summary || "任务已完成。可以继续发起检索或回答任务。";
      renderParts([]);
      renderRetrievalBoard({channels: {}, mode: "", top_k: ""});
      renderSelectedEvidence([]);
      if (["running", "pause_requested"].includes(task.status) && !appState.buildPollTimer) startBuildPolling(task.id);
    }

    function startBuildPolling(taskId) {
      stopBuildPolling();
      appState.buildPollTimer = window.setInterval(async () => {
        try {
          const data = await postJson("/api/task", taskPayload({task_id: taskId}));
          const task = data.task;
          if (!task) return;
          appState.currentTaskId = task.id;
          appState.currentTask = task;
          renderBuildTaskResult(task);
          await refreshTasks({quiet: true});
          if (!["running", "pause_requested"].includes(task.status)) {
            stopBuildPolling();
            setStatus(`构建任务 #${task.id} ${task.status}`, task.status === "failed" ? "error" : "success");
          }
        } catch (error) {
          stopBuildPolling();
          setStatus(String(error), "error");
        }
      }, 1200);
    }

    function stopBuildPolling() {
      if (appState.buildPollTimer) {
        window.clearInterval(appState.buildPollTimer);
        appState.buildPollTimer = null;
      }
    }

    function startAnswerPolling(taskId, questionTabId) {
      if (!taskId) return;
      stopAnswerPolling();
      pollAnswerTask(taskId, questionTabId);
      appState.answerPollTimer = window.setInterval(() => {
        pollAnswerTask(taskId, questionTabId);
      }, 900);
    }

    function stopAnswerPolling() {
      if (appState.answerPollTimer) {
        window.clearInterval(appState.answerPollTimer);
        appState.answerPollTimer = null;
      }
    }

    async function pollAnswerTask(taskId, questionTabId) {
      try {
        const data = await postJson("/api/task", taskPayload({task_id: taskId}));
        const task = data.task;
        if (!task) return;
        renderAnswerTaskResult(task, questionTabId);
        if (!["running", "pause_requested"].includes(task.status)) {
          stopAnswerPolling();
          await refreshTasks({quiet: true});
          await refreshQuestionTabsFromServer({quiet: true});
          setStatus(`问答任务 #${task.id} ${task.status}`, task.status === "failed" ? "error" : "success");
        }
      } catch (error) {
        stopAnswerPolling();
        setStatus(String(error), "error");
        setStage("answer", "error", {error: String(error), task_id: taskId}, "问答轮询失败");
      }
    }

    function renderAnswerTaskResult(task, questionTabId) {
      appState.currentTaskId = task.id;
      appState.currentTask = task;
      const targetTab = appState.questionTabs.find(item => item.id === questionTabId);
      if (targetTab) {
        targetTab.answerTaskId = task.id;
        targetTab.status = ["running", "pause_requested"].includes(task.status) ? "answering" : questionStatusFromTask(task, "answered");
        targetTab.lastResult = task.result && task.result.result ? task.result.result : (task.result || null);
        targetTab.updatedAt = task.updated_at || targetTab.updatedAt;
      }
      if (appState.activeView === "qa" && (!targetTab || appState.activeQuestionTabId === targetTab.id)) {
        renderPipelineResult(task.result || {});
        syncActiveQuestionState();
      }
      renderQuestionTabs();
    }

    function taskTypeLabel(taskType) {
      return {
        build: "构建",
        build_retry: "失败重试",
        embedding: "补Embedding",
        search: "检索",
        answer: "回答",
        batch_eval: "批量评测"
      }[taskType] || taskType || "任务";
    }

    function compactTime(value) {
      if (!value) return "";
      const date = new Date(value);
      if (Number.isNaN(date.getTime())) return value;
      return date.toLocaleString();
    }

    function renderPipelineResult(response) {
      const result = response.result || response;
      appState.lastResult = result;
      const stageView = appState.activeView === "batch" ? "batch" : "qa";
      const selectedBeforeRefresh = stageIdSetForView(stageView).has(appState.selectedStage) ? appState.selectedStage : "retrieval";
      resetStages(stageView, selectedBeforeRefresh, {preserveStageLock: true});
      const retrieval = result.retrieval || result;
      const answer = result.answer || {};
      const activeStage = result.active_stage || response.active_stage || "";
      const activeSummary = result.active_summary || response.summary || "正在执行";
      setStageFromTrace(result, {select: false});
      renderAnswer(answer);
      renderParts(answerPartsForDisplay(result, retrieval));
      renderRetrievalBoard(retrieval);
      renderSelectedEvidence(result.selected_evidence || []);
      if (result.retrieval) setStage("retrieval", "done", result.retrieval, formatRetrievalSummary(result.retrieval), {select: false});
      if (result.answer_harness && result.answer_harness.work_order_filter) {
        setStage("work_order_filter", result.answer_harness.work_order_filter.status || "done", result.answer_harness.work_order_filter, workOrderFilterSummary(result.answer_harness.work_order_filter), {select: false});
      }
      if (result.answer_harness && result.answer_harness.manual_filter) {
        setStage("manual_filter", result.answer_harness.manual_filter.status || "done", result.answer_harness.manual_filter, manualFilterSummary(result.answer_harness.manual_filter), {select: false});
      }
      if (result.answer_harness && result.answer_harness.facts) {
        setStage("fact_extraction", result.answer_harness.facts.status || "done", result.answer_harness.facts, factSummary(result.answer_harness.facts), {select: false});
      }
      if (result.answer) setStage("answer", result.answer.status || "done", result.answer, answerSummary(result.answer), {select: false});
      if (activeStage && activeStage !== "completed") {
        setStage(activeStage, "active", pipelineStageData(activeStage, result), activeSummary, {select: false});
        if (!appState.stageSelectionLocked) {
          selectStage(activeStage);
          renderStages();
          renderStageInspector();
          syncActiveQuestionState();
        }
      }
    }

    function pipelineStageData(stageId, result) {
      if (stageId === "retrieval") return result.retrieval || result;
      if (stageId === "work_order_filter") return (result.answer_harness && result.answer_harness.work_order_filter) || {};
      if (stageId === "manual_filter") return (result.answer_harness && result.answer_harness.manual_filter) || {};
      if (stageId === "fact_extraction") return (result.answer_harness && result.answer_harness.facts) || {};
      if (stageId === "answer") return result.answer || {};
      return result || {};
    }

    function setStageFromTrace(result, options = {}) {
      if (!Array.isArray(result.trace)) return;
      for (const item of result.trace) {
        if (!item || !item.stage) continue;
        setStage(item.stage, normalizeStatus(item.status), item, JSON.stringify(item.details || {}).slice(0, 120), options);
      }
    }

    function renderSearchResult(response) {
      const result = response.result || response;
      appState.lastResult = result;
      resetStages(appState.activeView === "batch" ? "batch" : "qa", "retrieval");
      setStage("retrieval", "done", result, formatRetrievalSummary(result));
      renderRetrievalBoard(result);
      renderParts(result.part_candidates || []);
      $("answer").textContent = "已完成检索。请查看“多路召回”和“阶段返回”。";
      renderSelectedEvidence([]);
    }

    function renderAnswer(answer) {
      const text = answer.text || "尚未生成答案。";
      $("answer").classList.add("markdown-body");
      $("answer").innerHTML = renderMarkdown(text);
    }

    function renderMarkdown(markdown) {
      const lines = String(markdown ?? "").replace(/\r\n?/g, "\n").split("\n");
      const blocks = [];
      let index = 0;
      while (index < lines.length) {
        const line = lines[index];
        if (!line.trim()) {
          index += 1;
          continue;
        }

        const fence = line.match(/^\s*```([A-Za-z0-9_-]+)?\s*$/);
        if (fence) {
          const language = fence[1] ? ` class="language-${escapeHtml(fence[1])}"` : "";
          const codeLines = [];
          index += 1;
          while (index < lines.length && !/^\s*```\s*$/.test(lines[index])) {
            codeLines.push(lines[index]);
            index += 1;
          }
          if (index < lines.length) index += 1;
          blocks.push(`<pre><code${language}>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
          continue;
        }

        if (isMarkdownTable(lines, index)) {
          const header = splitMarkdownTableRow(lines[index]);
          index += 2;
          const rows = [];
          while (index < lines.length && lines[index].includes("|") && lines[index].trim()) {
            rows.push(splitMarkdownTableRow(lines[index]));
            index += 1;
          }
          const headerHtml = header.map(cell => `<th>${renderMarkdownInline(cell)}</th>`).join("");
          const bodyHtml = rows.map(row => {
            const cells = header.map((_, cellIndex) => row[cellIndex] || "");
            return `<tr>${cells.map(cell => `<td>${renderMarkdownInline(cell)}</td>`).join("")}</tr>`;
          }).join("");
          blocks.push(`<table><thead><tr>${headerHtml}</tr></thead><tbody>${bodyHtml}</tbody></table>`);
          continue;
        }

        const heading = line.match(/^(#{1,4})\s+(.+)$/);
        if (heading) {
          const level = heading[1].length;
          blocks.push(`<h${level}>${renderMarkdownInline(heading[2])}</h${level}>`);
          index += 1;
          continue;
        }

        if (/^\s*(-{3,}|\*{3,}|_{3,})\s*$/.test(line)) {
          blocks.push("<hr>");
          index += 1;
          continue;
        }

        const unordered = line.match(/^\s*[-*+]\s+(.+)$/);
        if (unordered) {
          const items = [];
          while (index < lines.length) {
            const item = lines[index].match(/^\s*[-*+]\s+(.+)$/);
            if (!item) break;
            items.push(`<li>${renderMarkdownInline(item[1])}</li>`);
            index += 1;
          }
          blocks.push(`<ul>${items.join("")}</ul>`);
          continue;
        }

        const ordered = line.match(/^\s*\d+[.)]\s+(.+)$/);
        if (ordered) {
          const items = [];
          while (index < lines.length) {
            const item = lines[index].match(/^\s*\d+[.)]\s+(.+)$/);
            if (!item) break;
            items.push(`<li>${renderMarkdownInline(item[1])}</li>`);
            index += 1;
          }
          blocks.push(`<ol>${items.join("")}</ol>`);
          continue;
        }

        const quote = line.match(/^\s*>\s?(.+)$/);
        if (quote) {
          const quoteLines = [];
          while (index < lines.length) {
            const item = lines[index].match(/^\s*>\s?(.+)$/);
            if (!item) break;
            quoteLines.push(item[1]);
            index += 1;
          }
          blocks.push(`<blockquote>${renderMarkdownInline(quoteLines.join(" "))}</blockquote>`);
          continue;
        }

        const paragraph = [];
        while (index < lines.length && lines[index].trim() && !isMarkdownBlockStart(lines, index)) {
          paragraph.push(lines[index].trim());
          index += 1;
        }
        if (paragraph.length) {
          blocks.push(`<p>${renderMarkdownInline(paragraph.join(" "))}</p>`);
        } else {
          blocks.push(`<p>${renderMarkdownInline(line)}</p>`);
          index += 1;
        }
      }
      return blocks.join("\n");
    }

    function renderMarkdownInline(value) {
      const codeTokens = [];
      let text = String(value ?? "").replace(/`([^`]+)`/g, (_, code) => {
        const token = `@@CODE_${codeTokens.length}@@`;
        codeTokens.push(`<code>${escapeHtml(code)}</code>`);
        return token;
      });
      let html = escapeHtml(text);
      html = html.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_, label, url) => {
        const decodedUrl = String(url).replace(/&amp;/g, "&");
        if (!/^(https?:\/\/|#|\/)/i.test(decodedUrl)) return label;
        return `<a href="${escapeHtml(decodedUrl)}" target="_blank" rel="noopener noreferrer">${label}</a>`;
      });
      html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
      html = html.replace(/__([^_]+)__/g, "<strong>$1</strong>");
      html = html.replace(/~~([^~]+)~~/g, "<del>$1</del>");
      html = html.replace(/(^|[^\*])\*([^*]+)\*/g, "$1<em>$2</em>");
      html = html.replace(/(^|[^_])_([^_]+)_/g, "$1<em>$2</em>");
      codeTokens.forEach((codeHtml, tokenIndex) => {
        html = html.replaceAll(`@@CODE_${tokenIndex}@@`, codeHtml);
      });
      return html;
    }

    function isMarkdownBlockStart(lines, index) {
      const line = lines[index] || "";
      return Boolean(
        /^\s*```/.test(line)
        || /^#{1,4}\s+/.test(line)
        || /^\s*(-{3,}|\*{3,}|_{3,})\s*$/.test(line)
        || /^\s*[-*+]\s+/.test(line)
        || /^\s*\d+[.)]\s+/.test(line)
        || /^\s*>\s?/.test(line)
        || isMarkdownTable(lines, index)
      );
    }

    function isMarkdownTable(lines, index) {
      const line = lines[index] || "";
      const next = lines[index + 1] || "";
      return line.includes("|") && isMarkdownTableSeparator(next);
    }

    function isMarkdownTableSeparator(line) {
      const cells = splitMarkdownTableRow(line);
      return cells.length >= 2 && cells.every(cell => /^:?-{3,}:?$/.test(cell));
    }

    function splitMarkdownTableRow(line) {
      let text = String(line || "").trim();
      if (text.startsWith("|")) text = text.slice(1);
      if (text.endsWith("|")) text = text.slice(0, -1);
      return text.split("|").map(cell => cell.trim());
    }

    function answerPartsForDisplay(result, retrieval) {
      const codedParts = result.answer_harness && result.answer_harness.facts && Array.isArray(result.answer_harness.facts.coded_parts)
        ? result.answer_harness.facts.coded_parts
        : [];
      if (codedParts.length) return codedParts;
      return result.part_candidates || retrieval.part_candidates || [];
    }

    function renderParts(parts) {
      if (!parts.length) {
        $("parts").innerHTML = '<div class="empty">暂无备件候选</div>';
        return;
      }
      $("parts").innerHTML = parts.map(part => renderPartEvidenceCard(part)).join("");
    }

    function renderRetrievalBoard(retrieval) {
      const channelPayload = retrieval.channels || {};
      const queryTerms = retrieval.debug && Array.isArray(retrieval.debug.query_terms) ? retrieval.debug.query_terms : [];
      $("retrievalBoard").innerHTML = channels.map(([name, label, plan]) => {
        const hits = channelPayload[name] || [];
        const body = hits.length ? hits.map((hit, index) => renderHit(hit, index)).join("") : '<div class="empty">本路无召回</div>';
        return `
          <div class="route-column">
            <div class="route-head">
              <h3>${escapeHtml(label)} <span class="pill">${hits.length}</span></h3>
              <div class="row-meta">生成方式：${escapeHtml(plan)}</div>
              <div class="row-meta">${escapeHtml(routeLimitText(name, retrieval))}</div>
              <div class="term-list">${queryTerms.slice(0, 10).map(term => `<span class="pill">${escapeHtml(term)}</span>`).join("")}</div>
            </div>
            <div class="route-body">${body}</div>
          </div>
        `;
      }).join("");
    }

    function routeLimitText(name, retrieval) {
      const source = retrieval.part_candidate_source || {};
      if (name === "part_evidence" && source.limit_applied === false) {
        const count = Array.isArray(source.linked_work_order_ids) ? source.linked_work_order_ids.length : 0;
        return `仅工单关联备件 · work_orders=${count}`;
      }
      const channelModes = retrieval.channel_modes || {};
      return `mode=${channelModes[name] || retrieval.mode || ""} · top_k=${retrieval.top_k || ""}`;
    }

    function renderHit(hit, index) {
      if (isPartEvidence(hit)) return renderPartEvidenceCard(hit, index);
      const terms = (hit.matched_terms || []).slice(0, 10)
        .map(term => `<span class="pill">${escapeHtml(term.term || "")} · ${escapeHtml(term.field || "")}</span>`)
        .join("");
      return `
        <div class="hit">
          <div class="hit-title">#${index + 1} ${escapeHtml(hit.title || hit.doc_id || "")}</div>
          <div class="row-meta">score=${escapeHtml(hit.score ?? "")} · ${escapeHtml(hit.doc_type || "")}</div>
          <div class="row-meta">${escapeHtml(hit.doc_id || "")}</div>
          <div class="row-meta">${escapeHtml(hit.source_path || "")}</div>
          <div class="term-list">${terms}</div>
          <div class="hit-preview">${escapeHtml(hit.body_preview || "")}</div>
        </div>
      `;
    }

    function isPartEvidence(item) {
      return item && (item.doc_type === "part_evidence" || item.channel === "part_evidence");
    }

    function renderPartEvidenceCard(item, index = null) {
      const part = partEvidenceDisplay(item || {});
      const prefix = index === null ? "" : `#${index + 1} `;
      return `
        <div class="part-row part-evidence-hit">
          ${prefix ? `<div class="row-title">${escapeHtml(prefix)}备件证据</div>` : ""}
          <div class="part-evidence-fields">
            <div class="part-field">
              <div class="row-meta">新件备件名称</div>
              <div class="part-field-value">${escapeHtml(part.name)}</div>
            </div>
            <div class="part-field">
              <div class="row-meta">新件物料编码</div>
              <div class="part-field-value">${escapeHtml(part.code)}</div>
            </div>
            <div class="part-field">
              <div class="row-meta">新件数量</div>
              <div class="part-field-value">${escapeHtml(part.quantity)}</div>
            </div>
          </div>
        </div>
      `;
    }

    function partEvidenceDisplay(item) {
      const metadata = item.metadata || {};
      const text = String(item.body_preview || item.raw_text || "");
      return {
        name: firstText(
          item.name,
          item.part_name,
          metadata.part_name,
          item.part_number_name,
          metadata.part_number_name,
          extractPartLabel(text, ["新件备件名称", "新件配件名称", "新件物料名称", "新件名称"]),
          item.title
        ) || "未提供",
        quantity: firstText(
          item.quantity,
          metadata.quantity,
          extractPartLabel(text, ["新件数量"])
        ) || "未提供",
        code: firstText(
          item.code,
          item.part_code,
          metadata.part_code,
          extractPartLabel(text, ["新件物料编码", "新件备件编码", "新件配件编码", "新件编码"])
        ) || "未提供"
      };
    }

    function firstText(...values) {
      for (const value of values) {
        const text = String(value ?? "").trim();
        if (text) return text;
      }
      return "";
    }

    function extractPartLabel(text, labels) {
      const source = String(text || "");
      const stopLabels = [
        "旧件备件名称", "新件备件名称", "旧件配件名称", "新件配件名称", "旧件物料名称", "新件物料名称",
        "旧件名称", "新件名称", "新件数量", "旧件数量", "旧件物料编码", "新件物料编码",
        "旧件备件编码", "新件备件编码", "旧件配件编码", "新件配件编码", "旧件编码", "新件编码"
      ];
      for (const label of labels) {
        for (const separator of [":", "："]) {
          const token = `${label}${separator}`;
          const start = source.indexOf(token);
          if (start < 0) continue;
          const valueStart = start + token.length;
          let end = source.length;
          for (const stopLabel of stopLabels) {
            for (const stopSeparator of [":", "："]) {
              const stop = source.indexOf(`${stopLabel}${stopSeparator}`, valueStart);
              if (stop >= 0 && stop < end) end = stop;
            }
          }
          const value = source.slice(valueStart, end).trim().replace(/^[,，;；\s]+|[,，;；\s]+$/g, "");
          if (value) return value;
        }
      }
      return "";
    }

    function renderHarnessStage(stageId) {
      const harness = (appState.lastResult && appState.lastResult.answer_harness) || {};
      if (stageId === "work_order_filter") {
        $("evidencePanelTitle").textContent = "工单筛选";
        $("selectedEvidence").innerHTML = renderWorkOrderFilter(harness.work_order_filter || {});
        return;
      }
      if (stageId === "manual_filter") {
        $("evidencePanelTitle").textContent = "手册筛选";
        $("selectedEvidence").innerHTML = renderManualFilter(harness.manual_filter || {});
        return;
      }
      if (stageId === "fact_extraction") {
        $("evidencePanelTitle").textContent = "事实整理";
        $("selectedEvidence").innerHTML = renderFactExtraction(harness.facts || {});
      }
    }

    function renderWorkOrderFilter(payload) {
      const accepted = payload.accepted || [];
      const rejected = payload.rejected || [];
      const unknown = payload.unknown || [];
      return [
        renderWorkOrderFilterGroup("Accepted", accepted, "进入最终答案"),
        renderWorkOrderFilterGroup("Rejected", rejected, "不进入最终答案"),
        renderWorkOrderFilterGroup("Unknown", unknown, "筛选失败，默认不进入最终答案")
      ].join("");
    }

    function renderWorkOrderFilterGroup(title, items, note) {
      const body = items.length ? items.map(renderWorkOrderFilterItem).join("") : '<div class="empty">暂无</div>';
      return `
        <div class="evidence-row">
          <div class="row-title">${escapeHtml(title)} <span class="pill">${items.length}</span></div>
          <div class="row-meta">${escapeHtml(note)}</div>
          <div class="part-box">${body}</div>
        </div>
      `;
    }

    function renderWorkOrderFilterItem(item) {
      const parts = Array.isArray(item.usable_parts) && item.usable_parts.length
        ? `<div class="row-meta">关联备件：${item.usable_parts.map(partSummary).map(escapeHtml).join("；")}</div>`
        : '<div class="row-meta">关联备件：无</div>';
      const actions = Array.isArray(item.repair_actions) && item.repair_actions.length
        ? `<div class="hit-preview">${escapeHtml(item.repair_actions.join("；"))}</div>`
        : "";
      const error = item.error ? `<div class="row-meta">error=${escapeHtml(item.error)}</div>` : "";
      return `
        <div class="hit">
          <div class="hit-title">${escapeHtml(item.work_order_id || item.title || "未知工单")}</div>
          <div class="row-meta">level=${escapeHtml(item.relevance_level || "")} · score=${escapeHtml(item.score ?? "")}</div>
          <div class="row-meta">${escapeHtml(item.matched_reason || "")}</div>
          <div class="row-meta">${escapeHtml(item.source_path || "")}</div>
          ${actions}
          ${parts}
          ${error}
        </div>
      `;
    }

    function renderManualFilter(payload) {
      const selected = payload.selected || [];
      const rejected = payload.rejected || [];
      return `
        <div class="evidence-row">
          <div class="row-title">Selected <span class="pill">${selected.length}</span></div>
          <div class="part-box">${selected.length ? selected.map(renderManualFilterItem).join("") : '<div class="empty">暂无</div>'}</div>
        </div>
        <div class="evidence-row">
          <div class="row-title">Rejected <span class="pill">${rejected.length}</span></div>
          <div class="part-box">${rejected.length ? rejected.map(renderManualFilterItem).join("") : '<div class="empty">暂无</div>'}</div>
        </div>
      `;
    }

    function renderManualFilterItem(item) {
      return `
        <div class="hit">
          <div class="hit-title">${escapeHtml(item.title || item.doc_id || "未知手册")}</div>
          <div class="row-meta">doc_id=${escapeHtml(item.doc_id || "")} · level=${escapeHtml(item.relevance_level || "")} · score=${escapeHtml(item.score ?? "")}</div>
          <div class="row-meta">${escapeHtml(item.reason || "")}</div>
          <div class="row-meta">${escapeHtml(item.source_path || "")}</div>
          <div class="hit-preview">${escapeHtml(item.body_preview || "")}</div>
        </div>
      `;
    }

    function renderFactExtraction(facts) {
      return `
        ${renderFactGroup("故障码摘要", facts.fault_code_facts || [])}
        ${renderFactGroup("工单处理方式归并", facts.work_order_groups || [])}
        ${renderFactGroup("手册摘要", facts.manual_summaries || [])}
        <div class="evidence-row">
          <div class="row-title">备件汇总 JSON</div>
          <pre class="json-box">${escapeHtml(JSON.stringify({
            coded_parts: facts.coded_parts || [],
            uncoded_possible_parts: facts.uncoded_possible_parts || []
          }, null, 2))}</pre>
        </div>
      `;
    }

    function renderFactGroup(title, items) {
      if (!items.length) {
        return `
          <div class="evidence-row">
            <div class="row-title">${escapeHtml(title)} <span class="pill">0</span></div>
            <div class="empty">暂无</div>
          </div>
        `;
      }
      return `
        <div class="evidence-row">
          <div class="row-title">${escapeHtml(title)} <span class="pill">${items.length}</span></div>
          <pre class="json-box">${escapeHtml(JSON.stringify(items, null, 2))}</pre>
        </div>
      `;
    }

    function partSummary(part) {
      return `${part.name || part.part_name || "未知备件"} / ${part.code || part.part_code || "无编码"} / ${part.quantity || "无数量"}`;
    }

    function renderSelectedEvidence(items) {
      $("evidencePanelTitle").textContent = "答案生成依据";
      if (!items.length) {
        $("selectedEvidence").innerHTML = '<div class="empty">暂无选中证据</div>';
        return;
      }
      $("selectedEvidence").innerHTML = items.map((item, index) => {
        if (isPartEvidence(item)) return renderPartEvidenceCard(item, index);
        return `
          <div class="evidence-row">
            <div class="row-title">#${index + 1} ${escapeHtml(item.channel || "")} · ${escapeHtml(item.title || "")}</div>
            <div class="row-meta">doc_id=${escapeHtml(item.doc_id || "")} · score=${escapeHtml(item.score ?? "")}</div>
            <div class="hit-preview">${escapeHtml(item.body_preview || "")}</div>
          </div>
        `;
      }).join("");
    }

    function formatRetrievalSummary(retrieval) {
      const channelsPayload = retrieval.channels || {};
      const channelModes = retrieval.channel_modes || {};
      const modeText = channelModes.work_orders || channelModes.manual_typical_faults
        ? `工单=${channelModes.work_orders || retrieval.mode || ""} · 手册=${channelModes.manual_typical_faults || retrieval.mode || ""}`
        : `mode=${retrieval.mode || ""}`;
      return `${modeText} · ` + channels.map(([name, label]) => `${label}:${(channelsPayload[name] || []).length}`).join(" · ");
    }

    function workOrderFilterSummary(payload) {
      return `accepted=${(payload.accepted || []).length} · rejected=${(payload.rejected || []).length} · unknown=${(payload.unknown || []).length}`;
    }

    function manualFilterSummary(payload) {
      return `selected=${(payload.selected || []).length} · rejected=${(payload.rejected || []).length}`;
    }

    function factSummary(payload) {
      return `故障码=${(payload.fault_code_facts || []).length} · 工单归并=${(payload.work_order_groups || []).length} · 手册=${(payload.manual_summaries || []).length} · 编码备件=${(payload.coded_parts || []).length}`;
    }

    function rerankSummary(rerank) {
      if (!rerank.enabled) return "rerank 未启用";
      return `status=${rerank.status || ""} · returned=${(rerank.results || []).length}`;
    }

    function answerSummary(answer) {
      return `status=${answer.status || ""}` + (answer.debug && answer.debug.model ? ` · model=${answer.debug.model}` : "");
    }

    function normalizeStatus(status) {
      if (status === "ok") return "done";
      return status || "done";
    }

    async function runBuild(button) {
      button.disabled = true;
      switchView("build");
      stopBuildPolling();
      resetStages("build", "config");
      try {
        setStatus("配置预览中");
        setStage("config", "active", commonPayload(), "准备读取配置");
        const preview = await postJson("/api/config-preview", commonPayload());
        setStage("config", "done", preview, "配置读取完成");

        setStatus("初始化 PostgreSQL");
        setStage("init", "active", null, "创建或检查表结构");
        const initResult = await postJson("/api/init-db", {
          reset: false
        });
        setStage("init", "done", initResult, "数据库初始化完成");

        setStatus("构建索引中");
        setStage("ingest", "active", ingestPayload(), "解析工单、清洗 HTML、写入索引");
        const ingestResult = await postJson("/api/ingest-db", {...ingestPayload(), async: true});
        appState.currentTaskId = ingestResult.task_id || null;
        renderBuildProgress({progress: {message: ingestResult.summary || "构建任务已启动", percent: 0}});
        setStage("ingest", "active", ingestResult, ingestResult.summary || "构建任务已启动");
        $("answer").textContent = "构建完成。下一步可以点击“检索”查看多路召回结果，或点击“回答”生成最终答案。";
        renderRetrievalBoard({channels: {}, mode: "", top_k: ""});
        renderParts([]);
        renderSelectedEvidence([]);
        await refreshTasks({quiet: true});
        if (appState.currentTaskId) startBuildPolling(appState.currentTaskId);
        setStatus("构建任务已启动，可在“历史任务”清单查看进度", "success");
      } catch (error) {
        setStatus(String(error), "error");
        const current = appState.selectedStage || "ingest";
        setStage(current, "error", {error: String(error)}, "构建失败");
      } finally {
        button.disabled = false;
      }
    }

    async function runResumeBuild(button) {
      button.disabled = true;
      switchView("build");
      stopBuildPolling();
      try {
        setStatus("继续构建中");
        setStage("ingest", "active", ingestPayload(), "从断点继续处理未完成文件");
        const payload = {...ingestPayload(), reset: false, resume: true, async: true};
        const ingestResult = await postJson("/api/ingest-db", payload);
        appState.currentTaskId = ingestResult.task_id || null;
        renderBuildProgress({progress: {message: ingestResult.summary || "继续构建任务已启动", percent: 0}});
        setStage("ingest", "active", ingestResult, ingestResult.summary || "继续构建任务已启动");
        await refreshTasks({quiet: true});
        if (appState.currentTaskId) startBuildPolling(appState.currentTaskId);
        setStatus("继续构建任务已启动", "success");
      } catch (error) {
        setStatus(String(error), "error");
        setStage("ingest", "error", {error: String(error)}, "继续构建失败");
      } finally {
        button.disabled = false;
      }
    }

    async function retryFailedItems(button) {
      if (!appState.currentTaskId) {
        setStatus("请先在“历史任务”清单中选择一个有失败条目的构建任务", "error");
        return;
      }
      button.disabled = true;
      switchView("build");
      stopBuildPolling();
      try {
        setStatus("失败条目重试中");
        const payload = {...ingestPayload(), task_id: appState.currentTaskId, reset: false, resume: true};
        setStage("ingest", "active", payload, "只重试当前任务中的失败文件");
        const result = await postJson("/api/retry-failed-items", payload);
        appState.currentTaskId = result.task_id || null;
        renderBuildProgress({progress: {message: result.summary || "失败条目重试任务已启动", percent: 0}});
        setStage("ingest", "active", result, result.summary || "失败条目重试任务已启动");
        await refreshTasks({quiet: true});
        if (appState.currentTaskId) startBuildPolling(appState.currentTaskId);
        setStatus("失败条目重试任务已启动", "success");
      } catch (error) {
        setStatus(String(error), "error");
        setStage("ingest", "error", {error: String(error)}, "失败条目重试失败");
      } finally {
        button.disabled = false;
      }
    }

    async function runEmbeddingBackfill(button) {
      button.disabled = true;
      switchView("build");
      stopBuildPolling();
      try {
        setStatus("Embedding 补齐中");
        setStage("embedding", "active", commonPayload(), "扫描缺失向量的索引文档");
        const result = await postJson("/api/embed-db", {...commonPayload(), async: true});
        appState.currentTaskId = result.task_id || null;
        renderBuildProgress({progress: {message: result.summary || "Embedding 补齐任务已启动", percent: 0}});
        setStage("embedding", "active", result, result.summary || "Embedding 补齐任务已启动");
        await refreshTasks({quiet: true});
        if (appState.currentTaskId) startBuildPolling(appState.currentTaskId);
        setStatus("Embedding 补齐任务已启动", "success");
      } catch (error) {
        setStatus(String(error), "error");
        setStage("embedding", "error", {error: String(error)}, "Embedding 补齐失败");
      } finally {
        button.disabled = false;
      }
    }

    async function pauseCurrentTask(button) {
      if (!appState.currentTaskId) {
        setStatus("当前没有可暂停的任务", "error");
        return;
      }
      button.disabled = true;
      try {
        const result = await postJson("/api/pause-task", taskPayload({task_id: appState.currentTaskId}));
        setStatus(result.summary || "暂停请求已发送", "success");
        await refreshTasks({quiet: true});
      } catch (error) {
        setStatus(String(error), "error");
      } finally {
        button.disabled = false;
      }
    }

    async function runFullFlow(button) {
      button.disabled = true;
      const fullFlowQuery = $("query").value;
      switchView("build");
      resetStages("build", "config");
      try {
        setStatus("配置预览中");
        setStage("config", "active", commonPayload(), "准备读取配置");
        const preview = await postJson("/api/config-preview", commonPayload());
        setStage("config", "done", preview, "配置读取完成");

        setStatus("初始化 PostgreSQL");
        setStage("init", "active", null, "创建或检查表结构");
        const initResult = await postJson("/api/init-db", {
          reset: false
        });
        setStage("init", "done", initResult, "数据库初始化完成");

        setStatus("构建索引中");
        setStage("ingest", "active", ingestPayload(), "解析工单、清洗 HTML、写入索引");
        const ingestResult = await postJson("/api/ingest-db", ingestPayload());
        appState.currentTaskId = ingestResult.task_id || null;
        setStage("ingest", "done", ingestResult, ingestResult.summary || "入库完成");

        setStatus("多路召回与答案生成中");
        switchView("qa");
        const questionTab = ensureQuestionTabForQuery(fullFlowQuery, {loadPersisted: false});
        resetStages("qa", "retrieval");
        questionTab.status = "answering";
        renderQuestionTabs();
        const payload = queryPayload(questionTab.query);
        setStage("retrieval", "active", payload, "分路召回证据");
        const askResult = await postJson("/api/ask-db", {...payload, async: true});
        appState.currentTaskId = askResult.task_id || appState.currentTaskId;
        questionTab.answerTaskId = askResult.task_id || questionTab.answerTaskId;
        questionTab.status = "answering";
        startAnswerPolling(questionTab.answerTaskId, questionTab.id);
        syncActiveQuestionState();
        await refreshTasks({quiet: true});
        setStatus("问答任务已启动，阶段结果会自动刷新", "success");
      } catch (error) {
        setStatus(String(error), "error");
        const current = appState.selectedStage || "config";
        setStage(current, "error", {error: String(error)}, "执行失败");
      } finally {
        button.disabled = false;
      }
    }

    async function runSearch(button) {
      button.disabled = true;
      const requestedQuery = $("query").value;
      try {
        switchView("qa");
        const questionTab = ensureQuestionTabForQuery(requestedQuery, {loadPersisted: false});
        resetStages("qa", "retrieval");
        questionTab.status = "searching";
        renderQuestionTabs();
        const payload = queryPayload(questionTab.query);
        setStatus("检索中");
        setStage("retrieval", "active", payload, "分路召回证据");
        const result = await postJson("/api/search-db", payload);
        appState.currentTaskId = result.task_id || null;
        questionTab.searchTaskId = result.task_id || questionTab.searchTaskId;
        questionTab.status = "searched";
        renderSearchResult(result);
        syncActiveQuestionState();
        await refreshTasks({quiet: true});
        await refreshQuestionTabsFromServer({quiet: true});
        setStatus("检索完成", "success");
      } catch (error) {
        setStatus(String(error), "error");
        setStage("retrieval", "error", {error: String(error)}, "检索失败");
      } finally {
        button.disabled = false;
      }
    }

    async function runAsk(button) {
      button.disabled = true;
      const requestedQuery = $("query").value;
      try {
        switchView("qa");
        const questionTab = ensureQuestionTabForQuery(requestedQuery, {loadPersisted: false});
        resetStages("qa", "retrieval");
        questionTab.status = "answering";
        renderQuestionTabs();
        const payload = queryPayload(questionTab.query);
        setStatus("问答中");
        setStage("retrieval", "active", payload, "分路召回证据");
        const result = await postJson("/api/ask-db", {...payload, async: true});
        appState.currentTaskId = result.task_id || null;
        questionTab.answerTaskId = result.task_id || questionTab.answerTaskId;
        questionTab.status = "answering";
        startAnswerPolling(questionTab.answerTaskId, questionTab.id);
        syncActiveQuestionState();
        await refreshTasks({quiet: true});
        setStatus("问答任务已启动，阶段结果会自动刷新", "success");
      } catch (error) {
        setStatus(String(error), "error");
        setStage("answer", "error", {error: String(error)}, "问答失败");
      } finally {
        button.disabled = false;
      }
    }

    async function runPreviewConfig(button) {
      button.disabled = true;
      try {
        switchView("build");
        const result = await postJson("/api/config-preview", commonPayload());
        setStage("config", "done", result, "配置读取完成");
        setStatus("配置预览完成", "success");
      } catch (error) {
        setStatus(String(error), "error");
      } finally {
        button.disabled = false;
      }
    }

    async function clearData(button) {
      const confirmed = window.confirm("确定清空当前数据库中的索引数据、构建记录和任务记录吗？此操作不可撤销。");
      if (!confirmed) return;
      button.disabled = true;
      stopBuildPolling();
      try {
        setStatus("正在清空数据库数据");
        const result = await postJson("/api/clear-data", taskPayload({confirm: true}));
        appState.currentTaskId = null;
        appState.lastResult = null;
        appState.tasks = [];
        appState.batchEvalRuns = [];
        appState.activeBatchEvalTaskId = null;
        appState.activeBatchEvalTask = null;
        appState.activeBatchEvalRowNumber = null;
        appState.activeBatchEvalRetryTaskId = null;
        appState.activeBatchEvalReplay = null;
        appState.batchEval.results = [];
        appState.batchEval.taskId = null;
        appState.questionTabs = appState.questionTabs.map(tab => {
          const state = createStageState("qa", "retrieval");
          return {
            ...tab,
            stages: state.stages,
            selectedStage: state.selectedStage,
            lastResult: null,
            searchTaskId: null,
            answerTaskId: null,
            status: "draft"
          };
        });
        switchView("build");
        resetStages("build", "config");
        renderBuildProgress({});
        renderTaskList();
        renderRetrievalBoard({channels: {}, mode: "", top_k: ""});
        renderParts([]);
        renderSelectedEvidence([]);
        $("answer").textContent = "数据已清空。可以重新构建索引。";
        setStage("init", "done", result, "数据库数据已清空");
        await refreshTasks({quiet: true});
        await refreshQuestionTabsFromServer({quiet: true});
        const beforeCounts = result.before_counts || {};
        const clearedRows = Object.values(beforeCounts).reduce((total, value) => total + Number(value || 0), 0);
        setStatus(`数据已清空，删除前共有 ${clearedRows} 行记录`, "success");
      } catch (error) {
        setStatus(String(error), "error");
        setStage("init", "error", {error: String(error)}, "清空数据失败");
      } finally {
        button.disabled = false;
      }
    }

    function applyDemoDefaults(options = {}) {
      $("query").value = defaultQuery;
      $("topK").value = "1";
      $("evidenceTopK").value = "4";
      $("qaWorkOrderHybrid").checked = false;
      $("qaManualHybrid").checked = false;
      $("workOrderCandidateTopK").value = "50";
      $("workOrderMinRelativeScore").value = "0.45";
      $("workOrderMaxHits").value = "10";
      $("manualCandidateTopK").value = "30";
      $("manualMinRelativeScore").value = "0.55";
      $("manualMaxHits").value = "5";
      $("batchEvalRunMode").value = "search";
      $("batchEvalWorkOrderHybrid").checked = false;
      $("batchEvalManualHybrid").checked = false;
      $("batchEvalTopK").value = "1";
      $("batchEvalWorkOrderCandidateTopK").value = "50";
      $("batchEvalWorkOrderMinRelativeScore").value = "0.45";
      $("batchEvalWorkOrderMaxHits").value = "10";
      $("batchEvalManualCandidateTopK").value = "30";
      $("batchEvalManualMinRelativeScore").value = "0.55";
      $("batchEvalManualMaxHits").value = "5";
      $("batchEvalConcurrency").value = "4";
      $("workOrderLimit").value = "";
      $("manualLimit").value = "";
      $("ingestReset").checked = true;
      $("ingestResume").checked = true;
      $("enableEmbedding").checked = false;
      $("enableRerank").checked = false;
      $("enableLlm").checked = false;
      updateQaConfigSummary();
      if (options.save !== false) saveConfigToLocalStorage();
      setStatus("已加载默认页面参数", "success");
    }

    function formatSeconds(value) {
      const number = Number(value || 0);
      if (!Number.isFinite(number) || number <= 0) return "0.00s";
      return `${number.toFixed(number >= 10 ? 1 : 2)}s`;
    }

    function escapeHtml(value) {
      return String(value ?? "").replace(/[&<>"']/g, ch => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
      }[ch]));
    }

    $("openConfigBtn").addEventListener("click", () => $("configModal").classList.add("open"));
    $("closeConfigBtn").addEventListener("click", () => $("configModal").classList.remove("open"));
    $("openQaConfigBtn").addEventListener("click", () => $("qaConfigModal").classList.add("open"));
    $("closeQaConfigBtn").addEventListener("click", () => $("qaConfigModal").classList.remove("open"));
    $("saveQaConfigBtn").addEventListener("click", () => {
      updateQaConfigSummary();
      saveConfigToLocalStorage();
      $("qaConfigModal").classList.remove("open");
      setStatus("回答参数已保存", "success");
    });
    $("openBatchEvalBtn").addEventListener("click", openBatchEvalHome);
    $("batchEvalCsv").addEventListener("change", loadBatchEvalCsv);
    $("runBatchEvalBtn").addEventListener("click", () => runBatchEval());
    $("stopBatchEvalBtn").addEventListener("click", () => {
      appState.batchEval.stopRequested = true;
      $("batchEvalStatus").textContent = "正在停止，当前请求结束后停止。";
    });
    $("exportBatchEvalBtn").addEventListener("click", exportBatchEvalResults);
    $("refreshBatchEvalRunsBtn").addEventListener("click", () => refreshBatchEvalRuns());
    $("backToBatchHomeBtn").addEventListener("click", () => {
      openBatchEvalHome();
    });
    $("openHistoryBtn").addEventListener("click", async () => {
      $("taskHistoryModal").classList.add("open");
      await refreshTasks({quiet: true});
    });
    $("closeHistoryBtn").addEventListener("click", () => $("taskHistoryModal").classList.remove("open"));
    $("saveConfigBtn").addEventListener("click", async () => {
      try {
        saveConfigToLocalStorage();
        setStatus("页面偏好已保存；连接项仍以服务端配置文件为准", "success");
      } catch (error) {
        setStatus(`页面偏好保存失败：${error}`, "error");
      }
    });
    $("loadDemoBtn").addEventListener("click", applyDemoDefaults);
    $("exportConfigBtn").addEventListener("click", exportConfig);
    $("importConfigBtn").addEventListener("click", () => importConfig().catch(error => setStatus(String(error), "error")));
    $("buildViewBtn").addEventListener("click", () => {
      resetWorkbenchBrowserRoute();
      switchView("build");
    });
    $("qaViewBtn").addEventListener("click", () => {
      resetWorkbenchBrowserRoute();
      switchView("qa");
    });
    $("batchViewBtn").addEventListener("click", openBatchEvalHome);
    $("openQuestionSidebarHeaderBtn").addEventListener("click", () => setQuestionSidebar(true));
    $("openQuestionSidebarBtn").addEventListener("click", () => setQuestionSidebar(true));
    $("closeQuestionSidebarBtn").addEventListener("click", () => setQuestionSidebar(false));
    $("newQuestionBtn").addEventListener("click", createNewQuestionTab);
    $("batchMetricSelect").addEventListener("change", () => {
      appState.batchEval.selectedMetric = $("batchMetricSelect").value;
      renderBatchEvalPage();
      saveConfigToLocalStorage();
    });
    $("batchEvalRunMode").addEventListener("change", () => {
      if (!batchEvalMetricOptionsForSettings($("batchEvalRunMode").value).some(([value]) => value === appState.batchEval.selectedMetric)) {
        appState.batchEval.selectedMetric = "part_recall";
      }
      renderBatchMetricSelector();
      saveConfigToLocalStorage();
    });
    $("query").addEventListener("input", syncActiveQuestionInput);
    window.addEventListener("popstate", () => {
      const shareId = routeBatchEvalShareIdFromPath(window.location.pathname);
      if (shareId) {
        loadBatchEvalShareRoute(shareId);
      } else if (appState.activeView === "batch") {
        openBatchEvalHome({updateRoute: false});
      }
    });
    $("previewConfigBtn").addEventListener("click", () => runPreviewConfig($("previewConfigBtn")));
    $("refreshTasksBtn").addEventListener("click", async () => {
      await refreshTasks();
      await refreshQuestionTabsFromServer();
    });
    $("clearDataBtn").addEventListener("click", () => clearData($("clearDataBtn")));
    $("resumeBuildBtn").addEventListener("click", () => runResumeBuild($("resumeBuildBtn")));
    $("retryFailedBtn").addEventListener("click", () => retryFailedItems($("retryFailedBtn")));
    $("runEmbeddingBtn").addEventListener("click", () => runEmbeddingBackfill($("runEmbeddingBtn")));
    $("pauseTaskBtn").addEventListener("click", () => pauseCurrentTask($("pauseTaskBtn")));
    $("runBuildBtn").addEventListener("click", () => runBuild($("runBuildBtn")));
    $("runSearchBtn").addEventListener("click", () => runSearch($("runSearchBtn")));
    $("runAnswerBtn").addEventListener("click", () => runAsk($("runAnswerBtn")));
    $("runQuestionSearchBtn").addEventListener("click", () => runSearch($("runQuestionSearchBtn")));
    $("runQuestionAnswerBtn").addEventListener("click", () => runAsk($("runQuestionAnswerBtn")));
    $("runFullFlowBtn").addEventListener("click", () => runFullFlow($("runFullFlowBtn")));
    $("doctorBtn").addEventListener("click", async () => {
      try {
        switchView("build");
        const result = await getJson("/api/doctor");
        setStage("config", "done", result, "环境检查完成");
        setStatus("环境检查完成", "success");
      } catch (error) {
        setStatus(String(error), "error");
      }
    });

    async function initializeWorkbench() {
      getJson("/api/doctor").then(data => {
        $("version").textContent = data.waji_rag_version + " · " + data.platform;
      }).catch(() => {});
      resetStages("build", "config");
      applyDemoDefaults({save: false});
      bindAutoSave();
      const restoredSharedConfig = await restoreSharedConfigFromServer();
      const restoredLocalConfig = restoreConfigFromLocalStorage();
      if (!appState.questionTabs.length) {
        const tab = makeQuestionTab($("query").value || defaultQuery);
        appState.questionTabs.push(tab);
        appState.activeQuestionTabId = tab.id;
      }
      renderQuestionTabs();
      updateCurrentQuestionTitle();
      setQuestionSidebar(appState.questionSidebarOpen, {save: false});
      renderBuildProgress({});
      const routeLoaded = await loadBatchEvalShareRoute();
      if (!routeLoaded && !restoredLocalConfig && !restoredSharedConfig) switchView("build");
      refreshTasks({quiet: true});
      refreshQuestionTabsFromServer({quiet: true});
      refreshBatchEvalRuns({quiet: true});
    }

    initializeWorkbench().catch(error => setStatus(`页面初始化失败：${error}`, "error"));
  </script>
</body>
</html>
"""

    replacements = {
        "__DEFAULT_QUERY_TEXT__": html.escape(DEFAULT_QUERY),
        "__DEFAULT_QUERY_JSON__": json.dumps(DEFAULT_QUERY, ensure_ascii=False),
        "__DEMO_WORK_ORDER_DIR_TEXT__": html.escape(str(DEMO_WORK_ORDER_DIR)),
        "__DEMO_MANUAL_DIR_TEXT__": html.escape(str(DEMO_MANUAL_DIR)),
        "__DEMO_WORK_ORDER_DIR_JSON__": json.dumps(str(DEMO_WORK_ORDER_DIR), ensure_ascii=False),
        "__DEMO_MANUAL_DIR_JSON__": json.dumps(str(DEMO_MANUAL_DIR), ensure_ascii=False),
        "__DOCARBOR_ENV_PATH__": html.escape(DEFAULT_DOCARBOR_ENV_PATH),
        "__DOCARBOR_ENV_PATH_JSON__": json.dumps(DEFAULT_DOCARBOR_ENV_PATH, ensure_ascii=False),
        "__DASHSCOPE_BASE_URL__": html.escape(DEFAULT_DASHSCOPE_BASE_URL),
        "__DASHSCOPE_BASE_URL_JSON__": json.dumps(DEFAULT_DASHSCOPE_BASE_URL, ensure_ascii=False),
        "__DASHSCOPE_RERANK_BASE_URL__": html.escape(DEFAULT_DASHSCOPE_RERANK_BASE_URL),
        "__OPENAI_BASE_URL_JSON__": json.dumps(DEFAULT_OPENAI_BASE_URL, ensure_ascii=False),
        "__MODEL_API_REQUEST_LOG_PATH__": html.escape(DEFAULT_MODEL_API_REQUEST_LOG_PATH),
        "__MODEL_API_REQUEST_LOG_PATH_JSON__": json.dumps(DEFAULT_MODEL_API_REQUEST_LOG_PATH, ensure_ascii=False),
    }
    for placeholder, value in replacements.items():
        template = template.replace(placeholder, value)
    return template


INDEX_HTML = build_redesigned_index_html()


class RagDebugHandler(BaseHTTPRequestHandler):
    """HTTP handler for the local debugging UI."""

    server_version = "WajiRagDebug/0.3"

    def do_GET(self) -> None:
        """Serve the main page and read-only diagnostics."""

        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_text(INDEX_HTML, content_type="text/html; charset=utf-8")
            return
        if batch_eval_share_path(parsed.path):
            self._send_text(INDEX_HTML, content_type="text/html; charset=utf-8")
            return
        if parsed.path == "/api/doctor":
            self._send_json(doctor_payload())
            return
        if parsed.path == "/api/shared-config":
            self._handle_get_shared_config()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self) -> None:
        """Handle local PG-backed debug actions."""

        parsed = urlparse(self.path)
        if parsed.path == "/api/config-preview":
            self._handle_config_preview()
            return
        if parsed.path == "/api/shared-config":
            self._handle_save_shared_config()
            return
        if parsed.path == "/api/init-db":
            self._handle_init_db()
            return
        if parsed.path == "/api/clear-data":
            self._handle_clear_data()
            return
        if parsed.path == "/api/ingest-db":
            self._handle_ingest_db()
            return
        if parsed.path == "/api/retry-failed-items":
            self._handle_retry_failed_items()
            return
        if parsed.path == "/api/embed-db":
            self._handle_embed_db()
            return
        if parsed.path == "/api/pause-task":
            self._handle_pause_task()
            return
        if parsed.path == "/api/search-db":
            self._handle_search_db()
            return
        if parsed.path == "/api/ask-db":
            self._handle_ask_db()
            return
        if parsed.path == "/api/tasks":
            self._handle_tasks()
            return
        if parsed.path == "/api/question-tabs":
            self._handle_question_tabs()
            return
        if parsed.path == "/api/task":
            self._handle_task()
            return
        if parsed.path == "/api/parse-table":
            self._handle_parse_table()
            return
        if parsed.path == "/api/create-batch-eval":
            self._handle_create_batch_eval()
            return
        if parsed.path == "/api/update-batch-eval":
            self._handle_update_batch_eval()
            return
        if parsed.path == "/api/batch-evals":
            self._handle_batch_evals()
            return
        if parsed.path == "/api/batch-eval-share":
            self._handle_batch_eval_share()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "not found")

    def log_message(self, format: str, *args: object) -> None:
        """Write compact request logs to stderr."""

        super().log_message(format, *args)

    def _handle_config_preview(self) -> None:
        payload = self._read_json()
        try:
            config = load_config(
                config_path_from_payload(payload),
                overrides=config_overrides_from_payload(payload),
                env_path=env_path_from_payload(payload),
            )
            server_payload = redact_secrets(shared_config_payload())
            self._send_json(
                {
                    "config": config.to_dict(),
                    "server_config": server_payload,
                    "server_config_path": str(shared_config_path()),
                    "database": redact_database_url(database_from_payload(payload).database_url),
                }
            )
        except Exception as exc:  # noqa: BLE001 - local debug endpoint.
            self._send_exception_json(exc)

    def _handle_get_shared_config(self) -> None:
        try:
            config_path = shared_config_path()
            if not config_path.exists():
                self._send_json({"config": None, "path": str(config_path), "database": redact_database_url(database_from_payload({}).database_url)})
                return
            payload = shared_config_payload()
            self._send_json(
                {
                    "config": redact_secrets(payload),
                    "path": str(config_path),
                    "database": redact_database_url(database_from_payload({}).database_url),
                }
            )
        except Exception as exc:  # noqa: BLE001 - local debug endpoint.
            self._send_exception_json(exc)

    def _handle_save_shared_config(self) -> None:
        self._read_json()
        self._send_json(
            {
                "error": "server config is file-managed; edit WAJI_WEB_CONFIG_PATH or .git/info/waji-rag-shared-config.json on the server"
            },
            status=HTTPStatus.METHOD_NOT_ALLOWED,
        )

    def _handle_init_db(self) -> None:
        payload = self._read_json()
        try:
            result = PgSchemaManager(database_from_payload(payload)).initialize(reset=bool(payload.get("reset")))
            self._send_json(result)
        except Exception as exc:  # noqa: BLE001 - local debug endpoint.
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_clear_data(self) -> None:
        payload = self._read_json()
        if not bool(payload.get("confirm")):
            self._send_json({"error": "confirm=true is required"}, status=HTTPStatus.BAD_REQUEST)
            return
        try:
            result = clear_application_data(database_from_payload(payload))
            self._send_json(result)
        except Exception as exc:  # noqa: BLE001 - local debug endpoint.
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_ingest_db(self) -> None:
        payload = self._read_json()
        database = database_from_payload(payload)
        task_id: int | None = None
        try:
            task_id = create_task(database, "build", None, task_request_payload(payload))
            if bool(payload.get("async")):
                update_task_progress(
                    database,
                    task_id,
                    {"phase": "queued", "message": "构建任务已进入后台队列", "percent": 0, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
                    "构建任务已启动",
                )
                thread = threading.Thread(
                    target=run_ingest_task,
                    args=(database, task_id, payload),
                    name=f"waji-ingest-{task_id}",
                    daemon=True,
                )
                thread.start()
                self._send_json({"task_id": task_id, "summary": "构建任务已启动", "status": "running"}, status=HTTPStatus.ACCEPTED)
                return

            options = ingest_options_from_payload(payload, database)
            report = PgIngestBuilder(options).ingest()
            response = {"task_id": task_id, "summary": format_ingest_report_summary(report), "report": report.to_dict()}
            finish_task(
                database,
                task_id,
                "completed_with_errors" if report.failed_items else "completed",
                response,
                str(response["summary"]),
            )
            self._send_json(
                response,
                status=HTTPStatus.OK if not report.failed_items else HTTPStatus.MULTI_STATUS,
            )
        except IngestPaused as exc:
            if task_id is not None:
                finish_task(database, task_id, "paused", {"task_id": task_id, "summary": str(exc)}, "任务已暂停")
            self._send_json({"task_id": task_id, "summary": "任务已暂停", "status": "paused"})
        except Exception as exc:  # noqa: BLE001 - local debug endpoint.
            task_update_error = mark_task_failed(database, task_id, exc)
            body: dict[str, object] = {"task_id": task_id}
            if task_update_error:
                body["task_update_error"] = task_update_error
            self._send_exception_json(exc, extra=body)

    def _handle_embed_db(self) -> None:
        payload = self._read_json()
        database = database_from_payload(payload)
        task_id: int | None = None
        try:
            task_id = create_task(database, "embedding", None, task_request_payload(payload))
            if bool(payload.get("async", True)):
                update_task_progress(
                    database,
                    task_id,
                    {"phase": "embedding", "message": "Embedding 补齐任务已进入后台队列", "percent": 0, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
                    "Embedding 补齐任务已启动",
                )
                thread = threading.Thread(
                    target=run_embedding_task,
                    args=(database, task_id, payload),
                    name=f"waji-embedding-{task_id}",
                    daemon=True,
                )
                thread.start()
                self._send_json({"task_id": task_id, "summary": "Embedding 补齐任务已启动", "status": "running"}, status=HTTPStatus.ACCEPTED)
                return

            options = embedding_options_from_payload(payload, database)
            report = PgEmbeddingBackfill(options).backfill()
            response = {"task_id": task_id, "summary": format_embedding_report_summary(report), "report": report.to_dict()}
            finish_task(
                database,
                task_id,
                "completed_with_errors" if report.failed_items else "completed",
                response,
                str(response["summary"]),
            )
            self._send_json(response, status=HTTPStatus.OK if not report.failed_items else HTTPStatus.MULTI_STATUS)
        except IngestPaused as exc:
            if task_id is not None:
                finish_task(database, task_id, "paused", {"task_id": task_id, "summary": str(exc)}, "任务已暂停")
            self._send_json({"task_id": task_id, "summary": "任务已暂停", "status": "paused"})
        except Exception as exc:  # noqa: BLE001 - local debug endpoint.
            task_update_error = mark_task_failed(database, task_id, exc)
            body: dict[str, object] = {"task_id": task_id}
            if task_update_error:
                body["task_update_error"] = task_update_error
            self._send_exception_json(exc, extra=body)

    def _handle_retry_failed_items(self) -> None:
        payload = self._read_json()
        database = database_from_payload(payload)
        task_id: int | None = None
        try:
            source_task_id = int(payload.get("task_id") or 0)
            if source_task_id <= 0:
                self._send_json({"error": "task_id is required"}, status=HTTPStatus.BAD_REQUEST)
                return
            source_task = get_task(database, source_task_id)
            if source_task is None:
                self._send_json({"error": "task not found"}, status=HTTPStatus.NOT_FOUND)
                return
            failed_items = build_failed_items_from_task(source_task)
            source_request = source_task.get("request") if isinstance(source_task.get("request"), dict) else {}
            retry_payload = retry_ingest_payload(payload, failed_items, source_request)
            task_id = create_task(database, "build_retry", None, task_request_payload({**retry_payload, "retry_source_task_id": source_task_id}))
            update_task_progress(
                database,
                task_id,
                {"phase": "queued", "message": f"失败条目重试任务已进入后台队列，共 {len(failed_items)} 个文件", "percent": 0},
                "失败条目重试任务已启动",
            )
            thread = threading.Thread(
                target=run_failed_item_retry_task,
                args=(database, task_id, source_task_id, retry_payload, failed_items),
                name=f"waji-retry-{task_id}",
                daemon=True,
            )
            thread.start()
            self._send_json(
                {"task_id": task_id, "summary": f"失败条目重试任务已启动，共 {len(failed_items)} 个文件", "status": "running"},
                status=HTTPStatus.ACCEPTED,
            )
        except Exception as exc:  # noqa: BLE001 - local debug endpoint.
            task_update_error = mark_task_failed(database, task_id, exc)
            body: dict[str, object] = {"task_id": task_id}
            if task_update_error:
                body["task_update_error"] = task_update_error
            self._send_exception_json(exc, extra=body)

    def _handle_pause_task(self) -> None:
        payload = self._read_json()
        try:
            task_id = int(payload.get("task_id") or 0)
            if task_id <= 0:
                self._send_json({"error": "task_id is required"}, status=HTTPStatus.BAD_REQUEST)
                return
            result = request_task_pause(database_from_payload(payload), task_id)
            self._send_json(result)
        except Exception as exc:  # noqa: BLE001 - local debug endpoint.
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_search_db(self) -> None:
        payload = self._read_json()
        query = str(payload.get("query") or "").strip()
        if not query:
            self._send_json({"error": "query is required"}, status=HTTPStatus.BAD_REQUEST)
            return
        database = database_from_payload(payload)
        task_id: int | None = None
        try:
            task_id = create_task(database, "search", query, task_request_payload(payload))
            result = run_pg_search(
                PgSearchOptions(
                    database=database,
                    query=query,
                    config_path=config_path_from_payload(payload),
                    config_overrides=config_overrides_from_payload(payload),
                    env_path=env_path_from_payload(payload),
                    top_k=int(payload.get("top_k") or 5),
                    include_debug=bool(payload.get("debug")),
                )
            )
            response = {"task_id": task_id, "summary": format_search_summary(result), "result": result}
            finish_task(database, task_id, "completed", response, str(response["summary"]))
            self._send_json(response)
        except Exception as exc:  # noqa: BLE001 - local debug endpoint.
            task_update_error = mark_task_failed(database, task_id, exc)
            body: dict[str, object] = {"task_id": task_id}
            if task_update_error:
                body["task_update_error"] = task_update_error
            self._send_exception_json(exc, extra=body)

    def _handle_ask_db(self) -> None:
        payload = self._read_json()
        query = str(payload.get("query") or "").strip()
        if not query:
            self._send_json({"error": "query is required"}, status=HTTPStatus.BAD_REQUEST)
            return
        database = database_from_payload(payload)
        task_id: int | None = None
        response: dict[str, object] | None = None
        response_status = HTTPStatus.OK
        try:
            task_id = create_task(database, "answer", query, task_request_payload(payload))
            if bool(payload.get("async")):
                update_task_result(
                    database,
                    task_id,
                    "running",
                    {
                        "task_id": task_id,
                        "summary": "问答任务已启动",
                        "active_stage": "retrieval",
                        "result": {
                            "query": query,
                            "active_stage": "retrieval",
                            "active_summary": "正在多路召回证据",
                            "trace": [],
                        },
                    },
                    "问答任务已启动",
                )
                thread = threading.Thread(
                    target=run_answer_task,
                    args=(database, task_id, payload),
                    name=f"waji-answer-{task_id}",
                    daemon=True,
                )
                thread.start()
                response = {"task_id": task_id, "summary": "问答任务已启动", "status": "running"}
                response_status = HTTPStatus.ACCEPTED
            else:
                result = run_pg_pipeline(
                    PgPipelineOptions(
                        database=database,
                        query=query,
                        config_path=config_path_from_payload(payload),
                        config_overrides=config_overrides_from_payload(payload),
                        env_path=env_path_from_payload(payload),
                        top_k=int(payload.get("top_k") or 5),
                        include_debug=bool(payload.get("debug")),
                    )
                )
                answer = result.get("answer") if isinstance(result.get("answer"), dict) else {}
                response = {"task_id": task_id, "summary": str(answer.get("status") or "ok"), "result": result}
                finish_task(database, task_id, "completed", response, str(response["summary"]))
        except Exception as exc:  # noqa: BLE001 - local debug endpoint.
            task_update_error = mark_task_failed(database, task_id, exc)
            body: dict[str, object] = {"task_id": task_id}
            if task_update_error:
                body["task_update_error"] = task_update_error
            self._send_exception_json(exc, extra=body)
            return
        self._send_json(response or {"task_id": task_id}, status=response_status)

    def _handle_tasks(self) -> None:
        payload = self._read_json()
        try:
            tasks = list_tasks(database_from_payload(payload), limit=int(payload.get("limit") or 40))
            self._send_json({"tasks": tasks})
        except Exception as exc:  # noqa: BLE001 - local debug endpoint.
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_question_tabs(self) -> None:
        payload = self._read_json()
        try:
            question_tabs = list_question_tabs(database_from_payload(payload), limit=int(payload.get("limit") or 120))
            self._send_json({"question_tabs": question_tabs})
        except Exception as exc:  # noqa: BLE001 - local debug endpoint.
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_task(self) -> None:
        payload = self._read_json()
        try:
            task_id = int(payload.get("task_id") or 0)
            if task_id <= 0:
                self._send_json({"error": "task_id is required"}, status=HTTPStatus.BAD_REQUEST)
                return
            database = shared_config_database() if payload.get("use_shared_database") else database_from_payload(payload)
            task = get_task(database, task_id)
            if task is None:
                self._send_json({"error": "task not found"}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json({"task": task})
        except Exception as exc:  # noqa: BLE001 - local debug endpoint.
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_parse_table(self) -> None:
        payload = self._read_json()
        try:
            filename = str(payload.get("filename") or "").strip()
            encoded = str(payload.get("data_base64") or "").strip()
            if not filename:
                self._send_json({"error": "filename is required"}, status=HTTPStatus.BAD_REQUEST)
                return
            if not encoded:
                self._send_json({"error": "data_base64 is required"}, status=HTTPStatus.BAD_REQUEST)
                return
            content = base64.b64decode(encoded, validate=True)
            self._send_json(parse_table_file(filename, content))
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # noqa: BLE001 - local debug endpoint.
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_create_batch_eval(self) -> None:
        payload = self._read_json()
        database = database_from_payload(payload)
        try:
            batch_eval = object_payload(payload.get("batch_eval")) or {}
            file_name = str(batch_eval.get("file_name") or "未命名表格")
            row_count = int(batch_eval.get("row_count") or 0)
            settings = batch_eval.get("settings") if isinstance(batch_eval.get("settings"), dict) else {}
            run_mode = str(settings.get("runMode") or settings.get("run_mode") or batch_eval.get("run_mode") or "search")
            query = f"{file_name} · {row_count} 行"
            share_id = generate_unique_batch_eval_share_id(database)
            task_id = create_task(database, "batch_eval", query, task_request_payload(payload))
            result = {
                "task_id": task_id,
                "status": "running",
                "file_name": file_name,
                "share_id": share_id,
                "run_mode": run_mode,
                "headers": batch_eval.get("headers") if isinstance(batch_eval.get("headers"), list) else [],
                "row_count": row_count,
                "settings": settings,
                "selected_metric": batch_eval.get("selected_metric") or "part_recall",
                "question_column": batch_eval.get("question_column"),
                "work_order_column": batch_eval.get("work_order_column"),
                "part_columns": batch_eval.get("part_columns") if isinstance(batch_eval.get("part_columns"), dict) else {},
                "counts": {
                    "total": row_count,
                    "done": 0,
                    "pass": 0,
                    "fail": 0,
                    "error": 0,
                    "skipped": 0,
                    "selected_metric": batch_eval.get("selected_metric") or "part_recall",
                },
                "rows": [],
                "updated_at": iso_datetime(datetime.now(timezone.utc)),
            }
            update_task_result(database, task_id, "running", result, "批量评测运行中：0 / %s" % row_count)
            self._send_json({"task_id": task_id, "query": query, "result": result}, status=HTTPStatus.ACCEPTED)
        except Exception as exc:  # noqa: BLE001 - local debug endpoint.
            self._send_exception_json(exc)

    def _handle_update_batch_eval(self) -> None:
        payload = self._read_json()
        try:
            database = database_from_payload(payload)
            task_id = int(payload.get("task_id") or 0)
            if task_id <= 0:
                self._send_json({"error": "task_id is required"}, status=HTTPStatus.BAD_REQUEST)
                return
            existing_task = get_task(database, task_id)
            existing_result = existing_task.get("result") if existing_task and isinstance(existing_task.get("result"), dict) else {}
            result = object_payload(payload.get("result")) or {}
            status = str(payload.get("status") or result.get("status") or "running")
            if status not in {"running", "completed", "completed_with_errors", "stopped", "failed"}:
                status = "running"
            result["status"] = status
            result["task_id"] = task_id
            if not result.get("share_id"):
                result["share_id"] = existing_result.get("share_id") or generate_unique_batch_eval_share_id(database)
            summary = batch_eval_summary(result)
            error_message = str(result.get("error") or "") if status == "failed" else None
            update_task_result(
                database,
                task_id,
                status,
                result,
                summary,
                error=error_message or None,
            )
            self._send_json({"task_id": task_id, "status": status, "summary": summary})
        except Exception as exc:  # noqa: BLE001 - local debug endpoint.
            self._send_exception_json(exc)

    def _handle_batch_evals(self) -> None:
        payload = self._read_json()
        try:
            database = shared_config_database() if payload.get("use_shared_database") else database_from_payload(payload)
            batch_evals = list_batch_eval_tasks(database, limit=int(payload.get("limit") or 80))
            self._send_json({"batch_evals": batch_evals})
        except Exception as exc:  # noqa: BLE001 - local debug endpoint.
            self._send_exception_json(exc)

    def _handle_batch_eval_share(self) -> None:
        payload = self._read_json()
        try:
            share_id = normalize_batch_eval_share_id(payload.get("share_id"))
            if not share_id:
                self._send_json({"error": "share_id is required"}, status=HTTPStatus.BAD_REQUEST)
                return
            task = get_batch_eval_task_by_share_id(shared_config_database(), share_id)
            if task is None:
                self._send_json({"error": "batch evaluation not found"}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json({"task": task})
        except Exception as exc:  # noqa: BLE001 - local debug endpoint.
            self._send_exception_json(exc)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length", "0") or "0")
        body = self.rfile.read(length).decode("utf-8") if length else "{}"
        data = json.loads(body or "{}")
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object")
        return data

    def _send_exception_json(
        self,
        exc: Exception,
        *,
        status: HTTPStatus = HTTPStatus.INTERNAL_SERVER_ERROR,
        extra: dict[str, object] | None = None,
    ) -> None:
        trace = traceback.format_exc()
        self.log_error("%s %s failed: %s\n%s", self.command, self.path, exc, trace)
        body: dict[str, object] = {
            "error_type": type(exc).__name__,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": trace,
        }
        if extra:
            body.update(extra)
        self._send_json(body, status=status)

    def _send_json(self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text: str, *, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def parse_table_file(filename: str, content: bytes) -> dict[str, Any]:
    """Parse a CSV or XLSX table into headers and row values for batch evaluation."""

    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        return parse_csv_table(content)
    if suffix == ".xlsx":
        return parse_xlsx_table(content)
    if suffix == ".xls":
        raise ValueError("暂不支持旧版 .xls，请另存为 .xlsx 或 .csv")
    raise ValueError("仅支持 .csv 或 .xlsx 文件")


def parse_csv_table(content: bytes) -> dict[str, Any]:
    """Parse CSV bytes using utf-8-sig and return a table payload."""

    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"CSV 需要保存为 UTF-8 编码：{exc}") from exc
    reader = csv.reader(StringIO(text))
    return table_rows_to_payload([[cell for cell in row] for row in reader], "CSV")


def parse_xlsx_table(content: bytes) -> dict[str, Any]:
    """Parse the first worksheet in an XLSX workbook without third-party dependencies."""

    try:
        with zipfile.ZipFile(BytesIO(content)) as workbook:
            sheet_path = _first_xlsx_worksheet_path(workbook)
            shared_strings = _xlsx_shared_strings(workbook)
            rows = _xlsx_sheet_rows(workbook, sheet_path, shared_strings)
    except zipfile.BadZipFile as exc:
        raise ValueError("XLSX 文件格式无效") from exc
    return table_rows_to_payload(rows, "XLSX")


def table_rows_to_payload(rows: list[list[str]], format_name: str) -> dict[str, Any]:
    """Convert raw table rows into the front-end batch-evaluation shape."""

    non_empty_rows = [row for row in rows if any(str(cell or "").strip() for cell in row)]
    if not non_empty_rows:
        raise ValueError(f"{format_name} 内容为空")
    headers = [
        str(value or f"列{index + 1}").strip() or f"列{index + 1}"
        for index, value in enumerate(non_empty_rows[0])
    ]
    records = [
        [str(row[index] if index < len(row) else "") for index in range(len(headers))]
        for row in non_empty_rows[1:]
    ]
    return {"format": format_name, "headers": headers, "rows": records}


def _first_xlsx_worksheet_path(workbook: zipfile.ZipFile) -> str:
    names = set(workbook.namelist())
    if "xl/workbook.xml" in names and "xl/_rels/workbook.xml.rels" in names:
        workbook_root = ET.fromstring(workbook.read("xl/workbook.xml"))
        sheet = next((element for element in workbook_root.iter() if _xml_local_name(element.tag) == "sheet"), None)
        relationship_id = _xlsx_relationship_id(sheet.attrib) if sheet is not None else ""
        if relationship_id:
            rels_root = ET.fromstring(workbook.read("xl/_rels/workbook.xml.rels"))
            for relationship in rels_root.iter():
                if _xml_local_name(relationship.tag) != "Relationship":
                    continue
                if relationship.attrib.get("Id") != relationship_id:
                    continue
                target = relationship.attrib.get("Target", "")
                sheet_path = _xlsx_target_path("xl/workbook.xml", target)
                if sheet_path in names:
                    return sheet_path
    fallback = sorted(name for name in names if name.startswith("xl/worksheets/") and name.endswith(".xml"))
    if not fallback:
        raise ValueError("XLSX 中没有可读取的工作表")
    return fallback[0]


def _xlsx_relationship_id(attributes: dict[str, str]) -> str:
    for key, value in attributes.items():
        if key == "id" or key.endswith("}id"):
            return value
    return ""


def _xlsx_target_path(source_path: str, target: str) -> str:
    if not target:
        return ""
    if target.startswith("/"):
        return posixpath.normpath(target.lstrip("/"))
    return posixpath.normpath(posixpath.join(posixpath.dirname(source_path), target))


def _xlsx_shared_strings(workbook: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in workbook.namelist():
        return []
    root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
    strings: list[str] = []
    for item in root:
        if _xml_local_name(item.tag) != "si":
            continue
        strings.append("".join(text.text or "" for text in item.iter() if _xml_local_name(text.tag) == "t"))
    return strings


def _xlsx_sheet_rows(workbook: zipfile.ZipFile, sheet_path: str, shared_strings: list[str]) -> list[list[str]]:
    root = ET.fromstring(workbook.read(sheet_path))
    rows: list[list[str]] = []
    for row_element in root.iter():
        if _xml_local_name(row_element.tag) != "row":
            continue
        row_values: list[str] = []
        next_column = 0
        for cell_element in row_element:
            if _xml_local_name(cell_element.tag) != "c":
                continue
            column = _xlsx_column_index(cell_element.attrib.get("r", "")) if cell_element.attrib.get("r") else next_column
            while len(row_values) <= column:
                row_values.append("")
            row_values[column] = _xlsx_cell_value(cell_element, shared_strings)
            next_column = column + 1
        rows.append(row_values)
    return rows


def _xlsx_column_index(cell_ref: str) -> int:
    column = 0
    found = False
    for char in cell_ref.upper():
        if not ("A" <= char <= "Z"):
            break
        found = True
        column = column * 26 + (ord(char) - ord("A") + 1)
    return column - 1 if found else 0


def _xlsx_cell_value(cell_element: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell_element.attrib.get("t", "")
    if cell_type == "inlineStr":
        return _xlsx_inline_text(cell_element)
    value_text = _first_child_text(cell_element, "v")
    if cell_type == "s":
        try:
            return shared_strings[int(value_text)]
        except (ValueError, IndexError):
            return ""
    if cell_type == "b":
        return "TRUE" if value_text == "1" else "FALSE"
    if value_text:
        return value_text
    return _xlsx_inline_text(cell_element)


def _xlsx_inline_text(cell_element: ET.Element) -> str:
    return "".join(text.text or "" for text in cell_element.iter() if _xml_local_name(text.tag) == "t")


def _first_child_text(element: ET.Element, name: str) -> str:
    for child in element:
        if _xml_local_name(child.tag) == name:
            return child.text or ""
    return ""


def _xml_local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def shared_config_path() -> Path:
    """Return the server-side shared UI config path."""

    env_path = str(os.getenv("WAJI_WEB_CONFIG_PATH") or "").strip()
    if env_path:
        return Path(env_path)
    if DEFAULT_WEB_CONFIG_PATH.exists():
        return DEFAULT_WEB_CONFIG_PATH
    if DEFAULT_SHARED_CONFIG_PATH.exists():
        return DEFAULT_SHARED_CONFIG_PATH
    return DEFAULT_WEB_CONFIG_PATH


def shared_config_payload() -> dict[str, Any]:
    """Load the server-side web configuration file as a plain dictionary."""

    config_path = shared_config_path()
    if not config_path.exists():
        return {}
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("shared config file must contain a JSON object")
    return payload


def database_from_payload(payload: dict[str, Any]) -> DatabaseOptions:
    """Build database options from the server-side web configuration."""

    _ = payload
    return shared_config_database()


def shared_config_database() -> DatabaseOptions:
    """Build database options from server-side shared config or environment."""

    payload = shared_config_payload()
    database_url = str(payload.get("database_url") or "").strip() or None
    return DatabaseOptions.from_env(database_url)


def config_path_from_payload(payload: dict[str, Any]) -> Path | None:
    """Return the server-managed application config path, ignoring browser input."""

    _ = payload
    server_payload = shared_config_payload()
    return optional_path(server_payload.get("config_path") or server_payload.get("config"))


def env_path_from_payload(payload: dict[str, Any]) -> Path | None:
    """Return the server-managed env file path, ignoring browser input."""

    _ = payload
    return optional_path(shared_config_payload().get("env_file"))


def server_data_path(name: str) -> Path | None:
    """Return a server-managed data path such as work_order_dir or manual_dir."""

    return optional_path(shared_config_payload().get(name))


def data_path_from_payload(payload: dict[str, Any], name: str) -> Path | None:
    """Return an explicit payload data path or the server-managed default."""

    if name in payload:
        return optional_path(payload.get(name))
    return server_data_path(name)


def config_overrides_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Merge server model settings with safe browser-controlled runtime switches."""

    server_overrides = server_app_config_overrides(shared_config_payload())
    browser_overrides = safe_browser_config_overrides(object_payload(payload.get("config_overrides")) or {})
    return deep_merge_dict(server_overrides, browser_overrides)


def server_app_config_overrides(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract model/retrieval config sections from the server-side file."""

    merged: dict[str, Any] = {}
    for key in ("app_config", "config_overrides"):
        section = payload.get(key)
        if isinstance(section, dict):
            merged = deep_merge_dict(merged, {item_key: item_value for item_key, item_value in section.items() if item_key in APP_CONFIG_SECTION_KEYS})
    for key in APP_CONFIG_SECTION_KEYS:
        section = payload.get(key)
        if isinstance(section, dict):
            merged = deep_merge_dict(merged, {key: section})
    return merged


def safe_browser_config_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
    """Keep only non-connection runtime switches from browser-provided overrides."""

    safe: dict[str, Any] = {}
    retrieval = object_payload(overrides.get("retrieval")) or {}
    safe_retrieval = {
        key: retrieval[key]
        for key in (
            "work_order_mode",
            "manual_mode",
            "work_order_candidate_top_k",
            "work_order_min_relative_score",
            "work_order_max_hits",
            "manual_candidate_top_k",
            "manual_min_relative_score",
            "manual_max_hits",
        )
        if key in retrieval
    }
    if safe_retrieval:
        safe["retrieval"] = safe_retrieval
    for section_name in ("embedding", "rerank", "llm"):
        section = object_payload(overrides.get(section_name)) or {}
        if "enabled" in section:
            safe[section_name] = {"enabled": bool(section["enabled"])}
        if section_name == "rerank" and "top_n" in section:
            safe.setdefault(section_name, {})["top_n"] = section["top_n"]
    answer = object_payload(overrides.get("answer")) or {}
    safe_answer = {
        key: answer[key]
        for key in ("evidence_top_k", "include_debug")
        if key in answer
    }
    if safe_answer:
        safe["answer"] = safe_answer
    return safe


def deep_merge_dict(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge plain dictionaries without mutating the inputs."""

    merged = dict(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = deep_merge_dict(existing, value)
        else:
            merged[key] = value
    return merged


def optional_path(value: object) -> Path | None:
    """Return a Path only when the incoming value is non-empty."""

    text = str(value or "").strip()
    return Path(text) if text else None


def optional_int(value: object) -> int | None:
    """Return an integer only when the incoming value is non-empty."""

    if value in (None, ""):
        return None
    return int(value)


def path_list_payload(value: object) -> tuple[Path, ...]:
    """Return a tuple of non-empty paths from a JSON list payload."""

    if value in (None, ""):
        return ()
    if not isinstance(value, list):
        raise ValueError("path list payload must be an array")
    return tuple(Path(str(item)) for item in value if str(item or "").strip())


def object_payload(value: object) -> dict[str, Any] | None:
    """Return a dict payload or None."""

    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("payload section must be an object")
    return value


def task_request_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a redacted task request safe to persist in PostgreSQL."""

    safe_payload = redact_secrets(payload)
    if not isinstance(safe_payload, dict):
        return {}
    database_url = safe_payload.get("database_url")
    if database_url:
        safe_payload["database_url"] = redact_database_url(str(database_url))
    return safe_payload


def ingest_options_from_payload(
    payload: dict[str, Any],
    database: DatabaseOptions,
    *,
    progress_callback: Any | None = None,
    pause_callback: Any | None = None,
) -> PgIngestOptions:
    """Build ingest options from a web request payload."""

    return PgIngestOptions(
        database=database,
        work_order_dir=data_path_from_payload(payload, "work_order_dir"),
        manual_dir=data_path_from_payload(payload, "manual_dir"),
        work_order_paths=path_list_payload(payload.get("work_order_paths")),
        manual_paths=path_list_payload(payload.get("manual_paths")),
        config_path=config_path_from_payload(payload),
        config_overrides=config_overrides_from_payload(payload),
        env_path=env_path_from_payload(payload),
        reset=bool(payload.get("reset")),
        work_order_limit=optional_int(payload.get("work_order_limit")),
        manual_limit=optional_int(payload.get("manual_limit")),
        max_manual_chars=int(payload.get("max_manual_chars") or 1800),
        resume=bool(payload.get("resume", True)),
        progress_callback=progress_callback,
        pause_callback=pause_callback,
    )


def embedding_options_from_payload(
    payload: dict[str, Any],
    database: DatabaseOptions,
    *,
    progress_callback: Any | None = None,
    pause_callback: Any | None = None,
) -> PgEmbeddingOptions:
    """Build embedding backfill options from a web request payload."""

    return PgEmbeddingOptions(
        database=database,
        config_path=config_path_from_payload(payload),
        config_overrides=config_overrides_from_payload(payload),
        env_path=env_path_from_payload(payload),
        limit=optional_int(payload.get("embedding_limit")),
        progress_callback=progress_callback,
        pause_callback=pause_callback,
    )


def run_ingest_task(database: DatabaseOptions, task_id: int, payload: dict[str, Any]) -> None:
    """Run one ingest task in the background and persist progress."""

    try:
        options = ingest_options_from_payload(
            payload,
            database,
            progress_callback=lambda progress: update_task_progress(database, task_id, progress, str(progress.get("message") or "构建中")),
            pause_callback=lambda: is_task_pause_requested(database, task_id),
        )
        report = PgIngestBuilder(options).ingest()
        response = {"task_id": task_id, "summary": format_ingest_report_summary(report), "report": report.to_dict()}
        finish_task(
            database,
            task_id,
            "completed_with_errors" if report.failed_items else "completed",
            response,
            str(response["summary"]),
        )
    except IngestPaused:
        finish_task(database, task_id, "paused", {"task_id": task_id, "summary": "任务已暂停"}, "任务已暂停")
    except Exception as exc:  # noqa: BLE001 - background task must persist failure.
        mark_task_failed(database, task_id, exc)


def run_embedding_task(database: DatabaseOptions, task_id: int, payload: dict[str, Any]) -> None:
    """Run one embedding backfill task in the background and persist progress."""

    try:
        options = embedding_options_from_payload(
            payload,
            database,
            progress_callback=lambda progress: update_task_progress(database, task_id, progress, str(progress.get("message") or "Embedding 补齐中")),
            pause_callback=lambda: is_task_pause_requested(database, task_id),
        )
        report = PgEmbeddingBackfill(options).backfill()
        response = {"task_id": task_id, "summary": format_embedding_report_summary(report), "report": report.to_dict()}
        finish_task(
            database,
            task_id,
            "completed_with_errors" if report.failed_items else "completed",
            response,
            str(response["summary"]),
        )
    except IngestPaused:
        finish_task(database, task_id, "paused", {"task_id": task_id, "summary": "任务已暂停"}, "任务已暂停")
    except Exception as exc:  # noqa: BLE001 - background task must persist failure.
        mark_task_failed(database, task_id, exc)


def run_answer_task(database: DatabaseOptions, task_id: int, payload: dict[str, Any]) -> None:
    """Run one answer task in the background and persist per-stage snapshots."""

    query = str(payload.get("query") or "").strip()

    def persist_progress(progress: dict[str, object]) -> None:
        result = progress.get("result") if isinstance(progress.get("result"), dict) else {}
        active_stage = str(progress.get("active_stage") or result.get("active_stage") or "")
        summary = str(progress.get("summary") or result.get("active_summary") or "问答进行中")
        update_task_result(
            database,
            task_id,
            "running",
            {
                "task_id": task_id,
                "summary": summary,
                "active_stage": active_stage,
                "result": result,
            },
            summary,
        )

    try:
        result = run_pg_pipeline(
            PgPipelineOptions(
                database=database,
                query=query,
                config_path=config_path_from_payload(payload),
                config_overrides=config_overrides_from_payload(payload),
                env_path=env_path_from_payload(payload),
                top_k=int(payload.get("top_k") or 5),
                include_debug=bool(payload.get("debug")),
                progress_callback=persist_progress,
            )
        )
        answer = result.get("answer") if isinstance(result.get("answer"), dict) else {}
        response = {"task_id": task_id, "summary": str(answer.get("status") or "ok"), "result": result}
        finish_task(database, task_id, "completed", response, str(response["summary"]))
    except Exception as exc:  # noqa: BLE001 - background task must persist failure.
        mark_task_failed(database, task_id, exc)


def run_failed_item_retry_task(
    database: DatabaseOptions,
    task_id: int,
    source_task_id: int,
    payload: dict[str, Any],
    failed_items: list[dict[str, str]],
) -> None:
    """Run a build retry for only the failed source files from a previous task."""

    try:
        options = ingest_options_from_payload(
            payload,
            database,
            progress_callback=lambda progress: update_task_progress(database, task_id, progress, str(progress.get("message") or "失败条目重试中")),
            pause_callback=lambda: is_task_pause_requested(database, task_id),
        )
        report = PgIngestBuilder(options).ingest()
        response = {
            "task_id": task_id,
            "retry_source_task_id": source_task_id,
            "retried_items": failed_items,
            "summary": format_ingest_report_summary(report),
            "report": report.to_dict(),
        }
        finish_task(
            database,
            task_id,
            "completed_with_errors" if report.failed_items else "completed",
            response,
            str(response["summary"]),
        )
        resolve_source_task_failures(database, source_task_id, task_id, failed_items, report.failed_items)
    except IngestPaused:
        finish_task(database, task_id, "paused", {"task_id": task_id, "summary": "任务已暂停"}, "任务已暂停")
    except Exception as exc:  # noqa: BLE001 - background task must persist failure.
        mark_task_failed(database, task_id, exc)


def build_failed_items_from_task(task: dict[str, Any]) -> list[dict[str, str]]:
    """Extract retryable build failed items from a persisted task."""

    result = task.get("result") if isinstance(task.get("result"), dict) else {}
    report = result.get("report") if isinstance(result.get("report"), dict) else {}
    raw_items = report.get("failed_items") if isinstance(report.get("failed_items"), list) else []
    failed_items: list[dict[str, str]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        stage = str(item.get("stage") or "").strip()
        source_path = str(item.get("input") or "").strip()
        if stage not in {"work_order", "manual"} or not source_path:
            continue
        failed_items.append({"stage": stage, "input": source_path, "error": str(item.get("error") or "")})
    if not failed_items:
        raise ValueError("selected task has no retryable failed build items")
    return failed_items


def retry_ingest_payload(
    request_payload: dict[str, Any],
    failed_items: list[dict[str, str]],
    source_request: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an ingest payload that targets only failed source files."""

    retry_payload = dict(source_request or {})
    retry_payload.update(request_payload)
    retry_payload.pop("task_id", None)
    work_order_paths = [item["input"] for item in failed_items if item["stage"] == "work_order"]
    manual_paths = [item["input"] for item in failed_items if item["stage"] == "manual"]
    retry_payload["reset"] = False
    retry_payload["resume"] = True
    retry_payload["async"] = False
    retry_payload["work_order_paths"] = work_order_paths
    retry_payload["manual_paths"] = manual_paths
    if not work_order_paths:
        retry_payload["work_order_dir"] = None
    if not manual_paths:
        retry_payload["manual_dir"] = None
    return retry_payload


def resolve_source_task_failures(
    database: DatabaseOptions,
    source_task_id: int,
    retry_task_id: int,
    attempted_items: list[dict[str, str]],
    retry_failed_items: list[dict[str, str]],
) -> None:
    """Mark retried failures as resolved on the original task when possible."""

    source_task = get_task(database, source_task_id)
    if source_task is None:
        return
    result = source_task.get("result") if isinstance(source_task.get("result"), dict) else {}
    report = result.get("report") if isinstance(result.get("report"), dict) else None
    if report is None:
        return
    original_failed = report.get("failed_items") if isinstance(report.get("failed_items"), list) else []
    attempted_keys = failed_item_keys(attempted_items)
    retry_failed_keys = failed_item_keys(retry_failed_items)
    resolved_keys = attempted_keys - retry_failed_keys
    unresolved: list[dict[str, Any]] = []
    for item in original_failed:
        key = failed_item_key(item)
        if key and key in resolved_keys:
            continue
        if isinstance(item, dict):
            unresolved.append(item)
    report["failed_items"] = unresolved
    retry_history = report.get("retry_history") if isinstance(report.get("retry_history"), list) else []
    retry_history.append(
        {
            "retry_task_id": retry_task_id,
            "attempted_count": len(attempted_items),
            "resolved_count": len(resolved_keys),
            "unresolved_count": len(unresolved),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    report["retry_history"] = retry_history
    status = "completed" if not unresolved else "completed_with_errors"
    summary = f"失败条目重试完成：已解决 {len(resolved_keys)} 个，剩余 {len(unresolved)} 个"
    finish_task(database, source_task_id, status, result, summary, error=None if status == "completed" else source_task.get("error"))


def failed_item_keys(items: list[dict[str, str]]) -> set[tuple[str, str]]:
    """Return stable failed-item keys for comparison across retry reports."""

    keys: set[tuple[str, str]] = set()
    for item in items:
        key = failed_item_key(item)
        if key:
            keys.add(key)
    return keys


def failed_item_key(item: object) -> tuple[str, str] | None:
    """Return a comparable failed-item key when an item has stage and input."""

    if not isinstance(item, dict):
        return None
    stage = str(item.get("stage") or "").strip()
    source_path = str(item.get("input") or "").strip()
    if not stage or not source_path:
        return None
    return stage, source_path


def ensure_task_schema(database: DatabaseOptions) -> None:
    """Create the task-history schema once per process and database URL."""

    database_key = database.database_url
    if database_key in _TASK_SCHEMA_DATABASES:
        return
    with _TASK_SCHEMA_LOCK:
        if database_key in _TASK_SCHEMA_DATABASES:
            return
        for attempt in range(_TASK_DB_MAX_ATTEMPTS):
            try:
                with connect(database.database_url) as conn:
                    with conn.cursor() as cur:
                        create_task_schema(cur)
                    conn.commit()
                break
            except Exception as exc:  # noqa: BLE001 - inspect SQLSTATE without importing psycopg.
                if attempt >= _TASK_DB_MAX_ATTEMPTS - 1 or not is_retryable_task_db_error(exc):
                    raise
                task_retry_sleep(attempt)
        _TASK_SCHEMA_DATABASES.add(database_key)


def is_retryable_task_db_error(exc: Exception) -> bool:
    """Return whether a PostgreSQL task-history transaction should be retried."""

    sqlstate = str(getattr(exc, "sqlstate", "") or "")
    if sqlstate in _TASK_DB_RETRY_SQLSTATES:
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    return "deadlock detected" in text or "could not serialize access" in text or "lock not available" in text


def task_retry_sleep(attempt: int) -> None:
    """Sleep briefly before retrying a task-history transaction."""

    time.sleep(min(0.8, 0.05 * (2**attempt)))


def run_task_transaction(database: DatabaseOptions, operation: Any) -> Any:
    """Run a task-history transaction and retry transient PostgreSQL lock failures."""

    ensure_task_schema(database)
    for attempt in range(_TASK_DB_MAX_ATTEMPTS):
        try:
            with connect(database.database_url) as conn:
                with conn.cursor() as cur:
                    result = operation(cur)
                conn.commit()
            return result
        except Exception as exc:  # noqa: BLE001 - inspect SQLSTATE without importing psycopg.
            if attempt >= _TASK_DB_MAX_ATTEMPTS - 1 or not is_retryable_task_db_error(exc):
                raise
            task_retry_sleep(attempt)
    raise RuntimeError("task transaction retry loop exhausted")


def create_task(database: DatabaseOptions, task_type: str, query: str | None, request: dict[str, Any]) -> int:
    """Create a persistent workbench task and return its task ID."""

    def operation(cur: Any) -> int:
        cur.execute(
            """
            INSERT INTO rag_tasks(task_type, status, query, request)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (task_type, "running", query, json_param(redact_secrets(request))),
        )
        row = cur.fetchone()
        if row is None:
            raise RuntimeError("failed to create task")
        return int(row[0])

    return int(run_task_transaction(database, operation))


def update_task_progress(database: DatabaseOptions, task_id: int, progress: dict[str, object], summary: str) -> None:
    """Persist running progress for a task."""

    def operation(cur: Any) -> None:
        cur.execute(
            """
            UPDATE rag_tasks
            SET updated_at = now(),
                status = CASE WHEN status = 'pause_requested' THEN status ELSE %s END,
                result = %s,
                summary = %s
            WHERE id = %s
            """,
            ("running", json_param({"task_id": task_id, "progress": redact_secrets(progress)}), summary, task_id),
        )
        if cur.rowcount != 1:
            raise LookupError(f"task not found: {task_id}")

    run_task_transaction(database, operation)


def update_task_result(
    database: DatabaseOptions,
    task_id: int,
    status: str,
    result: dict[str, Any],
    summary: str,
    *,
    error: str | None = None,
) -> None:
    """Persist a task result snapshot while preserving the existing task row."""

    def operation(cur: Any) -> None:
        cur.execute(
            """
            UPDATE rag_tasks
            SET updated_at = now(), status = %s, result = %s, summary = %s, error = %s
            WHERE id = %s
            """,
            (status, json_param(redact_secrets(result)), summary, error, task_id),
        )
        if cur.rowcount != 1:
            raise LookupError(f"task not found: {task_id}")

    run_task_transaction(database, operation)


def request_task_pause(database: DatabaseOptions, task_id: int) -> dict[str, Any]:
    """Request a running task to pause at its next checkpoint."""

    def operation(cur: Any) -> dict[str, Any]:
        cur.execute(
            """
            UPDATE rag_tasks
            SET updated_at = now(), status = 'pause_requested', summary = %s
            WHERE id = %s AND status IN ('running', 'pause_requested')
            RETURNING id, status
            """,
            ("暂停请求已发送", task_id),
        )
        row = cur.fetchone()
        if row is None:
            raise LookupError(f"running task not found: {task_id}")
        return {"task_id": int(row[0]), "status": str(row[1]), "summary": "暂停请求已发送"}

    return dict(run_task_transaction(database, operation))


def is_task_pause_requested(database: DatabaseOptions, task_id: int) -> bool:
    """Return whether a task has a pending pause request."""

    ensure_task_schema(database)
    with connect(database.database_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT status FROM rag_tasks WHERE id = %s", (task_id,))
            row = cur.fetchone()
    return bool(row and str(row[0]) == "pause_requested")


def finish_task(
    database: DatabaseOptions,
    task_id: int,
    status: str,
    result: dict[str, Any],
    summary: str,
    *,
    error: str | None = None,
) -> None:
    """Persist the final state, result payload, and summary for a task."""

    def operation(cur: Any) -> None:
        cur.execute(
            """
            UPDATE rag_tasks
            SET updated_at = now(), status = %s, result = %s, summary = %s, error = %s
            WHERE id = %s
            """,
            (status, json_param(redact_secrets(result)), summary, error, task_id),
        )
        if cur.rowcount != 1:
            raise LookupError(f"task not found: {task_id}")

    run_task_transaction(database, operation)


def mark_task_failed(database: DatabaseOptions, task_id: int | None, exc: Exception) -> str | None:
    """Mark a task as failed and return any task-update error."""

    if task_id is None:
        return None
    message = f"{type(exc).__name__}: {exc}"
    try:
        finish_task(database, task_id, "failed", {"error": message}, message, error=message)
    except Exception as update_exc:  # noqa: BLE001 - surface both errors in debug UI.
        return f"{type(update_exc).__name__}: {update_exc}"
    return None


def list_tasks(database: DatabaseOptions, *, limit: int = 40) -> list[dict[str, Any]]:
    """Return recent workbench tasks without large request/result payloads."""

    safe_limit = max(1, min(limit, 200))
    ensure_task_schema(database)
    with connect(database.database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, task_type, status, query, summary, error, created_at, updated_at
                FROM rag_tasks
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (safe_limit,),
            )
            rows = cur.fetchall()
        conn.commit()
    return [
        {
            "id": int(row[0]),
            "task_type": row[1],
            "status": row[2],
            "query": row[3],
            "summary": row[4],
            "error": row[5],
            "created_at": iso_datetime(row[6]),
            "updated_at": iso_datetime(row[7]),
        }
        for row in rows
    ]


def list_batch_eval_tasks(database: DatabaseOptions, *, limit: int = 80) -> list[dict[str, Any]]:
    """Return recent batch-evaluation tasks with compact result counts."""

    safe_limit = max(1, min(limit, 200))
    ensure_task_schema(database)
    with connect(database.database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    task_type,
                    status,
                    query,
                    summary,
                    result->'counts' AS counts,
                    result->>'file_name' AS file_name,
                    result->>'share_id' AS share_id,
                    error,
                    created_at,
                    updated_at
                FROM rag_tasks
                WHERE task_type = 'batch_eval'
                ORDER BY updated_at DESC, id DESC
                LIMIT %s
                """,
                (safe_limit,),
            )
            rows = cur.fetchall()
            share_ids_by_task_id = ensure_batch_eval_share_ids_for_rows(cur, rows)
        conn.commit()
    items: list[dict[str, Any]] = []
    for row in rows:
        task_id = int(row[0])
        counts = row[5] if isinstance(row[5], dict) else {}
        share_id = str(row[7] or share_ids_by_task_id.get(task_id) or "").strip()
        items.append(
            {
                "id": task_id,
                "task_type": row[1],
                "status": row[2],
                "query": row[3],
                "summary": row[4],
                "counts": counts,
                "file_name": row[6],
                "share_id": share_id,
                "error": row[8],
                "created_at": iso_datetime(row[9]),
                "updated_at": iso_datetime(row[10]),
            }
        )
    return items


def batch_eval_share_path(path: str) -> bool:
    """Return whether a GET path should serve the SPA for a shared batch eval."""

    return bool(normalize_batch_eval_share_id(str(path or "").strip("/")))


def normalize_batch_eval_share_id(value: object) -> str:
    """Return a validated batch-evaluation share ID or an empty string."""

    share_id = str(value or "").strip().strip("/")
    if not share_id or share_id.lower().startswith("api") or "/" in share_id or "." in share_id:
        return ""
    if len(share_id) < 6 or len(share_id) > 64:
        return ""
    if not all(ch.isalnum() or ch in {"_", "-"} for ch in share_id):
        return ""
    return share_id


def new_batch_eval_share_id() -> str:
    """Generate a compact random share ID for batch evaluation routes."""

    return uuid.uuid4().hex[:BATCH_EVAL_SHARE_ID_LENGTH]


def generate_unique_batch_eval_share_id(database: DatabaseOptions) -> str:
    """Generate a share ID that is not currently used by a batch-evaluation task."""

    ensure_task_schema(database)
    with connect(database.database_url) as conn:
        with conn.cursor() as cur:
            for _attempt in range(10):
                share_id = new_batch_eval_share_id()
                cur.execute(
                    """
                    SELECT 1
                    FROM rag_tasks
                    WHERE task_type = 'batch_eval' AND result->>'share_id' = %s
                    LIMIT 1
                    """,
                    (share_id,),
                )
                if cur.fetchone() is None:
                    return share_id
    raise RuntimeError("failed to generate unique batch evaluation share_id")


def ensure_batch_eval_share_ids_for_rows(cur: Any, rows: Sequence[Any]) -> dict[int, str]:
    """Backfill share IDs only for the already-selected batch-evaluation rows."""

    missing_task_ids = [int(row[0]) for row in rows if not str(row[7] or "").strip()]
    if not missing_task_ids:
        return {}
    cur.execute(
        """
        SELECT COALESCE(result->>'share_id', '')
        FROM rag_tasks
        WHERE task_type = 'batch_eval'
          AND COALESCE(result->>'share_id', '') <> ''
        """
    )
    used_share_ids = {str(row[0]) for row in cur.fetchall()}
    share_ids_by_task_id: dict[int, str] = {}
    for task_id in missing_task_ids:
        share_id = new_batch_eval_share_id()
        for _attempt in range(10):
            if share_id not in used_share_ids:
                break
            share_id = new_batch_eval_share_id()
        else:
            raise RuntimeError("failed to generate unique batch evaluation share_id")
        used_share_ids.add(share_id)
        share_ids_by_task_id[task_id] = share_id
        cur.execute(
            """
            UPDATE rag_tasks
            SET result = result || jsonb_build_object('share_id', %s)
            WHERE id = %s
            """,
            (share_id, task_id),
        )
    return share_ids_by_task_id


def get_batch_eval_task_by_share_id(database: DatabaseOptions, share_id: str) -> dict[str, Any] | None:
    """Return a batch-evaluation task by its public share ID."""

    share_id = normalize_batch_eval_share_id(share_id)
    if not share_id:
        return None
    ensure_task_schema(database)
    with connect(database.database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id
                FROM rag_tasks
                WHERE task_type = 'batch_eval' AND result->>'share_id' = %s
                LIMIT 1
                """,
                (share_id,),
            )
            row = cur.fetchone()
        conn.commit()
    if row is None:
        return None
    return get_task(database, int(row[0]))


def list_question_tabs(database: DatabaseOptions, *, limit: int = 120) -> list[dict[str, Any]]:
    """Return persisted question tabs reconstructed from search and answer tasks."""

    safe_limit = max(1, min(limit, 300))
    ensure_task_schema(database)
    with connect(database.database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, task_type, status, query, summary, error, created_at, updated_at
                FROM rag_tasks
                WHERE task_type IN ('search', 'answer')
                  AND query IS NOT NULL
                  AND btrim(query) <> ''
                ORDER BY updated_at DESC, id DESC
                LIMIT %s
                """,
                (safe_limit * 4,),
            )
            rows = cur.fetchall()
        conn.commit()
    tasks = [
        {
            "id": int(row[0]),
            "task_type": row[1],
            "status": row[2],
            "query": row[3],
            "summary": row[4],
            "error": row[5],
            "created_at": iso_datetime(row[6]),
            "updated_at": iso_datetime(row[7]),
        }
        for row in rows
    ]
    return build_question_tabs_from_tasks(tasks, limit=safe_limit)


def build_question_tabs_from_tasks(tasks: list[dict[str, Any]], *, limit: int = 120) -> list[dict[str, Any]]:
    """Build shared question-tab metadata from persisted search and answer tasks."""

    tabs_by_key: dict[str, dict[str, Any]] = {}
    for task in tasks:
        task_type = str(task.get("task_type") or "")
        if task_type not in {"search", "answer"}:
            continue
        query = normalize_question_query(task.get("query"))
        if not query:
            continue
        key = normalize_question_key(query)
        tab = tabs_by_key.get(key)
        if tab is None:
            tab = {
                "id": f"q-server-{hashlib.sha1(key.encode('utf-8')).hexdigest()[:16]}",
                "query": query,
                "title": question_tab_title(query),
                "search_task_id": None,
                "answer_task_id": None,
                "status": "persisted",
                "summary": None,
                "created_at": task.get("created_at"),
                "updated_at": task.get("updated_at"),
                "latest_task_id": task.get("id"),
                "latest_task_type": task_type,
            }
            tabs_by_key[key] = tab
        tab["updated_at"] = max_iso_datetime(tab.get("updated_at"), task.get("updated_at"))
        if task_type == "answer" and tab.get("answer_task_id") is None:
            tab["answer_task_id"] = task.get("id")
            tab["status"] = question_status_from_task(task, answered_label="answered")
            tab["summary"] = task.get("summary")
        elif task_type == "search" and tab.get("search_task_id") is None:
            tab["search_task_id"] = task.get("id")
            if tab.get("answer_task_id") is None:
                tab["status"] = question_status_from_task(task, answered_label="searched")
                tab["summary"] = task.get("summary")
        if tab.get("latest_task_id") is None:
            tab["latest_task_id"] = task.get("id")
            tab["latest_task_type"] = task_type
    tabs = sorted(tabs_by_key.values(), key=lambda item: str(item.get("updated_at") or ""), reverse=True)
    return tabs[: max(1, min(limit, 300))]


def normalize_question_query(value: object) -> str:
    """Return a compact display query for question grouping."""

    return " ".join(str(value or "").split())


def normalize_question_key(query: str) -> str:
    """Return the stable grouping key for one diagnostic question."""

    return normalize_question_query(query)


def question_tab_title(query: str) -> str:
    """Return a concise title for a persisted question tab."""

    text = normalize_question_query(query)
    return f"{text[:24]}..." if len(text) > 24 else text or "新问题"


def question_status_from_task(task: dict[str, Any], *, answered_label: str) -> str:
    """Map a stored task status into a question-tab status label."""

    status = str(task.get("status") or "")
    if status == "completed":
        return answered_label
    return status or "persisted"


def batch_eval_summary(result: dict[str, Any]) -> str:
    """Return a compact summary for one stored batch-evaluation result."""

    counts = result.get("counts") if isinstance(result.get("counts"), dict) else {}
    done = int(counts.get("done") or len(result.get("rows") or []))
    total = int(counts.get("total") or result.get("row_count") or done)
    passed = int(counts.get("pass") or 0)
    warned = int(counts.get("warn") or 0)
    failed = int(counts.get("fail") or 0)
    errors = int(counts.get("error") or 0)
    status = str(result.get("status") or "running")
    metric = str(counts.get("selected_metric") or result.get("selected_metric") or "")
    metric_names = {
        "part_recall": "备件召回率",
        "work_order_recall": "工单召回率",
        "answer_part_recall": "回答备件召回率",
    }
    metric_text = f"{metric_names.get(metric)} " if metric in metric_names else ""
    warn_text = f" · 黄色 {warned}" if warned else ""
    return f"{status} · {done}/{total} · {metric_text}正确 {passed}{warn_text} · 失败 {failed} · 错误 {errors}"


def max_iso_datetime(left: object, right: object) -> object:
    """Return the lexicographically latest ISO datetime-like value."""

    if not left:
        return right
    if not right:
        return left
    return right if str(right) > str(left) else left


def get_task(database: DatabaseOptions, task_id: int) -> dict[str, Any] | None:
    """Return one workbench task with its stored request and result payloads."""

    ensure_task_schema(database)
    with connect(database.database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, task_type, status, query, summary, request, result, error, created_at, updated_at
                FROM rag_tasks
                WHERE id = %s
                """,
                (task_id,),
            )
            row = cur.fetchone()
        conn.commit()
    if row is None:
        return None
    return {
        "id": int(row[0]),
        "task_type": row[1],
        "status": row[2],
        "query": row[3],
        "summary": row[4],
        "request": row[5] or {},
        "result": row[6] or {},
        "error": row[7],
        "created_at": iso_datetime(row[8]),
        "updated_at": iso_datetime(row[9]),
    }


def iso_datetime(value: object) -> str | None:
    """Return an ISO string for datetime-like values from PostgreSQL."""

    if value is None:
        return None
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return str(isoformat())
    return str(value)


def doctor_payload() -> dict[str, str]:
    """Return environment details exposed in the web UI."""

    import platform
    import sys

    return {
        "waji_rag_version": __version__,
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "cwd": str(Path.cwd()),
    }


def serve(*, host: str, port: int) -> None:
    """Start the local web debugging server."""

    server = ThreadingHTTPServer((host, port), RagDebugHandler)
    url = f"http://{host}:{port}"
    print(f"Waji RAG Debug UI: {url}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server", flush=True)
    finally:
        server.server_close()
