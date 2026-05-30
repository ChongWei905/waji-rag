"""A local web UI for PostgreSQL-backed RAG debugging."""

from __future__ import annotations

import json
import html
import hashlib
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

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
        <label class="checkline" for="enableQueryParser">
          <input id="enableQueryParser" type="checkbox">
          <span>启用 LLM 问题解析</span>
        </label>
        <div>
          <label for="queryParserProvider">问题解析 Provider</label>
          <select id="queryParserProvider">
            <option value="dashscope">DashScope</option>
            <option value="openai">OpenAI compatible</option>
            <option value="vllm">vLLM / local</option>
          </select>
        </div>
        <div>
          <label for="queryParserModel">问题解析模型</label>
          <input id="queryParserModel" value="qwen3.5-plus">
        </div>
        <div class="full">
          <label for="queryParserBaseUrl">问题解析 Base URL</label>
          <input id="queryParserBaseUrl" value="__DASHSCOPE_BASE_URL__">
        </div>
        <div class="full">
          <label for="queryParserNoProxyHosts">问题解析 No Proxy Hosts</label>
          <input id="queryParserNoProxyHosts" value="localhost,127.0.0.1,127.0.0.0/8,::1" placeholder="逗号分隔，支持 IP、CIDR、*.domain">
        </div>
        <div class="full">
          <label for="queryParserApiKey">问题解析 API Key</label>
          <input id="queryParserApiKey" type="password" placeholder="可留空，留空时读取 Env 或配置文件">
        </div>

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
    .page-shell.history-collapsed .question-sidebar {
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
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .query-tools .query-actions {
      grid-column: 1 / -1;
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
    .workspace {
      margin-top: 14px;
      display: grid;
      grid-template-columns: minmax(220px, 260px) minmax(0, 1fr);
      gap: 14px;
      align-items: start;
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
      overflow: auto;
      border-radius: 8px;
      border: 1px solid var(--line);
      background: #fff;
      box-shadow: 0 22px 60px rgba(15, 23, 42, .24);
    }
    .history-modal {
      width: min(760px, 100%);
    }
    .modal-head, .modal-foot {
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    .modal-foot {
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
    .modal-body .full {
      grid-column: 1 / -1;
    }
    .history-body {
      padding: 16px;
      display: grid;
      gap: 12px;
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
      .query-tools, .retrieval-board, .modal-body { grid-template-columns: 1fr; }
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
      <button id="openQuestionSidebarHeaderBtn" class="secondary">回答历史</button>
      <button id="openHistoryBtn" class="secondary">历史任务</button>
      <button id="openConfigBtn" class="secondary">配置</button>
      <button id="doctorBtn" class="ghost">环境检查</button>
    </div>
  </header>

  <div id="pageShell" class="page-shell">
    <aside id="questionSidebar" class="question-sidebar">
      <div class="question-sidebar-head">
        <h2>历史回答</h2>
        <button id="closeQuestionSidebarBtn" class="ghost">收起</button>
      </div>
      <div id="questionTabs" class="question-tabs"></div>
      <button id="newQuestionBtn" class="secondary">新问题</button>
    </aside>

    <main>
    <div class="view-tabs">
      <button id="buildViewBtn" class="view-tab active">索引构建</button>
      <button id="qaViewBtn" class="view-tab">检索与回答</button>
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
            <div>
              <label for="topK">每路 Top K</label>
              <input id="topK" type="number" min="1" value="1">
            </div>
            <div>
              <label for="evidenceTopK">答案证据数</label>
              <input id="evidenceTopK" type="number" min="1" value="4">
            </div>
            <div class="query-actions">
              <button id="runQuestionSearchBtn" class="secondary">检索当前问题</button>
              <button id="runQuestionAnswerBtn">回答当前问题</button>
            </div>
          </div>
        </section>
      </div>
    </section>

    <div class="workspace">
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
        <div class="full">
          <label for="databaseUrl">Database URL</label>
          <input id="databaseUrl" value="postgresql://waji:waji@127.0.0.1:55432/waji_rag">
        </div>
        <div class="full">
          <label for="envFile">Env 文件</label>
          <input id="envFile" placeholder="__DOCARBOR_ENV_PATH__">
        </div>
        <div>
          <label for="workOrderDir">工单 TXT 目录</label>
          <input id="workOrderDir" value="__DEMO_WORK_ORDER_DIR_TEXT__">
        </div>
        <div>
          <label for="manualDir">手册 HTML/MD 目录</label>
          <input id="manualDir" value="__DEMO_MANUAL_DIR_TEXT__">
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
        <div>
          <label for="embeddingProvider">Embedding Provider</label>
          <select id="embeddingProvider">
            <option value="vllm">vLLM / local</option>
            <option value="openai">OpenAI compatible</option>
            <option value="dashscope">DashScope</option>
          </select>
        </div>
        <div>
          <label for="embeddingModel">Embedding 模型</label>
          <input id="embeddingModel" placeholder="vLLM 可留空；云模型填写模型名">
        </div>
        <div>
          <label for="embeddingDimensions">向量维度</label>
          <input id="embeddingDimensions" type="number" min="0" placeholder="留空则不发送 dimensions">
        </div>
        <div>
          <label for="embeddingBatchSize">批量大小</label>
          <input id="embeddingBatchSize" type="number" min="1" value="10">
        </div>
        <div class="full">
          <label for="embeddingBaseUrl">Embedding Base URL</label>
          <input id="embeddingBaseUrl" value="http://127.0.0.1:8888/v1">
        </div>
        <div class="full">
          <label for="embeddingNoProxyHosts">Embedding No Proxy Hosts</label>
          <input id="embeddingNoProxyHosts" value="localhost,127.0.0.1,127.0.0.0/8,::1" placeholder="逗号分隔，支持 IP、CIDR、*.domain">
        </div>
        <div class="full">
          <label for="embeddingApiKey">Embedding API Key</label>
          <input id="embeddingApiKey" type="password" placeholder="本地 vLLM 可留空；云服务填写 key">
        </div>

        <label class="checkline" for="enableRerank">
          <input id="enableRerank" type="checkbox">
          <span>启用 rerank</span>
        </label>
        <div>
          <label for="rerankModel">Rerank 模型</label>
          <input id="rerankModel" value="qwen3-rerank">
        </div>
        <div class="full">
          <label for="rerankBaseUrl">Rerank Base URL</label>
          <input id="rerankBaseUrl" value="__DASHSCOPE_RERANK_BASE_URL__">
        </div>
        <div class="full">
          <label for="rerankNoProxyHosts">Rerank No Proxy Hosts</label>
          <input id="rerankNoProxyHosts" value="localhost,127.0.0.1,127.0.0.0/8,::1" placeholder="逗号分隔，支持 IP、CIDR、*.domain">
        </div>
        <div class="full">
          <label for="rerankApiKey">Rerank API Key</label>
          <input id="rerankApiKey" type="password" placeholder="可留空，留空时读取 DOCARBOR_RERANK_API_KEY 或兜底 key">
        </div>

        <label class="checkline" for="enableQueryParser">
          <input id="enableQueryParser" type="checkbox">
          <span>启用 LLM 问题解析</span>
        </label>
        <div>
          <label for="queryParserProvider">问题解析 Provider</label>
          <select id="queryParserProvider">
            <option value="dashscope">DashScope</option>
            <option value="openai">OpenAI compatible</option>
            <option value="vllm">vLLM / local</option>
          </select>
        </div>
        <div>
          <label for="queryParserModel">问题解析模型</label>
          <input id="queryParserModel" value="qwen3.5-plus">
        </div>
        <div class="full">
          <label for="queryParserBaseUrl">问题解析 Base URL</label>
          <input id="queryParserBaseUrl" value="__DASHSCOPE_BASE_URL__">
        </div>
        <div class="full">
          <label for="queryParserNoProxyHosts">问题解析 No Proxy Hosts</label>
          <input id="queryParserNoProxyHosts" value="localhost,127.0.0.1,127.0.0.0/8,::1" placeholder="逗号分隔，支持 IP、CIDR、*.domain">
        </div>
        <div class="full">
          <label for="queryParserApiKey">问题解析 API Key</label>
          <input id="queryParserApiKey" type="password" placeholder="可留空，留空时读取 Env 或配置文件">
        </div>

        <label class="checkline" for="enableLlm">
          <input id="enableLlm" type="checkbox">
          <span>启用 LLM 答案生成</span>
        </label>
        <div>
          <label for="llmProvider">LLM Provider</label>
          <select id="llmProvider">
            <option value="dashscope">DashScope</option>
            <option value="openai">OpenAI compatible</option>
            <option value="vllm">vLLM / local</option>
          </select>
        </div>
        <div>
          <label for="llmModel">LLM 模型</label>
          <input id="llmModel" value="qwen3.5-plus">
        </div>
        <div class="full">
          <label for="llmBaseUrl">LLM Base URL</label>
          <input id="llmBaseUrl" value="__DASHSCOPE_BASE_URL__">
        </div>
        <div class="full">
          <label for="llmNoProxyHosts">LLM No Proxy Hosts</label>
          <input id="llmNoProxyHosts" value="localhost,127.0.0.1,127.0.0.0/8,::1" placeholder="逗号分隔，支持 IP、CIDR、*.domain">
        </div>
        <div class="full">
          <label for="llmApiKey">LLM API Key</label>
          <input id="llmApiKey" type="password" placeholder="可留空，留空时读取 Env 或配置文件">
        </div>
        <label class="checkline full" for="apiRequestLoggingEnabled">
          <input id="apiRequestLoggingEnabled" type="checkbox" checked>
          <span>记录 Embedding / LLM 请求日志</span>
        </label>
        <div class="full">
          <label for="apiRequestLogPath">请求日志文件</label>
          <input id="apiRequestLogPath" value="__MODEL_API_REQUEST_LOG_PATH__">
        </div>
        <div class="full">
          <label for="configImportFile">配置文件导入</label>
          <input id="configImportFile" type="file" accept="application/json,.json">
        </div>
      </div>
      <div class="modal-foot">
        <button id="loadDemoBtn" class="secondary">加载 Demo 配置</button>
        <button id="docArborEnvBtn" class="secondary">填入 DocArbor Env</button>
        <button id="exportConfigBtn" class="secondary">导出配置</button>
        <button id="importConfigBtn" class="secondary">导入配置</button>
        <button id="previewConfigBtn" class="secondary">预览配置</button>
        <button id="saveConfigBtn">保存配置</button>
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
    const channels = [
      ["work_orders", "历史工单", "doc_type=work_order；字段权重优先 reported_issue，再看 solution/raw_text。"],
      ["manual_typical_faults", "典型故障手册", "doc_type=manual_typical_fault；优先 fault_title/file_name，再看正文 chunk。"],
      ["manual_fault_codes", "故障码手册", "问题出现故障码时先精确匹配 fault_code，未出现故障码则为空。"],
      ["part_evidence", "备件证据", "doc_type=part_evidence；从命中工单的备件字段抽取，不从问题猜备件。"]
    ];
    const buildStageOrder = [
      ["config", "配置解析", "读取页面配置、env 和模型开关"],
      ["init", "初始化 PG", "创建 PostgreSQL / pgvector 表结构"],
      ["ingest", "构建索引", "解析工单、HTML 转 Markdown、入库、建 BM25/向量"],
      ["embedding", "补Embedding", "扫描已有索引文档，为缺失向量的文档补齐 embedding"]
    ];
    const qaStageOrder = [
      ["retrieval", "多路召回", "历史工单、手册、故障码、备件证据分路召回"],
      ["evidence_filter", "证据过滤", "按部件锚点过滤同症状但错部件的证据"],
      ["rerank", "重排", "可选 rerank；失败或关闭则保留原顺序"],
      ["answer", "答案生成", "用选中证据和备件候选生成最终答复"]
    ];
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
      buildPollTimer: null
    };

    function stageOrderForView(view = appState.activeView) {
      return view === "qa" ? qaStageOrder : buildStageOrder;
    }

    function initialStageForView(view = appState.activeView) {
      return view === "qa" ? "retrieval" : "config";
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
        $("answer").textContent = "已完成检索。请查看“多路召回”和“证据过滤”。";
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

    function configOverrides() {
      return {
        embedding: {
          enabled: $("enableEmbedding").checked,
          provider: $("embeddingProvider").value,
          model: $("embeddingModel").value.trim(),
          base_url: $("embeddingBaseUrl").value.trim(),
          api_key: $("embeddingApiKey").value.trim(),
          dimensions: $("embeddingDimensions").value ? Number($("embeddingDimensions").value) : null,
          batch_size: $("embeddingBatchSize").value ? Number($("embeddingBatchSize").value) : 10,
          no_proxy_hosts: parseCsv($("embeddingNoProxyHosts").value),
          log_requests_enabled: $("apiRequestLoggingEnabled").checked,
          request_log_path: $("apiRequestLogPath").value.trim() || defaultModelApiRequestLogPath
        },
        rerank: {
          enabled: $("enableRerank").checked,
          provider: "dashscope",
          model: $("rerankModel").value.trim(),
          base_url: $("rerankBaseUrl").value.trim(),
          api_key: $("rerankApiKey").value.trim(),
          no_proxy_hosts: parseCsv($("rerankNoProxyHosts").value),
          top_n: $("evidenceTopK").value ? Number($("evidenceTopK").value) : 8
        },
        query_parser: {
          enabled: $("enableQueryParser").checked,
          provider: $("queryParserProvider").value,
          model: $("queryParserModel").value.trim(),
          base_url: $("queryParserBaseUrl").value.trim(),
          api_key: $("queryParserApiKey").value.trim(),
          no_proxy_hosts: parseCsv($("queryParserNoProxyHosts").value),
          max_tokens: 700,
          temperature: 0,
          log_requests_enabled: $("apiRequestLoggingEnabled").checked,
          request_log_path: $("apiRequestLogPath").value.trim() || defaultModelApiRequestLogPath
        },
        llm: {
          enabled: $("enableLlm").checked,
          provider: $("llmProvider").value,
          model: $("llmModel").value.trim(),
          base_url: $("llmBaseUrl").value.trim(),
          api_key: $("llmApiKey").value.trim(),
          no_proxy_hosts: parseCsv($("llmNoProxyHosts").value),
          max_tokens: 1400,
          temperature: 0,
          log_requests_enabled: $("apiRequestLoggingEnabled").checked,
          request_log_path: $("apiRequestLogPath").value.trim() || defaultModelApiRequestLogPath
        },
        answer: {
          enabled: true,
          evidence_top_k: $("evidenceTopK").value ? Number($("evidenceTopK").value) : 8,
          include_debug: true
        }
      };
    }

    function commonPayload() {
      return {
        database_url: $("databaseUrl").value.trim() || null,
        env_file: $("envFile").value.trim() || null,
        config_overrides: configOverrides()
      };
    }

    function taskPayload(extra = {}) {
      return {
        database_url: $("databaseUrl").value.trim() || null,
        ...extra
      };
    }

    function parseCsv(value) {
      return String(value || "").split(",").map(item => item.trim()).filter(Boolean);
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
          evidence_top_k: $("evidenceTopK").value
        },
        database_url: $("databaseUrl").value,
        env_file: $("envFile").value,
        work_order_dir: $("workOrderDir").value,
        manual_dir: $("manualDir").value,
        work_order_limit: $("workOrderLimit").value,
        manual_limit: $("manualLimit").value,
        max_manual_chars: $("maxManualChars").value,
        ingest_reset: $("ingestReset").checked,
        ingest_resume: $("ingestResume").checked,
        model_api_log: {
          enabled: $("apiRequestLoggingEnabled").checked,
          path: $("apiRequestLogPath").value
        },
        embedding: {
          enabled: $("enableEmbedding").checked,
          provider: $("embeddingProvider").value,
          model: $("embeddingModel").value,
          dimensions: $("embeddingDimensions").value,
          batch_size: $("embeddingBatchSize").value,
          base_url: $("embeddingBaseUrl").value,
          no_proxy_hosts: $("embeddingNoProxyHosts").value,
          api_key: $("embeddingApiKey").value
        },
        rerank: {
          enabled: $("enableRerank").checked,
          model: $("rerankModel").value,
          base_url: $("rerankBaseUrl").value,
          no_proxy_hosts: $("rerankNoProxyHosts").value,
          api_key: $("rerankApiKey").value
        },
        query_parser: {
          enabled: $("enableQueryParser").checked,
          provider: $("queryParserProvider").value,
          model: $("queryParserModel").value,
          base_url: $("queryParserBaseUrl").value,
          no_proxy_hosts: $("queryParserNoProxyHosts").value,
          api_key: $("queryParserApiKey").value
        },
        llm: {
          enabled: $("enableLlm").checked,
          provider: $("llmProvider").value,
          model: $("llmModel").value,
          base_url: $("llmBaseUrl").value,
          no_proxy_hosts: $("llmNoProxyHosts").value,
          api_key: $("llmApiKey").value
        }
      };
    }

    function applyConfigSnapshot(config, options = {}) {
      if (!config || typeof config !== "object") throw new Error("配置文件格式不正确");
      const ui = config.ui || {};
      setInputValue("databaseUrl", config.database_url);
      setInputValue("envFile", config.env_file);
      setInputValue("workOrderDir", config.work_order_dir);
      setInputValue("manualDir", config.manual_dir);
      setInputValue("workOrderLimit", config.work_order_limit);
      setInputValue("manualLimit", config.manual_limit);
      setInputValue("maxManualChars", config.max_manual_chars);
      setCheckboxValue("ingestReset", config.ingest_reset);
      setCheckboxValue("ingestResume", config.ingest_resume);
      const modelApiLog = config.model_api_log || {};
      setCheckboxValue("apiRequestLoggingEnabled", modelApiLog.enabled);
      setInputValue("apiRequestLogPath", modelApiLog.path);
      setInputValue("query", ui.query);
      setInputValue("topK", ui.top_k);
      setInputValue("evidenceTopK", ui.evidence_top_k);
      if (ui.question_sidebar_open !== undefined) {
        setQuestionSidebar(Boolean(ui.question_sidebar_open), {save: false});
      }

      const embedding = config.embedding || {};
      if (modelApiLog.enabled === undefined) setCheckboxValue("apiRequestLoggingEnabled", embedding.log_requests_enabled);
      if (modelApiLog.path === undefined) setInputValue("apiRequestLogPath", embedding.request_log_path);
      setCheckboxValue("enableEmbedding", embedding.enabled);
      setInputValue("embeddingProvider", embedding.provider);
      setInputValue("embeddingModel", embedding.model);
      setInputValue("embeddingDimensions", embedding.dimensions);
      setInputValue("embeddingBatchSize", embedding.batch_size);
      setInputValue("embeddingBaseUrl", embedding.base_url);
      setInputValue("embeddingNoProxyHosts", embedding.no_proxy_hosts);
      setInputValue("embeddingApiKey", embedding.api_key);

      const rerank = config.rerank || {};
      setCheckboxValue("enableRerank", rerank.enabled);
      setInputValue("rerankModel", rerank.model);
      setInputValue("rerankBaseUrl", rerank.base_url);
      setInputValue("rerankNoProxyHosts", rerank.no_proxy_hosts);
      setInputValue("rerankApiKey", rerank.api_key);

      const queryParser = config.query_parser || {};
      setCheckboxValue("enableQueryParser", queryParser.enabled);
      setInputValue("queryParserProvider", queryParser.provider);
      setInputValue("queryParserModel", queryParser.model);
      setInputValue("queryParserBaseUrl", queryParser.base_url);
      setInputValue("queryParserNoProxyHosts", queryParser.no_proxy_hosts);
      setInputValue("queryParserApiKey", queryParser.api_key);

      const llm = config.llm || {};
      if (modelApiLog.enabled === undefined && embedding.log_requests_enabled === undefined) {
        setCheckboxValue("apiRequestLoggingEnabled", llm.log_requests_enabled);
      }
      if (modelApiLog.path === undefined && embedding.request_log_path === undefined) {
        setInputValue("apiRequestLogPath", llm.request_log_path);
      }
      setCheckboxValue("enableLlm", llm.enabled);
      setInputValue("llmProvider", llm.provider);
      setInputValue("llmModel", llm.model);
      setInputValue("llmBaseUrl", llm.base_url);
      setInputValue("llmNoProxyHosts", llm.no_proxy_hosts);
      setInputValue("llmApiKey", llm.api_key);

      if (ui.active_view) switchView(ui.active_view === "qa" ? "qa" : "build");
      if (!options.silent) setStatus("配置已导入", "success");
    }

    function setInputValue(id, value) {
      if (value === undefined || value === null) return;
      $(id).value = String(value);
    }

    function setCheckboxValue(id, value) {
      if (value === undefined || value === null) return;
      $(id).checked = Boolean(value);
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
      setStatus("配置已导出", "success");
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
    }

    function bindAutoSave() {
      const ids = [
        "databaseUrl", "envFile", "workOrderDir", "manualDir", "workOrderLimit", "manualLimit", "maxManualChars",
        "apiRequestLoggingEnabled", "apiRequestLogPath",
        "ingestReset", "ingestResume", "query", "topK", "evidenceTopK", "enableEmbedding", "embeddingProvider", "embeddingModel",
        "embeddingDimensions", "embeddingBatchSize", "embeddingBaseUrl", "embeddingNoProxyHosts", "embeddingApiKey",
        "enableRerank", "rerankModel", "rerankBaseUrl", "rerankNoProxyHosts", "rerankApiKey",
        "enableQueryParser", "queryParserProvider", "queryParserModel", "queryParserBaseUrl", "queryParserNoProxyHosts", "queryParserApiKey",
        "enableLlm",
        "llmProvider", "llmModel", "llmBaseUrl", "llmNoProxyHosts", "llmApiKey"
      ];
      for (const id of ids) {
        const element = $(id);
        if (!element) continue;
        element.addEventListener("change", saveConfigToLocalStorage);
        element.addEventListener("input", saveConfigToLocalStorage);
      }
    }

    function ingestPayload() {
      return {
        ...commonPayload(),
        work_order_dir: $("workOrderDir").value.trim() || null,
        manual_dir: $("manualDir").value.trim() || null,
        reset: $("ingestReset").checked,
        work_order_limit: $("workOrderLimit").value ? Number($("workOrderLimit").value) : null,
        manual_limit: $("manualLimit").value ? Number($("manualLimit").value) : null,
        max_manual_chars: $("maxManualChars").value ? Number($("maxManualChars").value) : 1800
      };
    }

    function queryPayload(query = $("query").value) {
      return {
        ...commonPayload(),
        query: String(query || "").trim(),
        top_k: $("topK").value ? Number($("topK").value) : 5,
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
      if (!response.ok) throw new Error(data.error || response.statusText);
      return data;
    }

    async function getJson(url) {
      const response = await fetch(url);
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || response.statusText);
      return data;
    }

    function setStatus(text, kind = "") {
      $("status").className = "status" + (kind ? " " + kind : "");
      $("status").textContent = text;
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
      } else {
        const buildState = hydrateStageState("build", appState.buildStages, appState.buildSelectedStage || "config");
        appState.buildStages = buildState.stages;
        appState.buildSelectedStage = buildState.selectedStage;
        appState.stages = buildState.stages;
        appState.selectedStage = buildState.selectedStage;
      }
      $("buildView").classList.toggle("active", view === "build");
      $("qaView").classList.toggle("active", view === "qa");
      $("buildViewBtn").classList.toggle("active", view === "build");
      $("qaViewBtn").classList.toggle("active", view === "qa");
      renderTaskList();
      renderStages();
      renderStageInspector();
      saveConfigToLocalStorage();
    }

    function setStage(id, status, data = null, summary = "") {
      if (!stageIdSetForView(appState.activeView).has(id)) return;
      appState.stages[id] = {status, data, summary};
      appState.selectedStage = id;
      if (appState.activeView === "build") {
        appState.buildStages = appState.stages;
        appState.buildSelectedStage = appState.selectedStage;
      }
      renderStages();
      renderStageInspector();
      syncActiveQuestionState();
    }

    function resetStages(view = appState.activeView, selectedStage = null) {
      const state = createStageState(view, selectedStage);
      appState.stages = state.stages;
      appState.selectedStage = state.selectedStage;
      if (view === "build") {
        appState.buildStages = state.stages;
        appState.buildSelectedStage = state.selectedStage;
      } else {
        syncActiveQuestionState();
      }
      renderStages();
      renderStageInspector();
    }

    function renderStages() {
      $("stageListTitle").textContent = appState.activeView === "qa" ? "回答阶段" : "构建阶段";
      $("stageList").innerHTML = "";
      for (const [id, title, note] of stageOrderForView(appState.activeView)) {
        const state = appState.stages[id] || {status: "pending", summary: ""};
        const button = document.createElement("button");
        button.className = `stage-node ${state.status || "pending"} ${appState.selectedStage === id ? "active" : ""}`;
        button.innerHTML = `
          <div class="stage-title">
            <span>${escapeHtml(title)}</span>
            <span class="pill ${state.status === "done" ? "ok" : state.status === "fallback" || state.status === "skipped" || state.status === "filtered" ? "warn" : ""}">${escapeHtml(state.status || "pending")}</span>
          </div>
          <div class="stage-note">${escapeHtml(state.summary || note)}</div>
        `;
        button.addEventListener("click", () => {
          appState.selectedStage = id;
          if (appState.activeView === "build") appState.buildSelectedStage = id;
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
      $("buildViewBtn").classList.toggle("active", appState.activeView === "build");
      $("qaViewBtn").classList.toggle("active", appState.activeView === "qa");
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
      const showAnswer = appState.activeView === "qa" && stageId === "answer";
      const showRetrieval = appState.activeView === "qa" && stageId === "retrieval";
      const showEvidence = appState.activeView === "qa" && ["evidence_filter", "rerank"].includes(stageId);
      const showInspector = !showAnswer && !showRetrieval && !showEvidence;
      $("answerPanel").classList.toggle("hidden", !showAnswer);
      $("retrievalPanel").classList.toggle("hidden", !showRetrieval);
      $("evidencePanel").classList.toggle("hidden", !showEvidence);
      $("inspectorPanel").classList.toggle("hidden", !showInspector);
      if (showEvidence) {
        if (stageId === "evidence_filter") {
          renderEvidenceFilter((appState.lastResult && appState.lastResult.evidence_filter) || {});
        } else {
          renderSelectedEvidence((appState.lastResult && appState.lastResult.selected_evidence) || []);
        }
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
        工单目录：${escapeHtml(report.work_order_dir || $("workOrderDir").value || "未配置")}<br>
        手册目录：${escapeHtml(report.manual_dir || $("manualDir").value || "未配置")}<br>
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
            <span class="pill ${task.status === "completed" ? "ok" : task.status === "failed" || task.status === "completed_with_errors" ? "warn" : ""}">${escapeHtml(task.status || "")}</span>
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

    function taskTypeLabel(taskType) {
      return {
        build: "构建",
        build_retry: "失败重试",
        embedding: "补Embedding",
        search: "检索",
        answer: "回答"
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
      resetStages("qa", "retrieval");
      const retrieval = result.retrieval || result;
      const answer = result.answer || {};
      setStageFromTrace(result);
      renderAnswer(answer);
      renderParts(result.part_candidates || retrieval.part_candidates || []);
      renderRetrievalBoard(retrieval);
      renderSelectedEvidence(result.selected_evidence || []);
      if (result.retrieval) setStage("retrieval", "done", result.retrieval, formatRetrievalSummary(result.retrieval));
      if (result.evidence_filter || retrieval.evidence_filter) {
        const evidenceFilter = result.evidence_filter || retrieval.evidence_filter;
        setStage("evidence_filter", evidenceFilter.status || "done", evidenceFilter, evidenceFilterSummary(evidenceFilter));
      }
      if (result.rerank) setStage("rerank", result.rerank.status || "done", result.rerank, rerankSummary(result.rerank));
      if (result.answer) setStage("answer", result.answer.status || "done", result.answer, answerSummary(result.answer));
    }

    function setStageFromTrace(result) {
      if (!Array.isArray(result.trace)) return;
      for (const item of result.trace) {
        if (!item || !item.stage) continue;
        setStage(item.stage, normalizeStatus(item.status), item, JSON.stringify(item.details || {}).slice(0, 120));
      }
    }

    function renderSearchResult(response) {
      const result = response.result || response;
      appState.lastResult = result;
      resetStages("qa", "retrieval");
      setStage("retrieval", "done", result, formatRetrievalSummary(result));
      if (result.evidence_filter) {
        setStage("evidence_filter", result.evidence_filter.status || "done", result.evidence_filter, evidenceFilterSummary(result.evidence_filter));
      }
      renderRetrievalBoard(result);
      renderParts(result.part_candidates || []);
      $("answer").textContent = "已完成检索。请查看“多路召回”和“阶段返回”。";
      renderSelectedEvidence([]);
    }

    function renderAnswer(answer) {
      $("answer").textContent = answer.text || "尚未生成答案。";
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
        return `关联工单全量备件 · work_orders=${count}`;
      }
      return `mode=${retrieval.mode || ""} · top_k=${retrieval.top_k || ""}`;
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

    function renderEvidenceFilter(filterPayload) {
      $("evidencePanelTitle").textContent = "证据过滤";
      const accepted = Array.isArray(filterPayload.accepted) ? filterPayload.accepted : [];
      const rejected = Array.isArray(filterPayload.rejected) ? filterPayload.rejected : [];
      const constraints = filterPayload.constraints || {};
      const renderGateItem = (item, index, decision) => {
        if (isPartEvidence(item)) return renderPartEvidenceCard(item, index);
        const gate = item.evidence_gate || {};
        const reason = gate.reason || decision;
        const componentHits = (gate.component_hits || []).join(", ");
        const symptomHits = (gate.symptom_hits || []).join(", ");
        return `
          <div class="evidence-row">
            <div class="row-title">#${index + 1} ${escapeHtml(decision)} · ${escapeHtml(item.channel || "")} · ${escapeHtml(item.title || "")}</div>
            <div class="row-meta">reason=${escapeHtml(reason)} · doc_id=${escapeHtml(item.doc_id || "")}</div>
            <div class="row-meta">component=${escapeHtml(componentHits || "-")} · symptom=${escapeHtml(symptomHits || "-")}</div>
            <div class="hit-preview">${escapeHtml(item.body_preview || "")}</div>
          </div>
        `;
      };
      $("selectedEvidence").innerHTML = `
        <div class="evidence-row">
          <div class="row-title">${escapeHtml(filterPayload.summary || "证据过滤结果")}</div>
          <div class="row-meta">故障短语：${escapeHtml(constraints.fault_phrase || "")}</div>
          <div class="row-meta">部件锚点：${escapeHtml((constraints.component_terms || []).join(", ") || "-")}</div>
          <div class="row-meta">异常词：${escapeHtml((constraints.symptom_terms || []).join(", ") || "-")}</div>
        </div>
        <h3>Accepted ${accepted.length}</h3>
        ${accepted.length ? accepted.map((item, index) => renderGateItem(item, index, "accepted")).join("") : '<div class="empty">暂无接受证据</div>'}
        <h3>Rejected ${rejected.length}</h3>
        ${rejected.length ? rejected.map((item, index) => renderGateItem(item, index, "rejected")).join("") : '<div class="empty">暂无丢弃证据</div>'}
      `;
    }

    function formatRetrievalSummary(retrieval) {
      const channelsPayload = retrieval.channels || {};
      return `mode=${retrieval.mode || ""} · ` + channels.map(([name, label]) => `${label}:${(channelsPayload[name] || []).length}`).join(" · ");
    }

    function evidenceFilterSummary(filterPayload) {
      if (!filterPayload) return "证据过滤未运行";
      const accepted = Array.isArray(filterPayload.accepted) ? filterPayload.accepted.length : 0;
      const rejected = Array.isArray(filterPayload.rejected) ? filterPayload.rejected.length : 0;
      return `${filterPayload.status || "done"} · accepted=${accepted} · rejected=${rejected}`;
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
          database_url: $("databaseUrl").value.trim() || null,
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
          database_url: $("databaseUrl").value.trim() || null,
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
        const askResult = await postJson("/api/ask-db", payload);
        appState.currentTaskId = askResult.task_id || appState.currentTaskId;
        questionTab.answerTaskId = askResult.task_id || questionTab.answerTaskId;
        questionTab.status = "answered";
        askResult.workflow = {config: preview, init: initResult, ingest: ingestResult};
        renderPipelineResult(askResult);
        syncActiveQuestionState();
        await refreshTasks({quiet: true});
        await refreshQuestionTabsFromServer({quiet: true});
        setStatus("全流程完成", "success");
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
        const result = await postJson("/api/ask-db", payload);
        appState.currentTaskId = result.task_id || null;
        questionTab.answerTaskId = result.task_id || questionTab.answerTaskId;
        questionTab.status = "answered";
        renderPipelineResult(result);
        syncActiveQuestionState();
        await refreshTasks({quiet: true});
        await refreshQuestionTabsFromServer({quiet: true});
        setStatus("问答完成", "success");
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

    function applyEmbeddingProviderDefaults(force = false) {
      const provider = $("embeddingProvider").value;
      const baseInput = $("embeddingBaseUrl");
      const modelInput = $("embeddingModel");
      const dimensionInput = $("embeddingDimensions");
      const shouldReplaceBase = force || !baseInput.value || [dashscopeBaseUrl, openaiBaseUrl, localEmbeddingBaseUrl].includes(baseInput.value.trim());
      if (provider === "vllm") {
        if (shouldReplaceBase) baseInput.value = localEmbeddingBaseUrl;
        if (force) {
          modelInput.value = "";
          dimensionInput.value = "";
          $("embeddingApiKey").value = "";
        }
      } else if (provider === "openai") {
        if (shouldReplaceBase) baseInput.value = openaiBaseUrl;
        if (force) dimensionInput.value = "";
      } else if (provider === "dashscope") {
        if (shouldReplaceBase) baseInput.value = dashscopeBaseUrl;
        if (force && !modelInput.value.trim()) modelInput.value = "text-embedding-v4";
        if (force && !dimensionInput.value.trim()) dimensionInput.value = "1024";
      }
    }

    function applyDemoDefaults(options = {}) {
      $("workOrderDir").value = demoWorkOrderDir;
      $("manualDir").value = demoManualDir;
      $("query").value = defaultQuery;
      $("topK").value = "1";
      $("evidenceTopK").value = "4";
      $("workOrderLimit").value = "";
      $("manualLimit").value = "";
      $("ingestReset").checked = true;
      $("ingestResume").checked = true;
      $("envFile").value = "";
      $("enableEmbedding").checked = false;
      $("embeddingProvider").value = "vllm";
      $("embeddingModel").value = "";
      $("embeddingDimensions").value = "";
      $("embeddingBaseUrl").value = localEmbeddingBaseUrl;
      $("embeddingNoProxyHosts").value = "localhost,127.0.0.1,127.0.0.0/8,::1";
      $("enableRerank").checked = false;
      $("enableLlm").checked = false;
      $("llmProvider").value = "dashscope";
      $("embeddingApiKey").value = "";
      $("rerankNoProxyHosts").value = "localhost,127.0.0.1,127.0.0.0/8,::1";
      $("llmNoProxyHosts").value = "localhost,127.0.0.1,127.0.0.0/8,::1";
      $("rerankApiKey").value = "";
      $("enableQueryParser").checked = false;
      $("queryParserProvider").value = "dashscope";
      $("queryParserModel").value = "qwen3.5-plus";
      $("queryParserBaseUrl").value = dashscopeBaseUrl;
      $("queryParserNoProxyHosts").value = "localhost,127.0.0.1,127.0.0.0/8,::1";
      $("queryParserApiKey").value = "";
      $("llmApiKey").value = "";
      $("apiRequestLoggingEnabled").checked = true;
      $("apiRequestLogPath").value = defaultModelApiRequestLogPath;
      if (options.save !== false) saveConfigToLocalStorage();
      setStatus("已加载 Demo 配置", "success");
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
    $("openHistoryBtn").addEventListener("click", async () => {
      $("taskHistoryModal").classList.add("open");
      await refreshTasks({quiet: true});
    });
    $("closeHistoryBtn").addEventListener("click", () => $("taskHistoryModal").classList.remove("open"));
    $("saveConfigBtn").addEventListener("click", () => {
      saveConfigToLocalStorage();
      setStatus("配置已保存到页面状态", "success");
    });
    $("loadDemoBtn").addEventListener("click", applyDemoDefaults);
    $("docArborEnvBtn").addEventListener("click", () => {
      $("envFile").value = docArborEnvPath;
      setStatus("已填入 DocArbor Env 路径", "success");
    });
    $("embeddingProvider").addEventListener("change", () => applyEmbeddingProviderDefaults(true));
    $("exportConfigBtn").addEventListener("click", exportConfig);
    $("importConfigBtn").addEventListener("click", () => importConfig().catch(error => setStatus(String(error), "error")));
    $("buildViewBtn").addEventListener("click", () => switchView("build"));
    $("qaViewBtn").addEventListener("click", () => switchView("qa"));
    $("openQuestionSidebarHeaderBtn").addEventListener("click", () => setQuestionSidebar(true));
    $("openQuestionSidebarBtn").addEventListener("click", () => setQuestionSidebar(true));
    $("closeQuestionSidebarBtn").addEventListener("click", () => setQuestionSidebar(false));
    $("newQuestionBtn").addEventListener("click", createNewQuestionTab);
    $("query").addEventListener("input", syncActiveQuestionInput);
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

    getJson("/api/doctor").then(data => {
      $("version").textContent = data.waji_rag_version + " · " + data.platform;
    }).catch(() => {});
    resetStages("build", "config");
    applyDemoDefaults({save: false});
    bindAutoSave();
    const restoredConfig = restoreConfigFromLocalStorage();
    if (!appState.questionTabs.length) {
      const tab = makeQuestionTab($("query").value || defaultQuery);
      appState.questionTabs.push(tab);
      appState.activeQuestionTabId = tab.id;
    }
    renderQuestionTabs();
    updateCurrentQuestionTitle();
    setQuestionSidebar(appState.questionSidebarOpen, {save: false});
    renderBuildProgress({});
    if (!restoredConfig) switchView("build");
    refreshTasks({quiet: true});
    refreshQuestionTabsFromServer({quiet: true});
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
        if parsed.path == "/api/doctor":
            self._send_json(doctor_payload())
            return
        self.send_error(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self) -> None:
        """Handle local PG-backed debug actions."""

        parsed = urlparse(self.path)
        if parsed.path == "/api/config-preview":
            self._handle_config_preview()
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
        self.send_error(HTTPStatus.NOT_FOUND, "not found")

    def log_message(self, format: str, *args: object) -> None:
        """Write compact request logs to stderr."""

        super().log_message(format, *args)

    def _handle_config_preview(self) -> None:
        payload = self._read_json()
        try:
            config = load_config(
                optional_path(payload.get("config")),
                overrides=object_payload(payload.get("config_overrides")),
                env_path=optional_path(payload.get("env_file")),
            )
            self._send_json({"config": config.to_dict(), "database": redact_database_url(database_from_payload(payload).database_url)})
        except Exception as exc:  # noqa: BLE001 - local debug endpoint.
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

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
            body: dict[str, object] = {"task_id": task_id, "error": f"{type(exc).__name__}: {exc}"}
            if task_update_error:
                body["task_update_error"] = task_update_error
            self._send_json(body, status=HTTPStatus.INTERNAL_SERVER_ERROR)

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
            body: dict[str, object] = {"task_id": task_id, "error": f"{type(exc).__name__}: {exc}"}
            if task_update_error:
                body["task_update_error"] = task_update_error
            self._send_json(body, status=HTTPStatus.INTERNAL_SERVER_ERROR)

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
            body: dict[str, object] = {"task_id": task_id, "error": f"{type(exc).__name__}: {exc}"}
            if task_update_error:
                body["task_update_error"] = task_update_error
            self._send_json(body, status=HTTPStatus.INTERNAL_SERVER_ERROR)

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
                    config_path=optional_path(payload.get("config")),
                    config_overrides=object_payload(payload.get("config_overrides")),
                    env_path=optional_path(payload.get("env_file")),
                    top_k=int(payload.get("top_k") or 5),
                    include_debug=bool(payload.get("debug")),
                )
            )
            response = {"task_id": task_id, "summary": format_search_summary(result), "result": result}
            finish_task(database, task_id, "completed", response, str(response["summary"]))
            self._send_json(response)
        except Exception as exc:  # noqa: BLE001 - local debug endpoint.
            task_update_error = mark_task_failed(database, task_id, exc)
            body: dict[str, object] = {"task_id": task_id, "error": f"{type(exc).__name__}: {exc}"}
            if task_update_error:
                body["task_update_error"] = task_update_error
            self._send_json(body, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_ask_db(self) -> None:
        payload = self._read_json()
        query = str(payload.get("query") or "").strip()
        if not query:
            self._send_json({"error": "query is required"}, status=HTTPStatus.BAD_REQUEST)
            return
        database = database_from_payload(payload)
        task_id: int | None = None
        try:
            task_id = create_task(database, "answer", query, task_request_payload(payload))
            result = run_pg_pipeline(
                PgPipelineOptions(
                    database=database,
                    query=query,
                    config_path=optional_path(payload.get("config")),
                    config_overrides=object_payload(payload.get("config_overrides")),
                    env_path=optional_path(payload.get("env_file")),
                    top_k=int(payload.get("top_k") or 5),
                    include_debug=bool(payload.get("debug")),
                )
            )
            answer = result.get("answer") if isinstance(result.get("answer"), dict) else {}
            response = {"task_id": task_id, "summary": str(answer.get("status") or "ok"), "result": result}
            finish_task(database, task_id, "completed", response, str(response["summary"]))
            self._send_json(response)
        except Exception as exc:  # noqa: BLE001 - local debug endpoint.
            task_update_error = mark_task_failed(database, task_id, exc)
            body: dict[str, object] = {"task_id": task_id, "error": f"{type(exc).__name__}: {exc}"}
            if task_update_error:
                body["task_update_error"] = task_update_error
            self._send_json(body, status=HTTPStatus.INTERNAL_SERVER_ERROR)

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
            task = get_task(database_from_payload(payload), task_id)
            if task is None:
                self._send_json({"error": "task not found"}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json({"task": task})
        except Exception as exc:  # noqa: BLE001 - local debug endpoint.
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length", "0") or "0")
        body = self.rfile.read(length).decode("utf-8") if length else "{}"
        data = json.loads(body or "{}")
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object")
        return data

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


def database_from_payload(payload: dict[str, Any]) -> DatabaseOptions:
    """Build database options from a web request payload."""

    database_url = str(payload.get("database_url") or "").strip() or None
    return DatabaseOptions.from_env(database_url)


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
        work_order_dir=optional_path(payload.get("work_order_dir")),
        manual_dir=optional_path(payload.get("manual_dir")),
        work_order_paths=path_list_payload(payload.get("work_order_paths")),
        manual_paths=path_list_payload(payload.get("manual_paths")),
        config_path=optional_path(payload.get("config")),
        config_overrides=object_payload(payload.get("config_overrides")),
        env_path=optional_path(payload.get("env_file")),
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
        config_path=optional_path(payload.get("config")),
        config_overrides=object_payload(payload.get("config_overrides")),
        env_path=optional_path(payload.get("env_file")),
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


def create_task(database: DatabaseOptions, task_type: str, query: str | None, request: dict[str, Any]) -> int:
    """Create a persistent workbench task and return its task ID."""

    with connect(database.database_url) as conn:
        with conn.cursor() as cur:
            create_task_schema(cur)
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
        conn.commit()
    return int(row[0])


def update_task_progress(database: DatabaseOptions, task_id: int, progress: dict[str, object], summary: str) -> None:
    """Persist running progress for a task."""

    with connect(database.database_url) as conn:
        with conn.cursor() as cur:
            create_task_schema(cur)
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
        conn.commit()


def request_task_pause(database: DatabaseOptions, task_id: int) -> dict[str, Any]:
    """Request a running task to pause at its next checkpoint."""

    with connect(database.database_url) as conn:
        with conn.cursor() as cur:
            create_task_schema(cur)
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
        conn.commit()
    return {"task_id": int(row[0]), "status": str(row[1]), "summary": "暂停请求已发送"}


def is_task_pause_requested(database: DatabaseOptions, task_id: int) -> bool:
    """Return whether a task has a pending pause request."""

    with connect(database.database_url) as conn:
        with conn.cursor() as cur:
            create_task_schema(cur)
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

    with connect(database.database_url) as conn:
        with conn.cursor() as cur:
            create_task_schema(cur)
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
        conn.commit()


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
    with connect(database.database_url) as conn:
        with conn.cursor() as cur:
            create_task_schema(cur)
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


def list_question_tabs(database: DatabaseOptions, *, limit: int = 120) -> list[dict[str, Any]]:
    """Return persisted question tabs reconstructed from search and answer tasks."""

    safe_limit = max(1, min(limit, 300))
    with connect(database.database_url) as conn:
        with conn.cursor() as cur:
            create_task_schema(cur)
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


def max_iso_datetime(left: object, right: object) -> object:
    """Return the lexicographically latest ISO datetime-like value."""

    if not left:
        return right
    if not right:
        return left
    return right if str(right) > str(left) else left


def get_task(database: DatabaseOptions, task_id: int) -> dict[str, Any] | None:
    """Return one workbench task with its stored request and result payloads."""

    with connect(database.database_url) as conn:
        with conn.cursor() as cur:
            create_task_schema(cur)
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
