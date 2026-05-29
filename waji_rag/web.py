"""A local web UI for PostgreSQL-backed RAG debugging."""

from __future__ import annotations

import json
import html
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from waji_rag import __version__
from waji_rag.config import DEFAULT_DASHSCOPE_BASE_URL, DEFAULT_DASHSCOPE_RERANK_BASE_URL, DEFAULT_DOCARBOR_ENV_PATH, load_config
from waji_rag.pg_index import (
    DatabaseOptions,
    PgIngestBuilder,
    PgIngestOptions,
    PgPipelineOptions,
    PgSchemaManager,
    PgSearchOptions,
    format_ingest_report_summary,
    format_search_summary,
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
    button, input, textarea {
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
    input, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      padding: 8px 10px;
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
    .header-actions button, .query-actions button {
      white-space: nowrap;
    }
    .meta {
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }
    main {
      width: min(1500px, calc(100vw - 28px));
      margin: 14px auto 28px;
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
    .workspace {
      margin-top: 14px;
      display: grid;
      grid-template-columns: minmax(260px, 300px) minmax(0, 1fr);
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
    .stage-node.fallback, .stage-node.skipped {
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
      .query-band, .workspace, .answer-layout, .inspector { grid-template-columns: 1fr; }
      .stage-rail { position: static; }
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
      <button id="openConfigBtn" class="secondary">配置</button>
      <button id="doctorBtn" class="ghost">环境检查</button>
    </div>
  </header>

  <main>
    <section class="query-band">
      <div>
        <label for="query">用户问题</label>
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
      </div>
    </section>

    <div id="status" class="status">页面已载入。可以单独运行“构建 / 检索 / 回答”，也可以运行“全流程”。</div>

    <div class="workspace">
      <aside class="stage-rail">
        <h2>执行阶段</h2>
        <div id="stageList" class="stage-list"></div>
      </aside>

      <div class="content-grid">
        <section class="panel">
          <h2>答案与备件</h2>
          <div class="answer-layout">
            <div id="answer" class="answer-box">尚未运行。</div>
            <div id="parts" class="part-box"><div class="empty">暂无备件候选</div></div>
          </div>
        </section>

        <section class="panel">
          <h2>多路召回</h2>
          <div id="retrievalBoard" class="retrieval-board"></div>
        </section>

        <section class="panel">
          <h2>阶段返回</h2>
          <div class="inspector">
            <div id="stageSummary"></div>
            <pre id="stageJson" class="json-box">{}</pre>
          </div>
        </section>

        <section class="panel">
          <h2>答案生成依据</h2>
          <div id="selectedEvidence" class="part-box"><div class="empty">暂无选中证据</div></div>
        </section>
      </div>
    </div>
  </main>

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

        <label class="checkline" for="enableEmbedding">
          <input id="enableEmbedding" type="checkbox">
          <span>启用 embedding / hybrid</span>
        </label>
        <div>
          <label for="embeddingModel">Embedding 模型</label>
          <input id="embeddingModel" value="text-embedding-v4">
        </div>
        <div>
          <label for="embeddingDimensions">向量维度</label>
          <input id="embeddingDimensions" type="number" min="1" value="1024">
        </div>
        <div>
          <label for="embeddingBatchSize">批量大小</label>
          <input id="embeddingBatchSize" type="number" min="1" value="10">
        </div>
        <div class="full">
          <label for="embeddingBaseUrl">Embedding Base URL</label>
          <input id="embeddingBaseUrl" value="__DASHSCOPE_BASE_URL__">
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

        <label class="checkline" for="enableLlm">
          <input id="enableLlm" type="checkbox">
          <span>启用 LLM 答案生成</span>
        </label>
        <div>
          <label for="llmModel">LLM 模型</label>
          <input id="llmModel" value="qwen3.5-plus">
        </div>
        <div class="full">
          <label for="llmBaseUrl">LLM Base URL</label>
          <input id="llmBaseUrl" value="__DASHSCOPE_BASE_URL__">
        </div>
      </div>
      <div class="modal-foot">
        <button id="loadDemoBtn" class="secondary">加载 Demo 配置</button>
        <button id="docArborEnvBtn" class="secondary">填入 DocArbor Env</button>
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
    const channels = [
      ["work_orders", "历史工单", "doc_type=work_order；字段权重优先 reported_issue，再看 solution/raw_text。"],
      ["manual_typical_faults", "典型故障手册", "doc_type=manual_typical_fault；优先 fault_title/file_name，再看正文 chunk。"],
      ["manual_fault_codes", "故障码手册", "问题出现故障码时先精确匹配 fault_code，未出现故障码则为空。"],
      ["part_evidence", "备件证据", "doc_type=part_evidence；从命中工单的备件字段抽取，不从问题猜备件。"]
    ];
    const stageOrder = [
      ["config", "配置解析", "读取页面配置、env 和模型开关"],
      ["init", "初始化 PG", "创建 PostgreSQL / pgvector 表结构"],
      ["ingest", "构建索引", "解析工单、HTML 转 Markdown、入库、建 BM25/向量"],
      ["retrieval", "多路召回", "历史工单、手册、故障码、备件证据分路召回"],
      ["rerank", "重排", "可选 rerank；失败或关闭则保留原顺序"],
      ["answer", "答案生成", "用选中证据和备件候选生成最终答复"]
    ];
    let appState = {
      stages: {},
      selectedStage: "config",
      lastResult: null
    };

    function configOverrides() {
      return {
        embedding: {
          enabled: $("enableEmbedding").checked,
          provider: "dashscope",
          model: $("embeddingModel").value.trim(),
          base_url: $("embeddingBaseUrl").value.trim(),
          dimensions: $("embeddingDimensions").value ? Number($("embeddingDimensions").value) : null,
          batch_size: $("embeddingBatchSize").value ? Number($("embeddingBatchSize").value) : 10
        },
        rerank: {
          enabled: $("enableRerank").checked,
          provider: "dashscope",
          model: $("rerankModel").value.trim(),
          base_url: $("rerankBaseUrl").value.trim(),
          top_n: $("evidenceTopK").value ? Number($("evidenceTopK").value) : 8
        },
        llm: {
          enabled: $("enableLlm").checked,
          provider: "dashscope",
          model: $("llmModel").value.trim(),
          base_url: $("llmBaseUrl").value.trim(),
          max_tokens: 1400,
          temperature: 0
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

    function queryPayload() {
      return {
        ...commonPayload(),
        query: $("query").value.trim(),
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

    function setStage(id, status, data = null, summary = "") {
      appState.stages[id] = {status, data, summary};
      appState.selectedStage = id;
      renderStages();
      renderStageInspector();
    }

    function resetStages() {
      appState.stages = {};
      for (const [id] of stageOrder) appState.stages[id] = {status: "pending", data: null, summary: ""};
      appState.selectedStage = "config";
      renderStages();
      renderStageInspector();
    }

    function renderStages() {
      $("stageList").innerHTML = "";
      for (const [id, title, note] of stageOrder) {
        const state = appState.stages[id] || {status: "pending", summary: ""};
        const button = document.createElement("button");
        button.className = `stage-node ${state.status || "pending"} ${appState.selectedStage === id ? "active" : ""}`;
        button.innerHTML = `
          <div class="stage-title">
            <span>${escapeHtml(title)}</span>
            <span class="pill ${state.status === "done" ? "ok" : state.status === "fallback" || state.status === "skipped" ? "warn" : ""}">${escapeHtml(state.status || "pending")}</span>
          </div>
          <div class="stage-note">${escapeHtml(state.summary || note)}</div>
        `;
        button.addEventListener("click", () => {
          appState.selectedStage = id;
          renderStages();
          renderStageInspector();
        });
        $("stageList").appendChild(button);
      }
    }

    function renderStageInspector() {
      const [stageId, title, note] = stageOrder.find(([id]) => id === appState.selectedStage) || stageOrder[0];
      const state = appState.stages[stageId] || {status: "pending", data: null, summary: ""};
      $("stageSummary").innerHTML = `
        <div class="evidence-row">
          <div class="row-title">${escapeHtml(title)}</div>
          <div class="row-meta">状态：${escapeHtml(state.status || "pending")}</div>
          <div class="row-meta">${escapeHtml(state.summary || note)}</div>
        </div>
      `;
      $("stageJson").textContent = JSON.stringify(state.data || {}, null, 2);
    }

    function renderPipelineResult(response) {
      const result = response.result || response;
      appState.lastResult = result;
      const retrieval = result.retrieval || result;
      const answer = result.answer || {};
      setStageFromTrace(result);
      renderAnswer(answer);
      renderParts(result.part_candidates || retrieval.part_candidates || []);
      renderRetrievalBoard(retrieval);
      renderSelectedEvidence(result.selected_evidence || []);
      if (result.retrieval) setStage("retrieval", "done", result.retrieval, formatRetrievalSummary(result.retrieval));
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
      setStage("retrieval", "done", result, formatRetrievalSummary(result));
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
      $("parts").innerHTML = parts.map(part => `
        <div class="part-row">
          <div class="row-title">${escapeHtml(part.part_number_name || part.part_name || "未知备件")}</div>
          <div class="row-meta">编码：${escapeHtml(part.part_code || "未提供")} · 数量：${escapeHtml(part.quantity || "未提供")}</div>
          <div class="row-meta">来源工单：${escapeHtml(part.work_order_id || "未知")} · ${escapeHtml(part.source_path || "")}</div>
        </div>
      `).join("");
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
              <div class="row-meta">mode=${escapeHtml(retrieval.mode || "")} · top_k=${escapeHtml(retrieval.top_k || "")}</div>
              <div class="term-list">${queryTerms.slice(0, 10).map(term => `<span class="pill">${escapeHtml(term)}</span>`).join("")}</div>
            </div>
            <div class="route-body">${body}</div>
          </div>
        `;
      }).join("");
    }

    function renderHit(hit, index) {
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

    function renderSelectedEvidence(items) {
      if (!items.length) {
        $("selectedEvidence").innerHTML = '<div class="empty">暂无选中证据</div>';
        return;
      }
      $("selectedEvidence").innerHTML = items.map((item, index) => `
        <div class="evidence-row">
          <div class="row-title">#${index + 1} ${escapeHtml(item.channel || "")} · ${escapeHtml(item.title || "")}</div>
          <div class="row-meta">doc_id=${escapeHtml(item.doc_id || "")} · score=${escapeHtml(item.score ?? "")}</div>
          <div class="hit-preview">${escapeHtml(item.body_preview || "")}</div>
        </div>
      `).join("");
    }

    function formatRetrievalSummary(retrieval) {
      const channelsPayload = retrieval.channels || {};
      return `mode=${retrieval.mode || ""} · ` + channels.map(([name, label]) => `${label}:${(channelsPayload[name] || []).length}`).join(" · ");
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
      resetStages();
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
        setStage("ingest", "done", ingestResult, ingestResult.summary || "入库完成");
        $("answer").textContent = "构建完成。下一步可以点击“检索”查看多路召回结果，或点击“回答”生成最终答案。";
        renderRetrievalBoard({channels: {}, mode: "", top_k: ""});
        renderParts([]);
        renderSelectedEvidence([]);
        setStatus("构建完成", "success");
      } catch (error) {
        setStatus(String(error), "error");
        const current = appState.selectedStage || "ingest";
        setStage(current, "error", {error: String(error)}, "构建失败");
      } finally {
        button.disabled = false;
      }
    }

    async function runFullFlow(button) {
      button.disabled = true;
      resetStages();
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
        setStage("ingest", "done", ingestResult, ingestResult.summary || "入库完成");

        setStatus("多路召回与答案生成中");
        setStage("retrieval", "active", queryPayload(), "分路召回证据");
        const askResult = await postJson("/api/ask-db", queryPayload());
        askResult.workflow = {config: preview, init: initResult, ingest: ingestResult};
        renderPipelineResult(askResult);
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
      try {
        setStatus("检索中");
        setStage("retrieval", "active", queryPayload(), "分路召回证据");
        const result = await postJson("/api/search-db", queryPayload());
        renderSearchResult(result);
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
      try {
        setStatus("问答中");
        const result = await postJson("/api/ask-db", queryPayload());
        renderPipelineResult(result);
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
        const result = await postJson("/api/config-preview", commonPayload());
        setStage("config", "done", result, "配置读取完成");
        setStatus("配置预览完成", "success");
      } catch (error) {
        setStatus(String(error), "error");
      } finally {
        button.disabled = false;
      }
    }

    function applyDemoDefaults() {
      $("workOrderDir").value = demoWorkOrderDir;
      $("manualDir").value = demoManualDir;
      $("query").value = defaultQuery;
      $("topK").value = "1";
      $("evidenceTopK").value = "4";
      $("workOrderLimit").value = "";
      $("manualLimit").value = "";
      $("ingestReset").checked = true;
      $("envFile").value = "";
      $("enableEmbedding").checked = false;
      $("enableRerank").checked = false;
      $("enableLlm").checked = false;
      setStatus("已加载 Demo 配置", "success");
    }

    function escapeHtml(value) {
      return String(value ?? "").replace(/[&<>"']/g, ch => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
      }[ch]));
    }

    $("openConfigBtn").addEventListener("click", () => $("configModal").classList.add("open"));
    $("closeConfigBtn").addEventListener("click", () => $("configModal").classList.remove("open"));
    $("saveConfigBtn").addEventListener("click", () => {
      $("configModal").classList.remove("open");
      setStatus("配置已保存到页面状态", "success");
    });
    $("configModal").addEventListener("click", (event) => {
      if (event.target.id === "configModal") $("configModal").classList.remove("open");
    });
    $("loadDemoBtn").addEventListener("click", applyDemoDefaults);
    $("docArborEnvBtn").addEventListener("click", () => {
      $("envFile").value = docArborEnvPath;
      setStatus("已填入 DocArbor Env 路径", "success");
    });
    $("previewConfigBtn").addEventListener("click", () => runPreviewConfig($("previewConfigBtn")));
    $("runBuildBtn").addEventListener("click", () => runBuild($("runBuildBtn")));
    $("runSearchBtn").addEventListener("click", () => runSearch($("runSearchBtn")));
    $("runAnswerBtn").addEventListener("click", () => runAsk($("runAnswerBtn")));
    $("runFullFlowBtn").addEventListener("click", () => runFullFlow($("runFullFlowBtn")));
    $("doctorBtn").addEventListener("click", async () => {
      try {
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
    resetStages();
    applyDemoDefaults();
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
        "__DASHSCOPE_RERANK_BASE_URL__": html.escape(DEFAULT_DASHSCOPE_RERANK_BASE_URL),
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
        if parsed.path == "/api/ingest-db":
            self._handle_ingest_db()
            return
        if parsed.path == "/api/search-db":
            self._handle_search_db()
            return
        if parsed.path == "/api/ask-db":
            self._handle_ask_db()
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

    def _handle_ingest_db(self) -> None:
        payload = self._read_json()
        try:
            options = PgIngestOptions(
                database=database_from_payload(payload),
                work_order_dir=optional_path(payload.get("work_order_dir")),
                manual_dir=optional_path(payload.get("manual_dir")),
                config_path=optional_path(payload.get("config")),
                config_overrides=object_payload(payload.get("config_overrides")),
                env_path=optional_path(payload.get("env_file")),
                reset=bool(payload.get("reset")),
                work_order_limit=optional_int(payload.get("work_order_limit")),
                manual_limit=optional_int(payload.get("manual_limit")),
                max_manual_chars=int(payload.get("max_manual_chars") or 1800),
            )
            report = PgIngestBuilder(options).ingest()
            self._send_json(
                {"summary": format_ingest_report_summary(report), "report": report.to_dict()},
                status=HTTPStatus.OK if not report.failed_items else HTTPStatus.MULTI_STATUS,
            )
        except Exception as exc:  # noqa: BLE001 - local debug endpoint.
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_search_db(self) -> None:
        payload = self._read_json()
        query = str(payload.get("query") or "").strip()
        if not query:
            self._send_json({"error": "query is required"}, status=HTTPStatus.BAD_REQUEST)
            return
        try:
            result = run_pg_search(
                PgSearchOptions(
                    database=database_from_payload(payload),
                    query=query,
                    config_path=optional_path(payload.get("config")),
                    config_overrides=object_payload(payload.get("config_overrides")),
                    env_path=optional_path(payload.get("env_file")),
                    top_k=int(payload.get("top_k") or 5),
                    include_debug=bool(payload.get("debug")),
                )
            )
            self._send_json({"summary": format_search_summary(result), "result": result})
        except Exception as exc:  # noqa: BLE001 - local debug endpoint.
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_ask_db(self) -> None:
        payload = self._read_json()
        query = str(payload.get("query") or "").strip()
        if not query:
            self._send_json({"error": "query is required"}, status=HTTPStatus.BAD_REQUEST)
            return
        try:
            result = run_pg_pipeline(
                PgPipelineOptions(
                    database=database_from_payload(payload),
                    query=query,
                    config_path=optional_path(payload.get("config")),
                    config_overrides=object_payload(payload.get("config_overrides")),
                    env_path=optional_path(payload.get("env_file")),
                    top_k=int(payload.get("top_k") or 5),
                    include_debug=bool(payload.get("debug")),
                )
            )
            answer = result.get("answer") if isinstance(result.get("answer"), dict) else {}
            self._send_json({"summary": str(answer.get("status") or "ok"), "result": result})
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


def object_payload(value: object) -> dict[str, Any] | None:
    """Return a dict payload or None."""

    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("payload section must be an object")
    return value


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
