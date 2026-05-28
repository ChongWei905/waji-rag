"""A tiny local web UI for Windows-friendly pipeline debugging."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from waji_rag import __version__
from waji_rag.html_batch import ConvertOptions, HtmlToMarkdownBatch, format_report_summary
from waji_rag.work_order import (
    WorkOrderBatchOptions,
    WorkOrderBatchParser,
    format_report_summary as format_work_order_report_summary,
    write_report as write_work_order_report,
)


INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Waji RAG Debug</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f7f4;
      --ink: #202124;
      --muted: #6f6f66;
      --line: #d7d3c8;
      --panel: #ffffff;
      --accent: #0f766e;
      --danger: #9f1239;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }
    header {
      border-bottom: 1px solid var(--line);
      background: #fff;
      padding: 14px 22px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
    }
    h1 {
      font-size: 18px;
      line-height: 1.2;
      margin: 0;
      font-weight: 700;
      letter-spacing: 0;
    }
    main {
      width: min(1180px, calc(100vw - 32px));
      margin: 18px auto 32px;
      display: grid;
      grid-template-columns: 360px 1fr;
      gap: 16px;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }
    h2 {
      font-size: 15px;
      margin: 0 0 14px;
      letter-spacing: 0;
    }
    label {
      display: block;
      font-size: 13px;
      color: var(--muted);
      margin: 12px 0 6px;
    }
    input {
      width: 100%;
      height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
      font: inherit;
      background: #fff;
    }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    button {
      border: 0;
      border-radius: 6px;
      background: var(--accent);
      color: #fff;
      height: 40px;
      padding: 0 14px;
      font-weight: 700;
      cursor: pointer;
      margin-top: 14px;
    }
    button.secondary {
      background: #303030;
    }
    button:disabled {
      opacity: .55;
      cursor: wait;
    }
    .status {
      min-height: 30px;
      padding: 8px 10px;
      border-radius: 6px;
      background: #f4f4ef;
      border: 1px solid var(--line);
      font-size: 13px;
      color: var(--muted);
      margin-bottom: 12px;
    }
    .status.error {
      color: var(--danger);
      border-color: #fecdd3;
      background: #fff1f2;
    }
    pre {
      margin: 0;
      min-height: 520px;
      max-height: 70vh;
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
    }
    .meta {
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }
    @media (max-width: 860px) {
      main { grid-template-columns: 1fr; }
      .row { grid-template-columns: 1fr; }
    }
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
        <h2>HTML 转 Markdown</h2>
        <label for="inputDir">HTML 目录</label>
        <input id="inputDir" placeholder="D:\\waji\\manual_html">
        <label for="outDir">Markdown 输出目录</label>
        <input id="outDir" placeholder="D:\\waji\\outputs\\manual_md">
        <div class="row">
          <div>
            <label for="limit">数量上限</label>
            <input id="limit" type="number" min="0" placeholder="留空表示全量">
          </div>
          <div>
            <label for="reportJson">报告文件</label>
            <input id="reportJson" placeholder="可选 report.json">
          </div>
        </div>
        <button id="convertBtn">运行转换</button>
        <button id="doctorBtn" class="secondary">环境检查</button>
      </section>
      <section style="margin-top: 16px;">
        <h2>工单 TXT 解析</h2>
        <label for="workOrderInputDir">工单 TXT 目录</label>
        <input id="workOrderInputDir" placeholder="D:\\waji\\data\\work_orders">
        <label for="workOrderOutDir">解析输出目录</label>
        <input id="workOrderOutDir" placeholder="D:\\waji\\outputs\\work_orders">
        <div class="row">
          <div>
            <label for="workOrderLimit">数量上限</label>
            <input id="workOrderLimit" type="number" min="0" placeholder="留空表示全量">
          </div>
          <div>
            <label for="workOrderReportJson">报告文件</label>
            <input id="workOrderReportJson" placeholder="可选 report.json">
          </div>
        </div>
        <button id="parseWorkOrdersBtn">解析工单</button>
      </section>
    </div>
    <section>
      <h2>运行结果</h2>
      <div id="status" class="status">ready</div>
      <pre id="output">{}</pre>
    </section>
  </main>
  <script>
    const $ = (id) => document.getElementById(id);

    async function postJson(url, payload) {
      const response = await fetch(url, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || response.statusText);
      }
      return data;
    }

    async function getJson(url) {
      const response = await fetch(url);
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || response.statusText);
      }
      return data;
    }

    function show(data, error) {
      $("status").className = error ? "status error" : "status";
      $("status").textContent = error ? String(error) : "ok";
      $("output").textContent = JSON.stringify(data, null, 2);
    }

    async function loadDoctor() {
      try {
        const data = await getJson("/api/doctor");
        $("version").textContent = data.waji_rag_version + " · " + data.platform;
        show(data);
      } catch (error) {
        show({}, error);
      }
    }

    $("doctorBtn").addEventListener("click", loadDoctor);
    $("convertBtn").addEventListener("click", async () => {
      const button = $("convertBtn");
      button.disabled = true;
      $("status").className = "status";
      $("status").textContent = "running";
      try {
        const payload = {
          input_dir: $("inputDir").value.trim(),
          out_dir: $("outDir").value.trim(),
          limit: $("limit").value ? Number($("limit").value) : null,
          report_json: $("reportJson").value.trim() || null
        };
        const data = await postJson("/api/html-to-md", payload);
        show(data);
      } catch (error) {
        show({}, error);
      } finally {
        button.disabled = false;
      }
    });
    $("parseWorkOrdersBtn").addEventListener("click", async () => {
      const button = $("parseWorkOrdersBtn");
      button.disabled = true;
      $("status").className = "status";
      $("status").textContent = "running";
      try {
        const payload = {
          input_dir: $("workOrderInputDir").value.trim(),
          out_dir: $("workOrderOutDir").value.trim(),
          limit: $("workOrderLimit").value ? Number($("workOrderLimit").value) : null,
          report_json: $("workOrderReportJson").value.trim() || null
        };
        const data = await postJson("/api/parse-workorders", payload);
        show(data);
      } catch (error) {
        show({}, error);
      } finally {
        button.disabled = false;
      }
    });
    loadDoctor();
  </script>
</body>
</html>
"""


class RagDebugHandler(BaseHTTPRequestHandler):
    """HTTP handler for the local debugging UI."""

    server_version = "WajiRagDebug/0.1"

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
        """Handle mutating local debug actions."""

        parsed = urlparse(self.path)
        if parsed.path == "/api/html-to-md":
            self._handle_html_to_md()
            return
        if parsed.path == "/api/parse-workorders":
            self._handle_parse_workorders()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "not found")

    def log_message(self, format: str, *args: object) -> None:
        """Write compact request logs to stderr."""

        super().log_message(format, *args)

    def _handle_html_to_md(self) -> None:
        payload = self._read_json()
        input_dir = str(payload.get("input_dir") or "").strip()
        out_dir = str(payload.get("out_dir") or "").strip()
        if not input_dir or not out_dir:
            self._send_json({"error": "input_dir and out_dir are required"}, status=HTTPStatus.BAD_REQUEST)
            return

        options = ConvertOptions(
            input_dir=Path(input_dir),
            output_dir=Path(out_dir),
            limit=payload.get("limit"),
        )
        try:
            report = HtmlToMarkdownBatch(options).convert_directory()
            report_json = payload.get("report_json")
            if report_json:
                Path(str(report_json)).parent.mkdir(parents=True, exist_ok=True)
                Path(str(report_json)).write_text(
                    json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            self._send_json(
                {
                    "summary": format_report_summary(report),
                    "report": report.to_dict(),
                },
                status=HTTPStatus.OK if not report.failed_files else HTTPStatus.MULTI_STATUS,
            )
        except Exception as exc:  # noqa: BLE001 - local debug endpoint.
            self._send_json(
                {"error": f"{type(exc).__name__}: {exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def _handle_parse_workorders(self) -> None:
        payload = self._read_json()
        input_dir = str(payload.get("input_dir") or "").strip()
        out_dir = str(payload.get("out_dir") or "").strip()
        if not input_dir or not out_dir:
            self._send_json({"error": "input_dir and out_dir are required"}, status=HTTPStatus.BAD_REQUEST)
            return

        options = WorkOrderBatchOptions(
            input_dir=Path(input_dir),
            output_dir=Path(out_dir),
            limit=payload.get("limit"),
        )
        try:
            report = WorkOrderBatchParser(options).parse_directory()
            report_json = payload.get("report_json")
            if report_json:
                write_work_order_report(report, Path(str(report_json)))
            self._send_json(
                {
                    "summary": format_work_order_report_summary(report),
                    "report": report.to_dict(),
                },
                status=HTTPStatus.OK if not report.failed_files else HTTPStatus.MULTI_STATUS,
            )
        except Exception as exc:  # noqa: BLE001 - local debug endpoint.
            self._send_json(
                {"error": f"{type(exc).__name__}: {exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

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
