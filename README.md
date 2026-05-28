# Waji RAG

Local debugging tools for an excavator after-sales diagnosis RAG workflow.

The current implementation focuses on the first executable stage:

- cross-platform Python CLI;
- local web debugging UI;
- HTML to Markdown cleaning for diagnosis manuals;
- work-order TXT parsing and part evidence export;
- JSON conversion reports for Windows validation.

## Quick Start

Environment check:

```bash
python -m waji_rag.cli doctor
```

Start the local web UI:

```bash
python -m waji_rag.cli serve --host 127.0.0.1 --port 8765
```

Open:

```text
http://127.0.0.1:8765
```

Batch convert HTML manuals to Markdown:

```bash
python -m waji_rag.cli html-to-md \
  --input-dir /path/to/manual_html \
  --out-dir /path/to/manual_md \
  --limit 10 \
  --report-json /path/to/html_to_md_report.json \
  --debug
```

On Windows, use paths such as:

```bat
python -m waji_rag.cli html-to-md ^
  --input-dir D:\waji\data\manual_html ^
  --out-dir D:\waji\outputs\manual_md ^
  --limit 10 ^
  --report-json D:\waji\outputs\html_to_md_report.json ^
  --debug
```

Parse work-order TXT files into JSONL and CSV evidence files:

```bash
python -m waji_rag.cli parse-workorders \
  --input-dir /path/to/work_order_txt \
  --out-dir /path/to/work_order_outputs \
  --limit 10 \
  --report-json /path/to/work_order_report.json \
  --debug
```

On Windows:

```bat
python -m waji_rag.cli parse-workorders ^
  --input-dir D:\waji\data\work_orders ^
  --out-dir D:\waji\outputs\work_orders ^
  --limit 10 ^
  --report-json D:\waji\outputs\work_order_report.json ^
  --debug
```

## Design Notes

See [docs/rag_design.md](docs/rag_design.md).
