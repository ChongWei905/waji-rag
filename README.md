# Waji RAG

Local debugging tools for an excavator after-sales diagnosis RAG workflow.

The formal executable path now uses PostgreSQL with `pgvector/pgvector:pg16`:

- work-order TXT parsing is written directly into PostgreSQL;
- manual HTML is converted to clean Markdown in memory before chunking;
- BM25 uses a standard field-weighted BM25 formula over PostgreSQL term tables;
- optional embedding config enables hybrid BM25 + pgvector retrieval;
- optional rerank and LLM config enables end-to-end answer generation;
- when no embedding provider is configured, retrieval automatically uses BM25 only.

## Quick Start

Start PostgreSQL:

```bash
docker compose up -d postgres
```

Environment check:

```bash
python -m waji_rag.cli doctor
```

Initialize the database schema:

```bash
python -m waji_rag.cli init-db
```

Ingest raw evidence:

```bash
python -m waji_rag.cli ingest-db \
  --work-order-dir /path/to/work_order_txt \
  --manual-dir /path/to/manual_html_or_md \
  --reset \
  --report-json /path/to/ingest_report.json \
  --debug
```

Search evidence:

```bash
python -m waji_rag.cli search-db \
  --query "用户报修机器风扇皮带异响，请回答有可能是哪些故障导致的，如何解决，相应故障需要更换备件的详细信息" \
  --top-k 5 \
  --debug
```

Run the full RAG pipeline:

```bash
python -m waji_rag.cli ask-db \
  --query "用户报修机器风扇皮带异响，请回答有可能是哪些故障导致的，如何解决，相应故障需要更换备件的详细信息（备件的编号及名称，备件编码，备件数量）" \
  --top-k 5 \
  --debug
```

On Windows, the same flow is:

```bat
docker compose up -d postgres

python -m waji_rag.cli init-db

python -m waji_rag.cli ingest-db ^
  --work-order-dir D:\waji\data\work_orders ^
  --manual-dir D:\waji\data\manuals ^
  --reset ^
  --report-json D:\waji\outputs\ingest_report.json ^
  --debug

python -m waji_rag.cli search-db ^
  --query "用户报修机器风扇皮带异响，请回答有可能是哪些故障导致的，如何解决，相应故障需要更换备件的详细信息" ^
  --top-k 5 ^
  --debug

python -m waji_rag.cli ask-db ^
  --query "用户报修机器风扇皮带异响，请回答有可能是哪些故障导致的，如何解决，相应故障需要更换备件的详细信息（备件的编号及名称，备件编码，备件数量）" ^
  --top-k 5 ^
  --debug
```

The default database URL is:

```text
postgresql://waji:waji@127.0.0.1:55432/waji_rag
```

Override it with `--database-url` or `WAJI_DATABASE_URL`.

## Web UI

Start PostgreSQL and the local web UI:

```bash
docker compose up -d postgres
python -m waji_rag.cli serve --host 127.0.0.1 --port 8765
```

Open:

```text
http://127.0.0.1:8765
```

The page can run the RAG workflow end to end. For the included demo data, click `加载 Demo 配置`, then `一键跑全流程`. The page initializes PostgreSQL, ingests work-order TXT and manual HTML, runs retrieval/answer generation, and shows the final answer, stage trace, recalled evidence, part candidates, and raw JSON in separate tabs.

## Environment Verification Scripts

Before running the full RAG flow on Windows, run these three checks:

```bat
python scripts\verify_pg.py

python scripts\verify_embedding.py ^
  --api-key "<embedding-api-key>" ^
  --provider dashscope ^
  --model text-embedding-v4 ^
  --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 ^
  --dimensions 1024

python scripts\verify_llm.py ^
  --api-key "<llm-api-key>" ^
  --provider dashscope ^
  --model qwen3.5-plus ^
  --base-url https://dashscope.aliyuncs.com/compatible-mode/v1
```

All three scripts return JSON and exit with code `0` only when the check passes. They verify:

- PostgreSQL connection, `pgvector`, `pg_trgm`, vector search, and write/read permissions;
- embedding document/query calls and returned vector dimensions;
- LLM chat-completions call and returned text.

## Optional Hybrid Retrieval

By default, `embedding.enabled=false`, so retrieval uses BM25. If an embedding provider is enabled and embeddings are stored in PostgreSQL, retrieval uses hybrid BM25 + pgvector. If the model call fails, the pipeline records the failure in `warnings` / `debug.retrieval_events` and degrades to BM25.

With a DocArbor-style `.env` file:

```bash
python -m waji_rag.cli ingest-db \
  --work-order-dir /path/to/work_order_txt \
  --manual-dir /path/to/manual_html_or_md \
  --reset \
  --env-file /path/to/.env \
  --enable-embedding \
  --embedding-model text-embedding-v4 \
  --embedding-dimensions 1024 \
  --debug

python -m waji_rag.cli ask-db \
  --query "风扇皮带异响，可能是什么故障，需要更换什么备件" \
  --env-file /path/to/.env \
  --enable-embedding \
  --enable-rerank \
  --enable-llm \
  --embedding-model text-embedding-v4 \
  --embedding-dimensions 1024 \
  --rerank-model qwen3-rerank \
  --llm-model qwen3.5-plus \
  --debug
```

Write a starter config:

```bash
python -m waji_rag.cli write-default-config --out config.json
```

The default HTTP provider is OpenAI-compatible. A command provider is also supported for local embedding services:

```json
{
  "retrieval": {
    "mode": "auto",
    "bm25_top_k": 20,
    "vector_top_k": 20,
    "hybrid_alpha": 0.75
  },
  "embedding": {
    "enabled": true,
    "provider": "command",
    "model": "your-local-embedding-model",
    "command": ["python", "embedder.py"]
  }
}
```

The command receives `{"model": "...", "texts": [...], "text_type": "document|query"}` on stdin and returns `{"embeddings": [[...], ...]}` on stdout.

## Compatibility Tools

The older JSONL/local-index commands are still available for debugging small slices:

```bash
python -m waji_rag.cli html-to-md --input-dir /path/to/html --out-dir /path/to/md
python -m waji_rag.cli parse-workorders --input-dir /path/to/txt --out-dir /path/to/jsonl
python -m waji_rag.cli build-index --work-orders-jsonl /path/to/work_orders.jsonl --parts-jsonl /path/to/parts_evidence.jsonl --manual-md-dir /path/to/md --out-dir /path/to/index
```

## Design Notes

See [docs/rag_design.md](docs/rag_design.md).
