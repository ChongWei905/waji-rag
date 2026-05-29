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

The page separates `索引构建` from `检索与回答`. The build page starts large ingest jobs in the background, polls task progress, and shows source directories, file progress, current file, document counts, term rows, embeddings, and failures. The QA page focuses on retrieval and answer generation. Selecting a stage in the left rail changes the main content to the corresponding stage details instead of showing every panel at once.

The web UI also persists build/search/answer runs in PostgreSQL as task records. After ingesting a document batch once, you can repeatedly run new retrieval or answer tasks from the same database, then reload any previous task from the left-side task list to inspect its status, request, result, retrieval channels, selected evidence, and generated answer.

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

python scripts\verify_embedding.py ^
  --provider vllm ^
  --base-url http://127.0.0.1:8888/v1 ^
  --dimensions 0 ^
  --no-proxy-hosts localhost,127.0.0.1,127.0.0.0/8,::1,10.30.4.5,192.168.0.0/16

python scripts\verify_llm.py ^
  --api-key "<llm-api-key>" ^
  --provider dashscope ^
  --model qwen3.5-plus ^
  --base-url https://dashscope.aliyuncs.com/compatible-mode/v1

python scripts\verify_llm.py ^
  --provider vllm ^
  --model qwen-local ^
  --base-url http://10.30.4.5:8000/v1 ^
  --no-proxy-hosts localhost,127.0.0.1,127.0.0.0/8,::1,10.30.4.5,192.168.0.0/16
```

All three scripts return JSON and exit with code `0` only when the check passes. They verify:

- PostgreSQL connection, `pgvector`, `pg_trgm`, vector search, and write/read permissions;
- embedding document/query calls and returned vector dimensions;
- LLM chat-completions call and returned text.

## Optional Hybrid Retrieval

By default, `embedding.enabled=false`, so retrieval uses BM25. If an embedding provider is enabled and embeddings are stored in PostgreSQL, retrieval uses hybrid BM25 + pgvector. If the model call fails, the pipeline records the failure in `warnings` / `debug.retrieval_events` and degrades to BM25.

For local vLLM/OpenAI-compatible embedding services, use `--embedding-provider vllm` and a base URL such as `http://127.0.0.1:8888/v1`. API key and model can be left empty for local vLLM, and `--embedding-dimensions 0` means the client will not send a `dimensions` field. Model calls bypass proxies for `localhost`, `127.0.0.1`, `127.0.0.0/8`, and `::1` by default; add more hosts or ranges with `--embedding-no-proxy-hosts`, `--llm-no-proxy-hosts`, `--rerank-no-proxy-hosts`, `DOCARBOR_MODEL_NO_PROXY_HOSTS`, `DOCARBOR_EMBEDDING_NO_PROXY_HOSTS`, or `DOCARBOR_LLM_NO_PROXY_HOSTS`, for example `10.30.4.5,192.168.0.0/16,*.company.local`.

With a DocArbor-style `.env` file:

```bash
python -m waji_rag.cli ingest-db \
  --work-order-dir /path/to/work_order_txt \
  --manual-dir /path/to/manual_html_or_md \
  --reset \
  --env-file /path/to/.env \
  --enable-embedding \
  --embedding-provider dashscope \
  --embedding-model text-embedding-v4 \
  --embedding-dimensions 1024 \
  --debug

python -m waji_rag.cli ask-db \
  --query "风扇皮带异响，可能是什么故障，需要更换什么备件" \
  --enable-embedding \
  --enable-rerank \
  --enable-llm \
  --embedding-provider dashscope \
  --embedding-model text-embedding-v4 \
  --embedding-base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --embedding-api-key "<embedding-api-key>" \
  --embedding-dimensions 1024 \
  --rerank-model qwen3-rerank \
  --rerank-base-url https://dashscope.aliyuncs.com/compatible-api/v1 \
  --rerank-api-key "<rerank-api-key>" \
  --rerank-no-proxy-hosts localhost,127.0.0.1,127.0.0.0/8,::1 \
  --llm-provider dashscope \
  --llm-model qwen3.5-plus \
  --llm-base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --llm-api-key "<llm-api-key>" \
  --llm-no-proxy-hosts localhost,127.0.0.1,127.0.0.0/8,::1 \
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
