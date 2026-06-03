"""PostgreSQL and pgvector backed indexing and retrieval."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from html_to_markdown import MarkdownConverter, detect_table_count

from waji_rag.config import AppConfig, load_config, redact_secrets
from waji_rag.embedding import CommandEmbeddingProvider, EmbeddingProviderError, OpenAICompatibleEmbeddingProvider
from waji_rag.html_batch import DEFAULT_ENCODINGS, is_lossy_encoding, read_text_with_fallback
from waji_rag.index_build import (
    FIELD_WEIGHTS,
    IndexDocument,
    chunk_markdown,
    clean_string,
    first_non_empty,
    infer_manual_metadata,
    join_text,
    short_hash,
    tokenize_text,
)
from waji_rag.llm import (
    DashScopeRerankClient,
    extract_answer_facts,
    generate_harness_answer,
    judge_work_order_relevance,
    ModelProviderError,
    parse_json_object,
    select_manual_titles,
)
from waji_rag.work_order import WorkOrderParser, WorkOrderRecord, iter_txt_files


DEFAULT_DATABASE_URL = "postgresql://waji:waji@127.0.0.1:55432/waji_rag"
SCHEMA_VERSION = 1
DEFAULT_BM25_K1 = 1.5
DEFAULT_BM25_B = 0.75
DEFAULT_BM25_BATCH_TERM_ROWS = 100_000
DEFAULT_SOURCE_CHECKPOINT_FILES = 50
DEFAULT_WORK_ORDER_FILTER_CONCURRENCY = 8
MAX_WORK_ORDER_FILTER_CONCURRENCY = 16
MANUAL_SUFFIXES = {".html", ".htm", ".md", ".markdown"}
FAULT_CODE_IN_QUERY = re.compile(r"\b[A-Za-z]\d{3,}[A-Za-z0-9_-]*\b")
APPLICATION_DATA_TABLES = (
    "rag_tasks",
    "document_embeddings",
    "document_terms",
    "document_fields",
    "documents",
    "manual_chunks",
    "part_evidence",
    "work_orders",
    "ingest_items",
    "ingest_runs",
)


@dataclass(slots=True)
class DatabaseOptions:
    """Connection settings for PostgreSQL."""

    database_url: str = DEFAULT_DATABASE_URL

    @classmethod
    def from_env(cls, database_url: str | None = None) -> "DatabaseOptions":
        """Build database options from explicit input or environment defaults."""

        return cls(database_url=database_url or os.getenv("WAJI_DATABASE_URL") or os.getenv("DATABASE_URL") or DEFAULT_DATABASE_URL)


@dataclass(slots=True)
class PgIngestOptions:
    """Configuration for ingesting raw RAG evidence into PostgreSQL."""

    database: DatabaseOptions
    work_order_dir: Path | None = None
    manual_dir: Path | None = None
    work_order_paths: tuple[Path, ...] = ()
    manual_paths: tuple[Path, ...] = ()
    config_path: Path | None = None
    config_overrides: dict[str, Any] | None = None
    env_path: Path | None = None
    reset: bool = False
    work_order_limit: int | None = None
    manual_limit: int | None = None
    max_manual_chars: int = 1800
    bm25_batch_term_rows: int = DEFAULT_BM25_BATCH_TERM_ROWS
    resume: bool = True
    encodings: tuple[str, ...] = DEFAULT_ENCODINGS
    progress_callback: Callable[[dict[str, object]], None] | None = None
    pause_callback: Callable[[], bool] | None = None


@dataclass(slots=True)
class PgIngestReport:
    """Summary report for a PostgreSQL ingest run."""

    started_at: str
    elapsed_seconds: float
    database_url: str
    work_order_dir: str | None = None
    manual_dir: str | None = None
    work_order_files: int = 0
    work_orders: int = 0
    part_records: int = 0
    manual_files: int = 0
    manual_chunks: int = 0
    total_documents: int = 0
    term_rows: int = 0
    embeddings: int = 0
    html_converted_in_memory: int = 0
    skipped_files: int = 0
    paused: bool = False
    timing_seconds: dict[str, float] = field(default_factory=dict)
    failed_items: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable report dictionary."""

        payload = asdict(self)
        payload["timing_seconds"] = {key: round(value, 3) for key, value in self.timing_seconds.items()}
        return payload


@dataclass(slots=True)
class PgSearchOptions:
    """Configuration for retrieving evidence from PostgreSQL."""

    database: DatabaseOptions
    query: str
    config_path: Path | None = None
    config_overrides: dict[str, Any] | None = None
    env_path: Path | None = None
    top_k: int = 8
    include_debug: bool = False


@dataclass(slots=True)
class PgPipelineOptions:
    """Configuration for retrieve, rerank, and answer generation."""

    database: DatabaseOptions
    query: str
    config_path: Path | None = None
    config_overrides: dict[str, Any] | None = None
    env_path: Path | None = None
    top_k: int = 8
    include_debug: bool = True
    progress_callback: Callable[[dict[str, object]], None] | None = None


@dataclass(slots=True)
class PgEmbeddingOptions:
    """Configuration for adding missing embeddings to existing documents."""

    database: DatabaseOptions
    config_path: Path | None = None
    config_overrides: dict[str, Any] | None = None
    env_path: Path | None = None
    limit: int | None = None
    progress_callback: Callable[[dict[str, object]], None] | None = None
    pause_callback: Callable[[], bool] | None = None


@dataclass(slots=True)
class PgEmbeddingReport:
    """Summary report for an embedding backfill run."""

    started_at: str
    elapsed_seconds: float
    database_url: str
    total_candidates: int = 0
    processed_documents: int = 0
    embeddings: int = 0
    paused: bool = False
    timing_seconds: dict[str, float] = field(default_factory=dict)
    failed_items: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable report dictionary."""

        payload = asdict(self)
        payload["timing_seconds"] = {key: round(value, 3) for key, value in self.timing_seconds.items()}
        return payload


@dataclass(slots=True)
class RetrievalHit:
    """One retrieved document hit."""

    document_id: int
    doc_id: str
    doc_type: str
    title: str
    score: float
    body_preview: str
    work_order_id: str | None
    source_path: str | None
    metadata: dict[str, object]
    matched_terms: list[dict[str, object]] = field(default_factory=list)
    vector_distance: float | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable hit dictionary."""

        return asdict(self)


@dataclass(slots=True)
class SearchDocumentUpsertResult:
    """Search document upsert output plus deferred BM25 rows."""

    document_id: int
    term_rows: int
    timings: dict[str, float]
    field_rows: list[tuple[object, ...]]
    term_row_values: list[tuple[object, ...]]


@dataclass(slots=True)
class SourceCompletion:
    """One source file that can be marked complete after BM25 rows are durable."""

    source_kind: str
    source_path: str
    content_hash: str
    counts: dict[str, int]


class IngestPaused(RuntimeError):
    """Raised when an ingest or embedding task is paused by user request."""


class PipelinePaused(RuntimeError):
    """Raised when the answer pipeline pauses and waits for a user retry."""

    def __init__(
        self,
        *,
        stage: str,
        reason: str,
        message: str,
        error: str | None = None,
        partial_result: dict[str, object] | None = None,
    ) -> None:
        """Store pause metadata and the latest observable pipeline snapshot."""

        super().__init__(message)
        self.stage = stage
        self.reason = reason
        self.error = error or message
        self.partial_result = partial_result or {}


class PgSchemaManager:
    """Create and reset the PostgreSQL schema used by the RAG pipeline."""

    def __init__(self, database: DatabaseOptions) -> None:
        """Store database options."""

        self.database = database

    def initialize(self, *, reset: bool = False) -> dict[str, object]:
        """Create required extensions, tables, and indexes."""

        started_at = time.time()
        with connect(self.database.database_url) as conn:
            with conn.cursor() as cur:
                if reset:
                    drop_schema_objects(cur)
                create_extensions(cur)
                create_schema(cur)
                cur.execute(
                    """
                    INSERT INTO schema_migrations(version, applied_at)
                    VALUES (%s, now())
                    ON CONFLICT (version) DO NOTHING
                    """,
                    (SCHEMA_VERSION,),
                )
            conn.commit()
        return {
            "database_url": redact_database_url(self.database.database_url),
            "schema_version": SCHEMA_VERSION,
            "reset": reset,
            "elapsed_seconds": round(time.time() - started_at, 3),
        }


class PgIngestBuilder:
    """Ingest raw work-order TXT and manual HTML/Markdown files into PostgreSQL."""

    def __init__(self, options: PgIngestOptions) -> None:
        """Store ingest options and lightweight parsers."""

        self.options = options
        self.parser = WorkOrderParser()
        self.converter = MarkdownConverter()
        self.config = load_config(
            options.config_path,
            overrides=options.config_overrides,
            env_path=options.env_path,
        )
        self.embedding_provider = build_embedding_provider(self.config)
        self.pending_embeddings: list[tuple[int, IndexDocument]] = []
        self.pending_bm25_field_rows: list[tuple[object, ...]] = []
        self.pending_bm25_term_rows: list[tuple[object, ...]] = []
        self.pending_source_completions: list[SourceCompletion] = []
        self.bm25_batch_term_rows = max(1, options.bm25_batch_term_rows)
        self.processed_file_count = 0

    def ingest(self) -> PgIngestReport:
        """Run schema initialization, parse files, and write searchable records."""

        start_time = time.time()
        self._emit_progress({"phase": "init", "message": "初始化 PostgreSQL schema", "percent": 0})
        PgSchemaManager(self.options.database).initialize(reset=self.options.reset)
        work_order_dir = self.options.work_order_dir.resolve() if self.options.work_order_dir else None
        manual_dir = self.options.manual_dir.resolve() if self.options.manual_dir else None
        work_order_files = resolve_work_order_input_files(
            work_order_dir,
            self.options.work_order_paths,
            limit=self.options.work_order_limit,
        )
        manual_files = resolve_manual_input_files(
            manual_dir,
            self.options.manual_paths,
            limit=self.options.manual_limit,
        )
        validate_ingest_inputs(work_order_dir, manual_dir, work_order_files, manual_files)
        self._emit_progress(
            {
                "phase": "scan_inputs",
                "message": "输入目录校验完成，准备扫描文件",
                "work_order_dir": str(work_order_dir) if work_order_dir else None,
                "manual_dir": str(manual_dir) if manual_dir else None,
                "percent": 1,
            }
        )

        report = PgIngestReport(
            started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            elapsed_seconds=0.0,
            database_url=redact_database_url(self.options.database.database_url),
            work_order_dir=str(work_order_dir) if work_order_dir else None,
            manual_dir=str(manual_dir) if manual_dir else None,
        )
        if work_order_dir is not None:
            report.work_order_files = len(work_order_files)
        if manual_dir is not None:
            report.manual_files = len(manual_files)
        if work_order_dir is None and work_order_files:
            report.work_order_files = len(work_order_files)
        if manual_dir is None and manual_files:
            report.manual_files = len(manual_files)
        self._emit_progress(
            self._progress_payload(
                phase="plan",
                message=f"发现工单 {report.work_order_files} 个，手册 {report.manual_files} 个",
                report=report,
            )
        )

        with connect(self.options.database.database_url) as conn:
            with conn.cursor() as cur:
                run_id = create_ingest_run(cur, self.config)
                if work_order_files:
                    self._ingest_work_orders(cur, work_order_files, report)
                    self._flush_bm25_batch(cur, report)
                    self._maybe_pause(cur, report)
                    self._flush_embedding_batch(cur, report)
                if manual_files:
                    self._ingest_manuals(cur, manual_dir, manual_files, report)
                    self._flush_bm25_batch(cur, report)
                    self._maybe_pause(cur, report)
                    self._flush_embedding_batch(cur, report)
                finish_ingest_run(cur, run_id, report)
            conn.commit()

        report.elapsed_seconds = round(time.time() - start_time, 3)
        self._emit_progress(
            self._progress_payload(
                phase="completed",
                message="索引构建完成",
                report=report,
                elapsed_seconds=report.elapsed_seconds,
            )
        )
        return report

    def _ingest_work_orders(self, cur: Any, txt_files: list[Path], report: PgIngestReport) -> None:
        report.work_order_files = len(txt_files)
        self._emit_progress(
            self._progress_payload(
                phase="work_orders",
                message=f"开始处理工单 TXT，共 {len(txt_files)} 个文件",
                report=report,
                current_index=0,
                current_total=len(txt_files),
            )
        )

        for file_index, txt_path in enumerate(txt_files, start=1):
            source_hash = ""
            try:
                self._maybe_pause(cur, report)
                source_hash = file_content_hash(txt_path)
                if self.options.resume and source_is_completed(cur, "work_order", str(txt_path), source_hash):
                    report.skipped_files += 1
                    self.processed_file_count = file_index
                    continue
                mark_source_running(cur, "work_order", str(txt_path), source_hash)
                parse_started = time.perf_counter()
                self._emit_progress(
                    self._progress_payload(
                        phase="work_orders",
                        message=f"处理工单 {file_index}/{len(txt_files)}",
                        report=report,
                        current_file=str(txt_path),
                        current_index=file_index,
                        current_total=len(txt_files),
                    )
                )
                text, encoding = read_text_with_fallback(txt_path, self.options.encodings)
                if is_lossy_encoding(encoding):
                    report.warnings.append(f"lossy_decode:{txt_path}:{encoding}")
                record = self.parser.parse(text, source_path=txt_path, encoding=encoding)
                add_timing(report, "parse_seconds", time.perf_counter() - parse_started)

                pg_started = time.perf_counter()
                delete_source_records(cur, str(txt_path))
                upsert_work_order(cur, record)
                add_timing(report, "pg_write_seconds", time.perf_counter() - pg_started)
                report.work_orders += 1
                if record.parse_warnings:
                    report.warnings.append(f"{txt_path}: {', '.join(record.parse_warnings)}")

                documents = build_documents_for_work_order(record)
                self.processed_file_count = file_index
                source_document_count = 0
                source_term_rows = 0
                source_field_rows: list[tuple[object, ...]] = []
                source_term_row_values: list[tuple[object, ...]] = []
                source_embedding_items: list[tuple[int, IndexDocument]] = []
                for document in documents:
                    result = upsert_search_document(cur, document, write_bm25=False)
                    merge_timings(report, result.timings)
                    report.total_documents += 1
                    report.term_rows += result.term_rows
                    source_document_count += 1
                    source_term_rows += result.term_rows
                    source_field_rows.extend(result.field_rows)
                    source_term_row_values.extend(result.term_row_values)
                    source_embedding_items.append((result.document_id, document))
                self.pending_bm25_field_rows.extend(source_field_rows)
                self.pending_bm25_term_rows.extend(source_term_row_values)
                self.pending_source_completions.append(
                    SourceCompletion(
                        source_kind="work_order",
                        source_path=str(txt_path),
                        content_hash=source_hash,
                        counts={
                            "documents": source_document_count,
                            "term_rows": source_term_rows,
                            "parts": len(record.parts),
                        },
                    )
                )
                for document_id, document in source_embedding_items:
                    self._queue_embedding(cur, document_id, document, report)
                if (
                    len(self.pending_bm25_term_rows) >= self.bm25_batch_term_rows
                    or len(self.pending_source_completions) >= DEFAULT_SOURCE_CHECKPOINT_FILES
                ):
                    self._flush_bm25_batch(cur, report)
                report.part_records += len(record.parts)
            except IngestPaused:
                raise
            except Exception as exc:  # noqa: BLE001 - keep per-file diagnostics.
                rollback_source_failure(cur, "work_order", str(txt_path), source_hash, exc)
                commit_cursor(cur)
                report.failed_items.append(
                    {"stage": "work_order", "input": str(txt_path), "error": f"{type(exc).__name__}: {exc}"}
                )
            finally:
                self.processed_file_count = max(self.processed_file_count, file_index)
                self._emit_progress(
                    self._progress_payload(
                        phase="work_orders",
                        message=f"已处理工单 {file_index}/{len(txt_files)}",
                        report=report,
                        current_file=str(txt_path),
                        current_index=file_index,
                        current_total=len(txt_files),
                    )
                )
            self._maybe_pause(cur, report)

    def _ingest_manuals(self, cur: Any, root: Path | None, manual_files: list[Path], report: PgIngestReport) -> None:
        report.manual_files = len(manual_files)
        self._emit_progress(
            self._progress_payload(
                phase="manuals",
                message=f"开始处理手册 HTML/MD，共 {len(manual_files)} 个文件",
                report=report,
                current_index=0,
                current_total=len(manual_files),
            )
        )

        for file_index, manual_path in enumerate(manual_files, start=1):
            source_hash = ""
            try:
                self._maybe_pause(cur, report)
                source_hash = file_content_hash(manual_path)
                if self.options.resume and source_is_completed(cur, "manual", str(manual_path), source_hash):
                    report.skipped_files += 1
                    self.processed_file_count = report.work_order_files + file_index
                    continue
                mark_source_running(cur, "manual", str(manual_path), source_hash)
                parse_started = time.perf_counter()
                self._emit_progress(
                    self._progress_payload(
                        phase="manuals",
                        message=f"处理手册 {file_index}/{len(manual_files)}",
                        report=report,
                        current_file=str(manual_path),
                        current_index=file_index,
                        current_total=len(manual_files),
                    )
                )
                markdown_text, encoding, converted_from_html, table_count = read_manual_as_markdown(
                    manual_path,
                    encodings=self.options.encodings,
                    converter=self.converter,
                )
                if is_lossy_encoding(encoding):
                    report.warnings.append(f"lossy_decode:{manual_path}:{encoding}")
                document_root = source_root_for_path(manual_path, root)
                metadata = infer_manual_metadata(manual_path, document_root)
                chunks = chunk_markdown(markdown_text, max_chars=max(self.options.max_manual_chars, 200))
                add_timing(report, "parse_seconds", time.perf_counter() - parse_started)
                if not chunks:
                    report.warnings.append(f"empty_manual_file: {manual_path}")
                    self.pending_source_completions.append(
                        SourceCompletion(
                            source_kind="manual",
                            source_path=str(manual_path),
                            content_hash=source_hash,
                            counts={"documents": 0, "term_rows": 0, "chunks": 0},
                        )
                    )
                    self._flush_bm25_batch(cur, report)
                    continue

                self.processed_file_count = report.work_order_files + file_index
                pg_started = time.perf_counter()
                delete_source_records(cur, str(manual_path))
                add_timing(report, "pg_write_seconds", time.perf_counter() - pg_started)
                source_document_count = 0
                source_term_rows = 0
                source_field_rows: list[tuple[object, ...]] = []
                source_term_row_values: list[tuple[object, ...]] = []
                source_embedding_items: list[tuple[int, IndexDocument]] = []
                for chunk_index, chunk_text in enumerate(chunks):
                    chunk_metadata = {
                        **metadata,
                        "chunk_index": chunk_index,
                        "chunk_count": len(chunks),
                        "encoding": encoding,
                        "converted_from_html": converted_from_html,
                        "table_count": table_count,
                    }
                    document = build_manual_document(
                        manual_path=manual_path,
                        root=document_root,
                        metadata=chunk_metadata,
                        chunk_text=chunk_text,
                        chunk_index=chunk_index,
                    )
                    pg_started = time.perf_counter()
                    insert_manual_chunk(cur, document, chunk_metadata)
                    add_timing(report, "pg_write_seconds", time.perf_counter() - pg_started)
                    result = upsert_search_document(cur, document, write_bm25=False)
                    merge_timings(report, result.timings)
                    report.manual_chunks += 1
                    report.total_documents += 1
                    report.term_rows += result.term_rows
                    source_document_count += 1
                    source_term_rows += result.term_rows
                    if converted_from_html:
                        report.html_converted_in_memory += 1 if chunk_index == 0 else 0
                    source_field_rows.extend(result.field_rows)
                    source_term_row_values.extend(result.term_row_values)
                    source_embedding_items.append((result.document_id, document))
                self.pending_bm25_field_rows.extend(source_field_rows)
                self.pending_bm25_term_rows.extend(source_term_row_values)
                self.pending_source_completions.append(
                    SourceCompletion(
                        source_kind="manual",
                        source_path=str(manual_path),
                        content_hash=source_hash,
                        counts={
                            "documents": source_document_count,
                            "term_rows": source_term_rows,
                            "chunks": len(chunks),
                        },
                    )
                )
                for document_id, document in source_embedding_items:
                    self._queue_embedding(cur, document_id, document, report)
                if (
                    len(self.pending_bm25_term_rows) >= self.bm25_batch_term_rows
                    or len(self.pending_source_completions) >= DEFAULT_SOURCE_CHECKPOINT_FILES
                ):
                    self._flush_bm25_batch(cur, report)
            except IngestPaused:
                raise
            except Exception as exc:  # noqa: BLE001 - keep per-file diagnostics.
                rollback_source_failure(cur, "manual", str(manual_path), source_hash, exc)
                commit_cursor(cur)
                report.failed_items.append(
                    {"stage": "manual", "input": str(manual_path), "error": f"{type(exc).__name__}: {exc}"}
                )
            finally:
                self.processed_file_count = max(self.processed_file_count, report.work_order_files + file_index)
                self._emit_progress(
                    self._progress_payload(
                        phase="manuals",
                        message=f"已处理手册 {file_index}/{len(manual_files)}",
                        report=report,
                        current_file=str(manual_path),
                        current_index=file_index,
                        current_total=len(manual_files),
                    )
                )
            self._maybe_pause(cur, report)

    def _flush_bm25_batch(self, cur: Any, report: PgIngestReport) -> None:
        if not self.pending_bm25_field_rows and not self.pending_bm25_term_rows and not self.pending_source_completions:
            return
        field_rows = self.pending_bm25_field_rows
        term_rows = self.pending_bm25_term_rows
        source_completions = self.pending_source_completions
        self.pending_bm25_field_rows = []
        self.pending_bm25_term_rows = []
        self.pending_source_completions = []
        self._emit_progress(
            self._progress_payload(
                phase="bm25",
                message=f"批量写入 BM25 词项 {len(term_rows)} 行",
                report=report,
            )
        )
        started_at = time.perf_counter()
        insert_bm25_rows(cur, field_rows=field_rows, term_rows=term_rows)
        for completion in source_completions:
            mark_source_completed(cur, completion)
        add_timing(report, "bm25_seconds", time.perf_counter() - started_at)
        commit_cursor(cur)
        self._emit_progress(
            self._progress_payload(
                phase="bm25",
                message=f"已写入 BM25 词项 {report.term_rows} 行",
                report=report,
            )
        )

    def _maybe_pause(self, cur: Any, report: PgIngestReport) -> None:
        if self.options.pause_callback is None or not self.options.pause_callback():
            return
        self._emit_progress(
            self._progress_payload(
                phase="paused",
                message="收到暂停请求，正在保存断点",
                report=report,
            )
        )
        self._flush_bm25_batch(cur, report)
        self._flush_embedding_batch(cur, report)
        commit_cursor(cur)
        report.paused = True
        raise IngestPaused("ingest paused by user request")

    def _queue_embedding(self, cur: Any, document_id: int, document: IndexDocument, report: PgIngestReport) -> None:
        if self.embedding_provider is None:
            return
        self.pending_embeddings.append((document_id, document))
        batch_size = max(1, int(self.config.embedding.batch_size))
        if len(self.pending_embeddings) >= batch_size:
            self._flush_embedding_batch(cur, report)

    def _flush_embedding_batch(self, cur: Any, report: PgIngestReport) -> None:
        if not self.pending_embeddings:
            return
        if self.embedding_provider is None:
            self.pending_embeddings.clear()
            return

        items = self.pending_embeddings
        self.pending_embeddings = []
        self._emit_progress(
            self._progress_payload(
                phase="embedding",
                message=f"批量生成向量 {len(items)} 条",
                report=report,
            )
        )
        try:
            stored_count = store_embedding_batch(cur, items, self.config, self.embedding_provider, report)
        except Exception as exc:  # noqa: BLE001 - embedding can be retried by backfill.
            report.warnings.append(f"embedding_flush_failed:{len(items)}: {type(exc).__name__}: {exc}")
            stored_count = 0
        report.embeddings += stored_count
        self._emit_progress(
            self._progress_payload(
                phase="embedding",
                message=f"已写入向量 {report.embeddings} 条",
                report=report,
            )
        )

    def _emit_progress(self, progress: dict[str, object]) -> None:
        """Emit ingest progress when a callback was configured."""

        if self.options.progress_callback is None:
            return
        self.options.progress_callback(progress)

    def _progress_payload(
        self,
        *,
        phase: str,
        message: str,
        report: PgIngestReport,
        current_file: str | None = None,
        current_index: int | None = None,
        current_total: int | None = None,
        elapsed_seconds: float | None = None,
    ) -> dict[str, object]:
        """Build a compact progress payload for UI polling."""

        total_files = report.work_order_files + report.manual_files
        processed_files = 0
        if phase == "work_orders" and current_index is not None:
            processed_files = current_index
        elif phase == "manuals" and current_index is not None:
            processed_files = report.work_order_files + current_index
        elif phase in {"bm25", "embedding", "paused"}:
            processed_files = min(total_files, self.processed_file_count)
        percent = round((processed_files / total_files) * 100, 1) if total_files else 0.0
        if phase == "completed":
            processed_files = total_files
            percent = 100.0
        return {
            "phase": phase,
            "message": message,
            "current_file": current_file,
            "current_index": current_index,
            "current_total": current_total,
            "percent": percent,
            "processed_files": processed_files,
            "total_files": total_files,
            "counts": ingest_report_counts(report),
            "timing_seconds": rounded_timings(report.timing_seconds),
            "failed_count": len(report.failed_items),
            "recent_failures": report.failed_items[-5:],
            "warnings": report.warnings[-5:],
            "elapsed_seconds": elapsed_seconds,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }


class PgEmbeddingBackfill:
    """Generate missing document embeddings for an existing PostgreSQL index."""

    def __init__(self, options: PgEmbeddingOptions) -> None:
        """Store embedding backfill options and provider."""

        self.options = options
        self.config = load_config(
            options.config_path,
            overrides=options.config_overrides,
            env_path=options.env_path,
        )
        self.embedding_provider = build_embedding_provider(self.config)

    def backfill(self) -> PgEmbeddingReport:
        """Scan documents missing embeddings and write vectors in batches."""

        if self.embedding_provider is None:
            raise ValueError("embedding config is not available")
        started_at = time.time()
        report = PgEmbeddingReport(
            started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            elapsed_seconds=0.0,
            database_url=redact_database_url(self.options.database.database_url),
        )
        self._emit_progress({"phase": "embedding", "message": "初始化 Embedding 补齐任务", "percent": 0})
        PgSchemaManager(self.options.database).initialize(reset=False)
        with connect(self.options.database.database_url) as conn:
            with conn.cursor() as cur:
                report.total_candidates = count_missing_embeddings(cur, self.config, limit=self.options.limit)
                self._emit_progress(self._progress_payload(report, message=f"待补向量 {report.total_candidates} 条"))
                while report.processed_documents < report.total_candidates:
                    self._maybe_pause(cur, report)
                    batch_limit = min(self.config.embedding.batch_size, report.total_candidates - report.processed_documents)
                    documents = fetch_missing_embedding_documents(cur, self.config, limit=max(batch_limit, 1))
                    if not documents:
                        break
                    stored_count = store_embedding_batch(cur, documents, self.config, self.embedding_provider, report)
                    report.processed_documents += len(documents)
                    report.embeddings += stored_count
                    commit_cursor(cur)
                    self._emit_progress(self._progress_payload(report, message=f"已补向量 {report.embeddings} 条"))
            conn.commit()
        report.elapsed_seconds = round(time.time() - started_at, 3)
        self._emit_progress(self._progress_payload(report, message="Embedding 补齐完成", completed=True))
        return report

    def _emit_progress(self, progress: dict[str, object]) -> None:
        """Emit embedding progress when a callback was configured."""

        if self.options.progress_callback is None:
            return
        self.options.progress_callback(progress)

    def _maybe_pause(self, cur: Any, report: PgEmbeddingReport) -> None:
        if self.options.pause_callback is None or not self.options.pause_callback():
            return
        commit_cursor(cur)
        report.paused = True
        self._emit_progress(self._progress_payload(report, message="Embedding 补齐已暂停"))
        raise IngestPaused("embedding backfill paused by user request")

    @staticmethod
    def _progress_payload(report: PgEmbeddingReport, *, message: str, completed: bool = False) -> dict[str, object]:
        total = max(report.total_candidates, 0)
        processed = min(report.processed_documents, total) if total else report.processed_documents
        percent = 100.0 if completed else round((processed / total) * 100, 1) if total else 0.0
        return {
            "phase": "embedding",
            "message": message,
            "percent": percent,
            "processed_files": processed,
            "total_files": total,
            "counts": {
                "total_documents": report.total_candidates,
                "embeddings": report.embeddings,
                "failed_items": len(report.failed_items),
            },
            "timing_seconds": rounded_timings(report.timing_seconds),
            "failed_count": len(report.failed_items),
            "recent_failures": report.failed_items[-5:],
            "warnings": report.warnings[-5:],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }


class PgRetriever:
    """Retrieve work-order, manual, and part evidence from PostgreSQL."""

    def __init__(self, database: DatabaseOptions, config: AppConfig) -> None:
        """Store retrieval dependencies."""

        self.database = database
        self.config = config
        self.embedding_provider = build_embedding_provider(config)

    def retrieve(self, query: str, *, top_k: int = 8, include_debug: bool = False) -> dict[str, object]:
        """Retrieve a structured evidence package for one diagnostic query."""

        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")

        effective_mode = self.config.retrieval_mode()
        work_order_mode = self.config.retrieval_mode_for("work_orders")
        manual_mode = self.config.retrieval_mode_for("manual_typical_faults")
        retrieval_events: list[dict[str, object]] = []
        terms = unique_terms(tokenize_text(query))
        channels: dict[str, list[RetrievalHit]] = {}
        with connect(self.database.database_url) as conn:
            with conn.cursor() as cur:
                work_order_hits = self._search_channel(
                    cur,
                    query,
                    terms,
                    top_k=top_k,
                    doc_types=["work_order"],
                    mode=work_order_mode,
                    channel_name="work_orders",
                    debug_events=retrieval_events,
                )
                typical_hits = self._search_channel(
                    cur,
                    query,
                    terms,
                    top_k=top_k,
                    doc_types=["manual_typical_fault"],
                    mode=manual_mode,
                    channel_name="manual_typical_faults",
                    debug_events=retrieval_events,
                )
                fault_code_hits = self._search_fault_code_channel(
                    cur,
                    query,
                    terms,
                    top_k=top_k,
                    mode=manual_mode,
                    debug_events=retrieval_events,
                )
                retrieval_events.append(
                    {
                        "channel": "part_evidence",
                        "stage": "search",
                        "status": "skipped",
                        "reason": "part_evidence_independent_retrieval_disabled",
                    }
                )
                channels = {
                    "work_orders": work_order_hits,
                    "manual_typical_faults": typical_hits,
                    "manual_fault_codes": fault_code_hits,
                    "part_evidence": [],
                }
                linked_work_order_ids = collect_work_order_ids(work_order_hits)
                part_candidates = fetch_part_candidates(cur, linked_work_order_ids)

        channel_payloads: dict[str, list[dict[str, object]]] = {
            "work_orders": [hit.to_dict() for hit in channels.get("work_orders", [])],
            "manual_typical_faults": [hit.to_dict() for hit in channels.get("manual_typical_faults", [])],
            "manual_fault_codes": [hit.to_dict() for hit in channels.get("manual_fault_codes", [])],
            "part_evidence": [
                part_candidate_to_channel_hit(part, rank=rank)
                for rank, part in enumerate(part_candidates, start=1)
            ],
        }
        payload: dict[str, object] = {
            "query": query,
            "mode": effective_mode,
            "channel_modes": {
                "work_orders": work_order_mode,
                "manual_typical_faults": manual_mode,
                "manual_fault_codes": manual_mode,
            },
            "top_k": top_k,
            "channels": channel_payloads,
            "part_candidates": part_candidates,
            "part_candidate_source": {
                "linked_work_order_ids": linked_work_order_ids,
                "limit_applied": False,
                "source": "work_order_hits",
                "independent_part_retrieval_enabled": False,
                "search_hit_count": 0,
                "work_order_hit_count": len(channels.get("work_orders", [])),
            },
        }
        fallback_events = [event for event in retrieval_events if event.get("status") == "fallback"]
        if fallback_events:
            payload["warnings"] = [
                f"{event.get('channel')}:{event.get('stage')}:{event.get('reason')}" for event in fallback_events
            ]
        if include_debug:
            payload["debug"] = {
                "query_terms": terms,
                "bm25": {"k1": DEFAULT_BM25_K1, "b": DEFAULT_BM25_B, "field_weights": FIELD_WEIGHTS},
                "embedding_enabled": self.config.embedding.is_available(),
                "retrieval_events": retrieval_events,
                "database_url": redact_database_url(self.database.database_url),
            }
        return payload

    def _search_fault_code_channel(
        self,
        cur: Any,
        query: str,
        terms: list[str],
        *,
        top_k: int,
        mode: str,
        debug_events: list[dict[str, object]],
    ) -> list[RetrievalHit]:
        codes = [code.upper() for code in FAULT_CODE_IN_QUERY.findall(query)]
        if codes:
            exact_hits = search_fault_codes_exact(cur, codes, top_k=top_k)
            if exact_hits:
                return exact_hits
            return self._search_channel(
                cur,
                query,
                terms,
                top_k=top_k,
                doc_types=["manual_fault_code"],
                mode=mode,
                channel_name="manual_fault_codes",
                debug_events=debug_events,
            )
        return []

    def _search_channel(
        self,
        cur: Any,
        query: str,
        terms: list[str],
        *,
        top_k: int,
        doc_types: list[str],
        mode: str,
        channel_name: str,
        debug_events: list[dict[str, object]],
    ) -> list[RetrievalHit]:
        candidate_top_k = max(top_k, self.config.retrieval.bm25_top_k)
        if channel_name == "work_orders":
            candidate_top_k = max(
                candidate_top_k,
                self.config.retrieval.work_order_candidate_top_k,
                self.config.retrieval.work_order_max_hits,
            )
        elif channel_name == "manual_typical_faults":
            candidate_top_k = max(
                candidate_top_k,
                self.config.retrieval.manual_candidate_top_k,
                self.config.retrieval.manual_max_hits,
            )
        bm25_hits = search_bm25(cur, terms, top_k=candidate_top_k, doc_types=doc_types)
        if mode != "hybrid" or self.embedding_provider is None:
            return self._finalize_channel_hits(
                bm25_hits,
                top_k=top_k,
                candidate_top_k=candidate_top_k,
                channel_name=channel_name,
                debug_events=debug_events,
            )

        try:
            query_vector = self.embedding_provider.embed_texts([query], text_type="query")[0]
        except EmbeddingProviderError as exc:
            debug_events.append(
                {
                    "channel": channel_name,
                    "stage": "query_embedding",
                    "status": "fallback",
                    "reason": str(exc),
                    "bm25_hits": len(bm25_hits),
                }
            )
            return self._finalize_channel_hits(
                bm25_hits,
                top_k=top_k,
                candidate_top_k=candidate_top_k,
                channel_name=channel_name,
                debug_events=debug_events,
            )
        try:
            vector_hits = search_vectors(
                cur,
                query_vector,
                provider=self.config.embedding.provider,
                model=self.config.embedding.model,
                top_k=max(candidate_top_k, self.config.retrieval.vector_top_k),
                doc_types=doc_types,
            )
        except Exception as exc:  # noqa: BLE001 - hybrid search can degrade to BM25.
            debug_events.append(
                {
                    "channel": channel_name,
                    "stage": "vector_search",
                    "status": "fallback",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "bm25_hits": len(bm25_hits),
                }
            )
            return self._finalize_channel_hits(
                bm25_hits,
                top_k=top_k,
                candidate_top_k=candidate_top_k,
                channel_name=channel_name,
                debug_events=debug_events,
            )
        debug_events.append(
            {
                "channel": channel_name,
                "stage": "vector_search",
                "status": "ok",
                "bm25_hits": len(bm25_hits),
                "vector_hits": len(vector_hits),
            }
        )
        merged_hits = merge_hybrid_hits(
            bm25_hits,
            vector_hits,
            top_k=candidate_top_k,
            bm25_weight=self.config.retrieval.hybrid_alpha,
        )
        return self._finalize_channel_hits(
            merged_hits,
            top_k=top_k,
            candidate_top_k=candidate_top_k,
            channel_name=channel_name,
            debug_events=debug_events,
        )

    def _finalize_channel_hits(
        self,
        hits: list[RetrievalHit],
        *,
        top_k: int,
        candidate_top_k: int,
        channel_name: str,
        debug_events: list[dict[str, object]],
    ) -> list[RetrievalHit]:
        """Apply channel-specific final ranking and limits."""

        if channel_name not in {"work_orders", "manual_typical_faults"}:
            return hits[:top_k]

        if channel_name == "work_orders":
            min_relative_score = self.config.retrieval.work_order_min_relative_score
            max_hits = self.config.retrieval.work_order_max_hits
        else:
            min_relative_score = self.config.retrieval.manual_min_relative_score
            max_hits = self.config.retrieval.manual_max_hits
        filtered = filter_hits_by_relative_threshold(hits, min_relative_score=min_relative_score, max_hits=max_hits)
        top_score = hits[0].score if hits else 0.0
        debug_events.append(
            {
                "channel": channel_name,
                "stage": "threshold_filter",
                "status": "ok",
                "candidate_count": len(hits),
                "returned_count": len(filtered),
                "candidate_top_k": candidate_top_k,
                "top_score": round(top_score, 6),
                "min_relative_score": min_relative_score,
                "score_threshold": round(top_score * min_relative_score, 6),
                "max_hits": max_hits,
            }
        )
        return filtered


class RagPipeline:
    """Run retrieval, evidence harness filtering, and answer generation."""

    def __init__(self, database: DatabaseOptions, config: AppConfig) -> None:
        """Store pipeline dependencies."""

        self.database = database
        self.config = config

    def run(
        self,
        query: str,
        *,
        top_k: int = 8,
        include_debug: bool = True,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> dict[str, object]:
        """Run the full RAG pipeline and return answer, evidence, and debug logs."""

        trace: list[dict[str, object]] = []
        started_at = time.time()
        trace.append(stage_event("config", "ok", {"config": self.config.to_dict()}))
        partial_payload: dict[str, object] = {
            "query": query,
            "rerank": {
                "enabled": self.config.rerank.enabled,
                "available": self.config.rerank.is_available(),
                "status": "skipped",
                "reason": "answer_harness_primary_path",
            },
            "trace": trace,
        }

        def publish(active_stage: str, summary: str) -> None:
            if progress_callback is None:
                return
            snapshot = dict(partial_payload)
            snapshot["trace"] = list(trace)
            snapshot["active_stage"] = active_stage
            snapshot["active_summary"] = summary
            snapshot["elapsed_seconds"] = round(time.time() - started_at, 3)
            progress_callback({"active_stage": active_stage, "summary": summary, "result": snapshot})

        def publish_harness(active_stage: str, summary: str, harness_payload: dict[str, object]) -> None:
            partial_payload["answer_harness"] = harness_payload
            final_context = dict_payload(harness_payload.get("final_answer_context"))
            if final_context:
                partial_payload["selected_evidence"] = list_payload(final_context.get("selected_evidence"))
            publish(active_stage, summary)

        def pause_pipeline(exc: PipelinePaused) -> None:
            summary = str(exc)
            paused_answer = {"status": "paused", "reason": exc.reason, "error": exc.error, "text": ""}
            trace.append(stage_event(exc.stage, "paused", {"reason": exc.reason, "error": exc.error}))
            partial_payload["answer"] = paused_answer
            partial_payload["trace"] = trace
            partial_payload["active_stage"] = exc.stage
            partial_payload["active_summary"] = summary
            partial_payload["elapsed_seconds"] = round(time.time() - started_at, 3)
            exc.partial_result = dict(partial_payload)
            if progress_callback is not None:
                progress_callback({"active_stage": exc.stage, "summary": summary, "result": dict(partial_payload)})

        publish("retrieval", "正在多路召回证据")
        retriever = PgRetriever(self.database, self.config)
        retrieve_started_at = time.time()
        retrieval = retriever.retrieve(query, top_k=top_k, include_debug=include_debug)
        trace.append(
            stage_event(
                "retrieval",
                "ok",
                {
                    "elapsed_ms": elapsed_ms(retrieve_started_at),
                    "mode": retrieval.get("mode"),
                    "summary": format_search_summary(retrieval),
                    "warnings": retrieval.get("warnings") or [],
                },
            )
        )

        evidence_items = flatten_evidence(retrieval, limit=max(self.config.answer.evidence_top_k, top_k))
        part_candidates = list_payload(retrieval.get("part_candidates"))
        partial_payload["retrieval"] = retrieval
        partial_payload["part_candidates"] = part_candidates
        partial_payload["selected_evidence"] = evidence_items
        publish("work_order_filter", "多路召回完成，正在筛选历史工单")
        try:
            answer_harness = self._run_answer_harness(
                query=query,
                retrieval=retrieval,
                part_candidates=part_candidates,
                trace=trace,
                progress_callback=publish_harness,
            )
            final_context = dict_payload(answer_harness.get("final_answer_context"))
            selected_evidence = list_payload(final_context.get("selected_evidence")) or evidence_items
            partial_payload["answer_harness"] = answer_harness
            partial_payload["selected_evidence"] = selected_evidence
            answer_payload = self._answer(query, final_context, trace)
        except PipelinePaused as exc:
            pause_pipeline(exc)
            raise
        partial_payload["answer"] = answer_payload
        payload: dict[str, object] = {
            "query": query,
            "answer": answer_payload,
            "retrieval": retrieval,
            "rerank": {
                "enabled": self.config.rerank.enabled,
                "available": self.config.rerank.is_available(),
                "status": "skipped",
                "reason": "answer_harness_primary_path",
            },
            "answer_harness": answer_harness,
            "selected_evidence": selected_evidence,
            "part_candidates": part_candidates,
            "trace": trace,
            "elapsed_seconds": round(time.time() - started_at, 3),
        }
        if include_debug:
            payload["debug"] = {
                "database_url": redact_database_url(self.database.database_url),
                "config": self.config.to_dict(),
            }
        partial_payload.update(payload)
        publish("completed", "答案生成完成")
        return payload

    def _run_answer_harness(
        self,
        *,
        query: str,
        retrieval: dict[str, object],
        part_candidates: list[dict[str, object]],
        trace: list[dict[str, object]],
        progress_callback: Callable[[str, str, dict[str, object]], None] | None = None,
    ) -> dict[str, object]:
        work_order_hits = retrieval_channel_items(retrieval, "work_orders")
        manual_hits = retrieval_channel_items(retrieval, "manual_typical_faults")
        fault_code_hits = retrieval_channel_items(retrieval, "manual_fault_codes")

        work_order_started_at = time.time()
        work_order_filter = self._filter_work_orders(query=query, work_order_hits=work_order_hits, part_candidates=part_candidates)
        trace.append(
            stage_event(
                "work_order_filter",
                str(work_order_filter.get("status") or "ok"),
                {
                    "elapsed_ms": elapsed_ms(work_order_started_at),
                    "accepted": len(list_payload(work_order_filter.get("accepted"))),
                    "rejected": len(list_payload(work_order_filter.get("rejected"))),
                    "unknown": len(list_payload(work_order_filter.get("unknown"))),
                    "reason": work_order_filter.get("reason"),
                },
            )
        )
        harness_progress: dict[str, object] = {"work_order_filter": work_order_filter}
        if progress_callback is not None:
            progress_callback("manual_filter", "工单筛选完成，正在筛选手册标题", dict(harness_progress))

        manual_started_at = time.time()
        manual_filter = self._filter_manuals(query=query, manual_hits=manual_hits)
        trace.append(
            stage_event(
                "manual_filter",
                str(manual_filter.get("status") or "ok"),
                {
                    "elapsed_ms": elapsed_ms(manual_started_at),
                    "selected": len(list_payload(manual_filter.get("selected"))),
                    "rejected": len(list_payload(manual_filter.get("rejected"))),
                    "reason": manual_filter.get("reason"),
                },
            )
        )
        harness_progress["manual_filter"] = manual_filter
        if progress_callback is not None:
            progress_callback("fact_extraction", "手册筛选完成，正在整理答案事实", dict(harness_progress))

        accepted_orders = list_payload(work_order_filter.get("accepted"))
        selected_manuals = list_payload(manual_filter.get("selected"))
        selected_parts = parts_from_accepted_orders(accepted_orders)
        selected_evidence_context = {
            "fault_code_evidence": fault_code_hits,
            "work_orders": accepted_orders,
            "manuals": selected_manuals,
        }

        facts_started_at = time.time()
        facts = self._extract_facts(
            query=query,
            selected_evidence=selected_evidence_context,
            selected_parts=selected_parts,
        )
        trace.append(
            stage_event(
                "fact_extraction",
                str(facts.get("status") or "ok"),
                {
                    "elapsed_ms": elapsed_ms(facts_started_at),
                    "fault_codes": len(list_payload(facts.get("fault_code_facts"))),
                    "work_order_groups": len(list_payload(facts.get("work_order_groups"))),
                    "manual_summaries": len(list_payload(facts.get("manual_summaries"))),
                    "coded_parts": len(list_payload(facts.get("coded_parts"))),
                    "reason": facts.get("reason"),
                },
            )
        )
        harness_progress["facts"] = facts

        final_answer_context = build_final_answer_context(
            query=query,
            fault_code_hits=fault_code_hits,
            accepted_orders=accepted_orders,
            selected_manuals=selected_manuals,
            facts=facts,
            selected_parts=selected_parts,
        )
        harness_progress["final_answer_context"] = final_answer_context
        if progress_callback is not None:
            progress_callback("answer", "事实整理完成，正在生成最终答案", dict(harness_progress))
        return {
            "work_order_filter": work_order_filter,
            "manual_filter": manual_filter,
            "facts": facts,
            "final_answer_context": final_answer_context,
        }

    def _filter_work_orders(
        self,
        *,
        query: str,
        work_order_hits: list[dict[str, object]],
        part_candidates: list[dict[str, object]],
    ) -> dict[str, object]:
        if not work_order_hits:
            return {"status": "skipped", "reason": "empty_work_order_hits", "accepted": [], "rejected": [], "unknown": []}
        if not self.config.llm.enabled or not self.config.llm.is_available():
            accepted = [
                work_order_filter_payload(
                    hit,
                    parts_for_work_order(part_candidates, work_order_id_from_hit(hit)),
                    relevance_level="medium",
                    matched_reason="LLM 未启用或不可用，保留检索召回工单作为 deterministic fallback。",
                )
                for hit in work_order_hits
            ]
            return {
                "status": "fallback",
                "reason": "llm_unavailable",
                "concurrency": 0,
                "accepted": accepted,
                "rejected": [],
                "unknown": [],
            }

        max_workers = min(DEFAULT_WORK_ORDER_FILTER_CONCURRENCY, MAX_WORK_ORDER_FILTER_CONCURRENCY, max(len(work_order_hits), 1))
        indexed_results: dict[int, dict[str, object]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(
                    self._filter_one_work_order,
                    query=query,
                    hit=hit,
                    linked_parts=parts_for_work_order(part_candidates, work_order_id_from_hit(hit)),
                ): index
                for index, hit in enumerate(work_order_hits)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    indexed_results[index] = future.result()
                except ModelProviderError as exc:
                    raise PipelinePaused(
                        stage="work_order_filter",
                        reason="llm_api_failed",
                        message=f"工单筛选 LLM API 调用失败，任务已暂停：{exc}",
                        error=str(exc),
                    ) from exc
                except Exception as exc:  # noqa: BLE001 - one bad LLM call must not break the harness.
                    indexed_results[index] = work_order_unknown_payload(work_order_hits[index], error=str(exc))

        accepted: list[dict[str, object]] = []
        rejected: list[dict[str, object]] = []
        unknown: list[dict[str, object]] = []
        for index in range(len(work_order_hits)):
            item = indexed_results.get(index) or work_order_unknown_payload(work_order_hits[index], error="missing_result")
            level = clean_string(item.get("relevance_level")).lower()
            if level in {"high", "medium"} and item.get("related") is not False:
                accepted.append(item)
            elif level == "unknown":
                unknown.append(item)
            else:
                rejected.append(item)
        return {
            "status": "ok",
            "concurrency": max_workers,
            "accepted": accepted,
            "rejected": rejected,
            "unknown": unknown,
        }

    def _filter_one_work_order(
        self,
        *,
        query: str,
        hit: dict[str, object],
        linked_parts: list[dict[str, object]],
    ) -> dict[str, object]:
        result = judge_work_order_relevance(
            query=query,
            work_order_hit=hit,
            linked_parts=linked_parts,
            config=self.config.llm,
        )
        try:
            parsed = parse_json_object(result.text)
        except Exception as exc:  # noqa: BLE001 - parsing failures become observable unknown items.
            payload = work_order_unknown_payload(hit, error=f"{type(exc).__name__}: {exc}", raw_text=result.text[:800])
            payload["debug"] = result.debug
            return payload
        return normalize_work_order_filter_result(parsed, hit, linked_parts, debug=result.debug)

    def _filter_manuals(self, *, query: str, manual_hits: list[dict[str, object]]) -> dict[str, object]:
        if not manual_hits:
            return {"status": "skipped", "reason": "empty_manual_hits", "selected": [], "rejected": []}
        if not self.config.llm.enabled or not self.config.llm.is_available():
            return {
                "status": "fallback",
                "reason": "llm_unavailable",
                "selected": [
                    manual_selection_payload(hit, relevance_level="medium", reason="LLM 未启用或不可用，保留召回手册作为补充证据。")
                    for hit in manual_hits
                ],
                "rejected": [],
            }
        try:
            result = select_manual_titles(query=query, manual_hits=manual_hits, config=self.config.llm)
        except ModelProviderError as exc:
            raise PipelinePaused(
                stage="manual_filter",
                reason="llm_api_failed",
                message=f"手册筛选 LLM API 调用失败，任务已暂停：{exc}",
                error=str(exc),
            ) from exc
        try:
            parsed = parse_json_object(result.text)
            payload = normalize_manual_filter_result(parsed, manual_hits)
            payload["status"] = "ok"
            payload["debug"] = result.debug
            return payload
        except Exception as exc:  # noqa: BLE001 - manual filtering must degrade.
            return {
                "status": "fallback",
                "reason": "manual_filter_failed",
                "error": f"{type(exc).__name__}: {exc}",
                "raw_text_preview": result.text[:800],
                "selected": [
                    manual_selection_payload(hit, relevance_level="medium", reason="手册筛选失败，保留召回手册作为补充证据。")
                    for hit in manual_hits
                ],
                "rejected": [],
            }

    def _extract_facts(
        self,
        *,
        query: str,
        selected_evidence: dict[str, object],
        selected_parts: list[dict[str, object]],
    ) -> dict[str, object]:
        fallback_facts = build_deterministic_facts(selected_evidence=selected_evidence, selected_parts=selected_parts)
        if not self.config.llm.enabled or not self.config.llm.is_available():
            fallback_facts["status"] = "fallback"
            fallback_facts["reason"] = "llm_unavailable"
            return fallback_facts
        try:
            result = extract_answer_facts(
                query=query,
                selected_evidence=selected_evidence,
                selected_parts=selected_parts,
                config=self.config.llm,
            )
        except ModelProviderError as exc:
            raise PipelinePaused(
                stage="fact_extraction",
                reason="llm_api_failed",
                message=f"事实整理 LLM API 调用失败，任务已暂停：{exc}",
                error=str(exc),
            ) from exc
        try:
            parsed = parse_json_object(result.text)
            facts = normalize_answer_facts(parsed, fallback=fallback_facts)
            facts["status"] = "ok"
            facts["debug"] = result.debug
            return facts
        except Exception as exc:  # noqa: BLE001 - answer facts must degrade.
            fallback_facts["status"] = "fallback"
            fallback_facts["reason"] = "fact_extraction_failed"
            fallback_facts["error"] = f"{type(exc).__name__}: {exc}"
            fallback_facts["raw_text_preview"] = result.text[:800]
            return fallback_facts

    def _rerank(
        self,
        query: str,
        evidence_items: list[dict[str, object]],
        trace: list[dict[str, object]],
    ) -> dict[str, object]:
        if not self.config.rerank.enabled:
            trace.append(stage_event("rerank", "skipped", {"reason": "disabled"}))
            return {"status": "skipped", "reason": "disabled", "evidence": evidence_items}
        if not self.config.rerank.is_available():
            trace.append(stage_event("rerank", "skipped", {"reason": "missing_config"}))
            return {"status": "skipped", "reason": "missing_config", "evidence": evidence_items}
        if not evidence_items:
            trace.append(stage_event("rerank", "skipped", {"reason": "empty_evidence"}))
            return {"status": "skipped", "reason": "empty_evidence", "evidence": evidence_items}

        started_at = time.time()
        docs = [build_rerank_document(item, char_limit=self.config.rerank.doc_char_limit) for item in evidence_items]
        try:
            rerank_results, debug = DashScopeRerankClient(self.config.rerank).rerank(
                query=query,
                documents=docs,
                top_n=min(self.config.rerank.top_n, len(docs)),
            )
            by_index = {item.index: item for item in rerank_results}
            reranked: list[dict[str, object]] = []
            for item in rerank_results:
                if 0 <= item.index < len(evidence_items):
                    reranked_item = dict(evidence_items[item.index])
                    reranked_item["rerank_score"] = item.score
                    reranked.append(reranked_item)
            if not reranked:
                reranked = evidence_items
            trace.append(
                stage_event(
                    "rerank",
                    "ok",
                    {
                        "elapsed_ms": elapsed_ms(started_at),
                        "model": self.config.rerank.model,
                        "input_count": len(evidence_items),
                        "returned_count": len(reranked),
                    },
                )
            )
            return {
                "status": "ok",
                "debug": debug,
                "results": [
                    {"index": index, "score": result.score, "doc_id": evidence_items[index].get("doc_id")}
                    for index, result in sorted(by_index.items())
                    if 0 <= index < len(evidence_items)
                ],
                "evidence": reranked,
            }
        except Exception as exc:  # noqa: BLE001 - rerank must degrade.
            trace.append(stage_event("rerank", "fallback", {"elapsed_ms": elapsed_ms(started_at), "error": str(exc)}))
            return {"status": "fallback", "error": str(exc), "evidence": evidence_items}

    def _answer(
        self,
        query: str,
        final_context: dict[str, object],
        trace: list[dict[str, object]],
    ) -> dict[str, object]:
        if not self.config.answer.enabled:
            trace.append(stage_event("answer", "skipped", {"reason": "disabled"}))
            return {"status": "skipped", "text": ""}
        if not self.config.llm.enabled:
            raise PipelinePaused(
                stage="answer",
                reason="llm_disabled",
                message="LLM 未启用，答案生成已暂停；启用 LLM 后可继续/重试。",
            )
        if not self.config.llm.is_available():
            raise PipelinePaused(
                stage="answer",
                reason="missing_llm_config",
                message="LLM 配置不完整，答案生成已暂停；补齐配置后可继续/重试。",
            )

        started_at = time.time()
        try:
            result = generate_harness_answer(
                query=query,
                final_context=final_context,
                config=self.config.llm,
            )
            text = result.text
            if not text:
                raise PipelinePaused(
                    stage="answer",
                    reason="empty_llm_response",
                    message="LLM 返回空答案，答案生成已暂停；可继续/重试。",
                )
            trace.append(
                stage_event(
                    "answer",
                    "ok",
                    {
                        "elapsed_ms": elapsed_ms(started_at),
                        "model": self.config.llm.model,
                        "usage": result.debug.get("usage"),
                    },
                )
            )
            return {"status": "ok", "text": text, "debug": result.debug}
        except PipelinePaused:
            raise
        except Exception as exc:  # noqa: BLE001 - answer generation failures pause for user retry.
            raise PipelinePaused(
                stage="answer",
                reason="llm_api_failed" if isinstance(exc, ModelProviderError) else "answer_generation_failed",
                message=f"答案生成 LLM 调用失败，任务已暂停：{exc}",
                error=str(exc),
            ) from exc


def run_pg_search(options: PgSearchOptions) -> dict[str, object]:
    """Run retrieval from CLI/web options and return a JSON payload."""

    config = load_config(
        options.config_path,
        overrides=options.config_overrides,
        env_path=options.env_path,
    )
    return PgRetriever(options.database, config).retrieve(
        options.query,
        top_k=max(options.top_k, 1),
        include_debug=options.include_debug,
    )


def run_pg_pipeline(options: PgPipelineOptions) -> dict[str, object]:
    """Run full retrieve, rerank, and answer generation."""

    config = load_config(
        options.config_path,
        overrides=options.config_overrides,
        env_path=options.env_path,
    )
    return RagPipeline(options.database, config).run(
        options.query,
        top_k=max(options.top_k, 1),
        include_debug=options.include_debug,
        progress_callback=options.progress_callback,
    )


def connect(database_url: str) -> Any:
    """Open a psycopg connection lazily so import errors are user-facing."""

    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - depends on environment.
        raise RuntimeError("psycopg is required. Install with: pip install -e .") from exc
    return psycopg.connect(database_url)


def create_extensions(cur: Any) -> None:
    """Create PostgreSQL extensions used by this pipeline."""

    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")


def clear_application_data(database: DatabaseOptions) -> dict[str, object]:
    """Clear indexed evidence and workbench task history while keeping the schema."""

    started_at = time.time()
    with connect(database.database_url) as conn:
        with conn.cursor() as cur:
            create_extensions(cur)
            create_schema(cur)
            before_counts = count_application_rows(cur)
            clear_application_data_with_cursor(cur)
        conn.commit()
    return {
        "database_url": redact_database_url(database.database_url),
        "cleared_tables": list(APPLICATION_DATA_TABLES),
        "before_counts": before_counts,
        "elapsed_seconds": round(time.time() - started_at, 3),
    }


def count_application_rows(cur: Any) -> dict[str, int]:
    """Count rows in application data tables before cleanup."""

    counts: dict[str, int] = {}
    for table_name in APPLICATION_DATA_TABLES:
        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        row = cur.fetchone()
        counts[table_name] = int(row[0]) if row else 0
    return counts


def clear_application_data_with_cursor(cur: Any) -> None:
    """Truncate application data tables using an existing cursor."""

    table_sql = ", ".join(APPLICATION_DATA_TABLES)
    cur.execute(f"TRUNCATE TABLE {table_sql} RESTART IDENTITY CASCADE")


def drop_schema_objects(cur: Any) -> None:
    """Drop application tables before a clean schema rebuild."""

    cur.execute(
        """
        DROP TABLE IF EXISTS
            document_embeddings,
            document_terms,
            document_fields,
            documents,
            manual_chunks,
            part_evidence,
            work_orders,
            ingest_items,
            ingest_runs,
            schema_migrations
        CASCADE
        """
    )


def create_task_schema(cur: Any) -> None:
    """Create persistent task-history tables used by the workbench."""

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS rag_tasks (
            id bigserial PRIMARY KEY,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now(),
            task_type text NOT NULL,
            status text NOT NULL,
            query text,
            summary text,
            request jsonb NOT NULL DEFAULT '{}'::jsonb,
            result jsonb NOT NULL DEFAULT '{}'::jsonb,
            error text
        );
        CREATE INDEX IF NOT EXISTS rag_tasks_created_at_idx ON rag_tasks(created_at DESC);
        CREATE INDEX IF NOT EXISTS rag_tasks_status_idx ON rag_tasks(status);
        CREATE INDEX IF NOT EXISTS rag_tasks_task_type_idx ON rag_tasks(task_type);
        CREATE INDEX IF NOT EXISTS rag_tasks_batch_eval_share_id_idx
            ON rag_tasks((result->>'share_id'))
            WHERE task_type = 'batch_eval';
        """
    )


def create_schema(cur: Any) -> None:
    """Create tables and indexes for evidence storage and retrieval."""

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version integer PRIMARY KEY,
            applied_at timestamptz NOT NULL DEFAULT now()
        );

        CREATE TABLE IF NOT EXISTS ingest_runs (
            id bigserial PRIMARY KEY,
            started_at timestamptz NOT NULL DEFAULT now(),
            completed_at timestamptz,
            status text NOT NULL DEFAULT 'running',
            config jsonb NOT NULL DEFAULT '{}'::jsonb,
            counts jsonb NOT NULL DEFAULT '{}'::jsonb,
            warnings jsonb NOT NULL DEFAULT '[]'::jsonb
        );

        CREATE TABLE IF NOT EXISTS work_orders (
            id bigserial PRIMARY KEY,
            work_order_id text,
            reported_issue text,
            solution text,
            remarks text,
            raw_text text,
            source_path text NOT NULL UNIQUE,
            encoding text,
            parse_warnings jsonb NOT NULL DEFAULT '[]'::jsonb,
            updated_at timestamptz NOT NULL DEFAULT now()
        );
        CREATE INDEX IF NOT EXISTS work_orders_work_order_id_idx
            ON work_orders(work_order_id);

        CREATE TABLE IF NOT EXISTS ingest_items (
            id bigserial PRIMARY KEY,
            source_kind text NOT NULL,
            source_path text NOT NULL,
            content_hash text NOT NULL DEFAULT '',
            status text NOT NULL,
            started_at timestamptz,
            completed_at timestamptz,
            updated_at timestamptz NOT NULL DEFAULT now(),
            counts jsonb NOT NULL DEFAULT '{}'::jsonb,
            error text,
            UNIQUE(source_kind, source_path)
        );
        CREATE INDEX IF NOT EXISTS ingest_items_status_idx ON ingest_items(status);
        CREATE INDEX IF NOT EXISTS ingest_items_source_idx ON ingest_items(source_kind, source_path);

        CREATE TABLE IF NOT EXISTS part_evidence (
            id bigserial PRIMARY KEY,
            work_order_id text,
            reported_issue text,
            solution text,
            remarks text,
            part_number_name text,
            part_number text,
            part_name text,
            part_code text,
            quantity text,
            raw_text text,
            source_path text NOT NULL,
            updated_at timestamptz NOT NULL DEFAULT now()
        );
        CREATE INDEX IF NOT EXISTS part_evidence_work_order_id_idx ON part_evidence(work_order_id);
        CREATE INDEX IF NOT EXISTS part_evidence_part_code_idx ON part_evidence(part_code);

        CREATE TABLE IF NOT EXISTS manual_chunks (
            id bigserial PRIMARY KEY,
            doc_type text NOT NULL,
            machine_type text,
            manual_section text,
            system_dir text,
            file_name text,
            fault_title text,
            fault_code text,
            fault_description text,
            chunk_index integer NOT NULL,
            chunk_count integer NOT NULL,
            chunk_text text NOT NULL,
            source_path text NOT NULL,
            original_html_path text,
            metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
            updated_at timestamptz NOT NULL DEFAULT now()
        );
        CREATE INDEX IF NOT EXISTS manual_chunks_fault_code_idx ON manual_chunks(fault_code);
        CREATE INDEX IF NOT EXISTS manual_chunks_doc_type_idx ON manual_chunks(doc_type);

        CREATE TABLE IF NOT EXISTS documents (
            id bigserial PRIMARY KEY,
            doc_id text NOT NULL UNIQUE,
            doc_type text NOT NULL,
            title text NOT NULL DEFAULT '',
            body text NOT NULL DEFAULT '',
            work_order_id text,
            source_path text,
            metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
            updated_at timestamptz NOT NULL DEFAULT now()
        );
        CREATE INDEX IF NOT EXISTS documents_doc_type_idx ON documents(doc_type);
        CREATE INDEX IF NOT EXISTS documents_work_order_id_idx ON documents(work_order_id);
        CREATE INDEX IF NOT EXISTS documents_source_path_idx ON documents(source_path);

        CREATE TABLE IF NOT EXISTS document_fields (
            document_id bigint NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            field_name text NOT NULL,
            field_text text NOT NULL DEFAULT '',
            field_weight double precision NOT NULL DEFAULT 1.0,
            field_length integer NOT NULL DEFAULT 0,
            PRIMARY KEY(document_id, field_name)
        );

        CREATE TABLE IF NOT EXISTS document_terms (
            term text NOT NULL,
            document_id bigint NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            field_name text NOT NULL,
            tf integer NOT NULL,
            field_weight double precision NOT NULL DEFAULT 1.0,
            field_length integer NOT NULL DEFAULT 0,
            PRIMARY KEY(term, document_id, field_name)
        );
        CREATE INDEX IF NOT EXISTS document_terms_document_idx ON document_terms(document_id);
        CREATE INDEX IF NOT EXISTS document_terms_term_idx ON document_terms(term);

        CREATE TABLE IF NOT EXISTS document_embeddings (
            document_id bigint NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            provider text NOT NULL,
            model text NOT NULL,
            dimensions integer NOT NULL,
            embedding vector NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY(document_id, provider, model)
        );
        CREATE INDEX IF NOT EXISTS document_embeddings_provider_model_idx
            ON document_embeddings(provider, model);
        """
    )
    create_task_schema(cur)


def validate_ingest_inputs(
    work_order_dir: Path | None,
    manual_dir: Path | None,
    work_order_files: Sequence[Path] = (),
    manual_files: Sequence[Path] = (),
) -> None:
    """Validate ingest input directories and explicit retry files."""

    if work_order_dir is None and manual_dir is None and not work_order_files and not manual_files:
        raise ValueError("at least one of work_order_dir or manual_dir is required")
    for label, path in (("work_order_dir", work_order_dir), ("manual_dir", manual_dir)):
        if path is None:
            continue
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"{label} is not a directory: {path}")
    for label, paths in (("work_order_paths", work_order_files), ("manual_paths", manual_files)):
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"{label} item does not exist: {path}")
            if not path.is_file():
                raise IsADirectoryError(f"{label} item is not a file: {path}")


def resolve_work_order_input_files(root: Path | None, explicit_paths: Sequence[Path], *, limit: int | None) -> list[Path]:
    """Return work-order files from explicit retry paths or a scanned directory."""

    if explicit_paths:
        return unique_existing_paths(path for path in explicit_paths if path.suffix.lower() == ".txt")
    if root is None:
        return []
    return limited_paths(list(iter_txt_files(root)), limit)


def resolve_manual_input_files(root: Path | None, explicit_paths: Sequence[Path], *, limit: int | None) -> list[Path]:
    """Return manual files from explicit retry paths or a scanned directory."""

    if explicit_paths:
        return unique_existing_paths(path for path in explicit_paths if path.suffix.lower() in MANUAL_SUFFIXES)
    if root is None:
        return []
    return limited_paths(list(iter_manual_files(root)), limit)


def unique_existing_paths(paths: Iterable[Path]) -> list[Path]:
    """Return unique resolved paths while preserving input order."""

    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        result.append(resolved)
    return result


def source_root_for_path(path: Path, preferred_root: Path | None) -> Path:
    """Return a metadata root that contains the source path."""

    if preferred_root is not None:
        try:
            path.relative_to(preferred_root)
            return preferred_root
        except ValueError:
            pass
    return path.parent


def file_content_hash(path: Path) -> str:
    """Return a stable SHA-256 hash for one source file."""

    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_is_completed(cur: Any, source_kind: str, source_path: str, content_hash: str) -> bool:
    """Return whether one source file is already fully ingested."""

    cur.execute(
        """
        SELECT 1
        FROM ingest_items
        WHERE source_kind = %s AND source_path = %s AND content_hash = %s AND status = 'completed'
        LIMIT 1
        """,
        (source_kind, source_path, content_hash),
    )
    return cur.fetchone() is not None


def mark_source_running(cur: Any, source_kind: str, source_path: str, content_hash: str) -> None:
    """Mark a source file as currently being ingested."""

    cur.execute(
        """
        INSERT INTO ingest_items(source_kind, source_path, content_hash, status, started_at, completed_at, counts, error)
        VALUES (%s, %s, %s, 'running', now(), NULL, '{}'::jsonb, NULL)
        ON CONFLICT(source_kind, source_path) DO UPDATE SET
            content_hash = EXCLUDED.content_hash,
            status = 'running',
            started_at = now(),
            completed_at = NULL,
            updated_at = now(),
            counts = '{}'::jsonb,
            error = NULL
        """,
        (source_kind, source_path, content_hash),
    )


def mark_source_completed(cur: Any, completion: SourceCompletion) -> None:
    """Mark a source file as fully ingested."""

    cur.execute(
        """
        INSERT INTO ingest_items(source_kind, source_path, content_hash, status, started_at, completed_at, counts, error)
        VALUES (%s, %s, %s, 'completed', now(), now(), %s, NULL)
        ON CONFLICT(source_kind, source_path) DO UPDATE SET
            content_hash = EXCLUDED.content_hash,
            status = 'completed',
            completed_at = now(),
            updated_at = now(),
            counts = EXCLUDED.counts,
            error = NULL
        """,
        (completion.source_kind, completion.source_path, completion.content_hash, json_param(completion.counts)),
    )


def mark_source_failed(cur: Any, source_kind: str, source_path: str, content_hash: str, exc: Exception) -> None:
    """Mark a source file as failed without making it resumable as completed."""

    cur.execute(
        """
        INSERT INTO ingest_items(source_kind, source_path, content_hash, status, started_at, completed_at, counts, error)
        VALUES (%s, %s, %s, 'failed', now(), NULL, '{}'::jsonb, %s)
        ON CONFLICT(source_kind, source_path) DO UPDATE SET
            content_hash = EXCLUDED.content_hash,
            status = 'failed',
            updated_at = now(),
            completed_at = NULL,
            error = EXCLUDED.error
        """,
        (source_kind, source_path, content_hash, f"{type(exc).__name__}: {exc}"),
    )


def rollback_source_failure(cur: Any, source_kind: str, source_path: str, content_hash: str, exc: Exception) -> None:
    """Remove partial source records and persist failure metadata."""

    delete_source_records(cur, source_path)
    mark_source_failed(cur, source_kind, source_path, content_hash, exc)


def commit_cursor(cur: Any) -> None:
    """Commit through a psycopg cursor when a real connection is available."""

    connection = getattr(cur, "connection", None)
    commit = getattr(connection, "commit", None)
    if callable(commit):
        commit()


def create_ingest_run(cur: Any, config: AppConfig) -> int:
    """Create an ingest run row and return its ID."""

    cur.execute(
        "INSERT INTO ingest_runs(config) VALUES (%s) RETURNING id",
        (json_param(config.to_dict()),),
    )
    row = cur.fetchone()
    if row is None:
        raise RuntimeError("failed to create ingest run")
    return int(row[0])


def finish_ingest_run(cur: Any, run_id: int, report: PgIngestReport) -> None:
    """Mark an ingest run as complete with summary counts."""

    counts = ingest_report_counts(report)
    cur.execute(
        """
        UPDATE ingest_runs
        SET completed_at = now(), status = %s, counts = %s, warnings = %s
        WHERE id = %s
        """,
        ("completed_with_errors" if report.failed_items else "completed", json_param(counts), json_param(report.warnings), run_id),
    )


def ingest_report_counts(report: PgIngestReport) -> dict[str, int]:
    """Return the core numeric ingest counters."""

    return {
        "work_order_files": report.work_order_files,
        "work_orders": report.work_orders,
        "part_records": report.part_records,
        "manual_files": report.manual_files,
        "manual_chunks": report.manual_chunks,
        "total_documents": report.total_documents,
        "term_rows": report.term_rows,
        "embeddings": report.embeddings,
        "html_converted_in_memory": report.html_converted_in_memory,
        "skipped_files": report.skipped_files,
        "failed_items": len(report.failed_items),
    }


def add_timing(report: PgIngestReport | PgEmbeddingReport, key: str, elapsed_seconds: float) -> None:
    """Accumulate a positive timing value on an ingest report."""

    report.timing_seconds[key] = report.timing_seconds.get(key, 0.0) + max(0.0, elapsed_seconds)


def merge_timings(report: PgIngestReport, timings: dict[str, float]) -> None:
    """Merge timing counters into an ingest report."""

    for key, elapsed_seconds in timings.items():
        add_timing(report, key, elapsed_seconds)


def add_timing_value(timings: dict[str, float], key: str, elapsed_seconds: float) -> None:
    """Accumulate a positive timing value in a plain timing dictionary."""

    timings[key] = timings.get(key, 0.0) + max(0.0, elapsed_seconds)


def rounded_timings(timings: dict[str, float]) -> dict[str, float]:
    """Return timing counters rounded for stable JSON progress payloads."""

    return {key: round(value, 3) for key, value in timings.items()}


def delete_source_records(cur: Any, source_path: str) -> None:
    """Delete existing records for one source path before re-ingesting it."""

    cur.execute("DELETE FROM documents WHERE source_path = %s", (source_path,))
    cur.execute("DELETE FROM manual_chunks WHERE source_path = %s", (source_path,))
    cur.execute("DELETE FROM part_evidence WHERE source_path = %s", (source_path,))
    cur.execute("DELETE FROM work_orders WHERE source_path = %s", (source_path,))


def upsert_work_order(cur: Any, record: WorkOrderRecord) -> None:
    """Insert or replace one parsed work order."""

    cur.execute(
        """
        INSERT INTO work_orders(
            work_order_id, reported_issue, solution, remarks, raw_text, source_path, encoding, parse_warnings
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (source_path) DO UPDATE SET
            work_order_id = EXCLUDED.work_order_id,
            reported_issue = EXCLUDED.reported_issue,
            solution = EXCLUDED.solution,
            remarks = EXCLUDED.remarks,
            raw_text = EXCLUDED.raw_text,
            encoding = EXCLUDED.encoding,
            parse_warnings = EXCLUDED.parse_warnings,
            updated_at = now()
        """,
        (
            record.work_order_id,
            record.reported_issue,
            record.solution,
            record.remarks,
            record.raw_text,
            record.source_path,
            record.encoding,
            json_param(record.parse_warnings),
        ),
    )

    for part in record.parts:
        cur.execute(
            """
            INSERT INTO part_evidence(
                work_order_id, reported_issue, solution, remarks,
                part_number_name, part_number, part_name, part_code, quantity,
                raw_text, source_path
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                record.work_order_id,
                record.reported_issue,
                record.solution,
                record.remarks,
                part.part_number_name,
                part.part_number,
                part.part_name,
                part.part_code,
                part.quantity,
                part.raw_text,
                record.source_path,
            ),
        )


def build_documents_for_work_order(record: WorkOrderRecord) -> list[IndexDocument]:
    """Build searchable documents from one parsed work-order record."""

    work_order_id = clean_string(record.work_order_id)
    fallback_id = short_hash(f"{record.source_path}:{record.reported_issue}")
    fields = {
        "reported_issue": clean_string(record.reported_issue),
        "solution": clean_string(record.solution),
        "remarks": clean_string(record.remarks),
        "raw_text": clean_string(record.raw_text),
    }
    documents = [
        IndexDocument(
            doc_id=f"wo:{work_order_id or fallback_id}",
            doc_type="work_order",
            title=first_non_empty(record.reported_issue, work_order_id, Path(record.source_path).stem) or "",
            text=join_text(fields.values()),
            fields=fields,
            metadata={
                "work_order_id": work_order_id,
                "part_count": len(record.parts),
                "parse_warnings": record.parse_warnings,
                "encoding": record.encoding,
            },
            source_path=record.source_path,
        )
    ]
    for index, part in enumerate(record.parts, start=1):
        part_fields = {
            "reported_issue": clean_string(record.reported_issue),
            "solution": clean_string(record.solution),
            "remarks": clean_string(record.remarks),
            "part_number_name": clean_string(part.part_number_name),
            "part_number": clean_string(part.part_number),
            "part_name": clean_string(part.part_name),
            "part_code": clean_string(part.part_code),
            "quantity": clean_string(part.quantity),
            "raw_text": clean_string(part.raw_text),
        }
        documents.append(
            IndexDocument(
                doc_id=f"part:{work_order_id or fallback_id}:{index}",
                doc_type="part_evidence",
                title=first_non_empty(part.part_name, part.part_number_name, part.part_code, f"part {index}") or "",
                text=join_text(part_fields.values()),
                fields=part_fields,
                metadata={
                    "work_order_id": work_order_id,
                    "part_number_name": clean_string(part.part_number_name),
                    "part_number": clean_string(part.part_number),
                    "part_name": clean_string(part.part_name),
                    "part_code": clean_string(part.part_code),
                    "quantity": clean_string(part.quantity),
                    "part_index": index,
                },
                source_path=record.source_path,
            )
        )
    return documents


def read_manual_as_markdown(
    manual_path: Path,
    *,
    encodings: tuple[str, ...],
    converter: MarkdownConverter,
) -> tuple[str, str, bool, int]:
    """Read a manual file, converting HTML to Markdown in memory when needed."""

    text, encoding = read_text_with_fallback(manual_path, encodings)
    if manual_path.suffix.lower() in {".html", ".htm"}:
        return converter.convert(text), encoding, True, detect_table_count(text)
    return text, encoding, False, 0


def build_manual_document(
    *,
    manual_path: Path,
    root: Path,
    metadata: dict[str, object],
    chunk_text: str,
    chunk_index: int,
) -> IndexDocument:
    """Build a searchable document from one manual chunk."""

    relative_path = str(manual_path.relative_to(root))
    fields = {
        "fault_title": clean_string(metadata.get("fault_title")),
        "fault_code": clean_string(metadata.get("fault_code")),
        "fault_description": clean_string(metadata.get("fault_description")),
        "file_name": clean_string(metadata.get("file_name")),
        "path_text": " ".join(manual_path.relative_to(root).parts),
        "chunk_text": chunk_text,
    }
    return IndexDocument(
        doc_id=f"manual:{short_hash(relative_path)}:{chunk_index}",
        doc_type=str(metadata["doc_type"]),
        title=first_non_empty(fields["fault_title"], fields["fault_description"], manual_path.stem) or "",
        text=join_text(fields.values()),
        fields=fields,
        metadata=metadata,
        source_path=str(manual_path),
    )


def insert_manual_chunk(cur: Any, document: IndexDocument, metadata: dict[str, object]) -> None:
    """Insert one manual chunk metadata row."""

    cur.execute(
        """
        INSERT INTO manual_chunks(
            doc_type, machine_type, manual_section, system_dir, file_name,
            fault_title, fault_code, fault_description,
            chunk_index, chunk_count, chunk_text, source_path, original_html_path, metadata
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            document.doc_type,
            metadata.get("machine_type"),
            metadata.get("manual_section"),
            metadata.get("system_dir"),
            metadata.get("file_name"),
            metadata.get("fault_title"),
            metadata.get("fault_code"),
            metadata.get("fault_description"),
            metadata.get("chunk_index"),
            metadata.get("chunk_count"),
            document.fields.get("chunk_text", ""),
            document.source_path,
            document.source_path if metadata.get("converted_from_html") else None,
            json_param(metadata),
        ),
    )


def upsert_search_document(cur: Any, document: IndexDocument, *, write_bm25: bool = True) -> SearchDocumentUpsertResult:
    """Insert one searchable document and its BM25 term statistics."""

    timings: dict[str, float] = {}
    work_order_id = clean_string(document.metadata.get("work_order_id")) or None
    pg_started = time.perf_counter()
    cur.execute(
        """
        INSERT INTO documents(doc_id, doc_type, title, body, work_order_id, source_path, metadata)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (doc_id) DO UPDATE SET
            doc_type = EXCLUDED.doc_type,
            title = EXCLUDED.title,
            body = EXCLUDED.body,
            work_order_id = EXCLUDED.work_order_id,
            source_path = EXCLUDED.source_path,
            metadata = EXCLUDED.metadata,
            updated_at = now()
        RETURNING id
        """,
        (
            document.doc_id,
            document.doc_type,
            document.title,
            document.text,
            work_order_id,
            document.source_path,
            json_param(document.metadata),
        ),
    )
    row = cur.fetchone()
    if row is None:
        raise RuntimeError(f"failed to upsert document: {document.doc_id}")
    document_id = int(row[0])
    cur.execute("DELETE FROM document_fields WHERE document_id = %s", (document_id,))
    cur.execute("DELETE FROM document_terms WHERE document_id = %s", (document_id,))
    cur.execute("DELETE FROM document_embeddings WHERE document_id = %s", (document_id,))
    add_timing_value(timings, "pg_write_seconds", time.perf_counter() - pg_started)

    term_rows = 0
    bm25_started = time.perf_counter()
    field_rows: list[tuple[object, ...]] = []
    term_row_values: list[tuple[object, ...]] = []
    for field_name, field_text in document.fields.items():
        terms = tokenize_text(field_text)
        field_length = len(terms)
        field_weight = float(FIELD_WEIGHTS.get(field_name, 1.0))
        field_rows.append((document_id, field_name, clean_string(field_text), field_weight, field_length))
        for term, tf in Counter(terms).items():
            term_row_values.append((term, document_id, field_name, tf, field_weight, field_length))
            term_rows += 1
    if write_bm25:
        insert_bm25_rows(cur, field_rows=field_rows, term_rows=term_row_values)
    add_timing_value(timings, "bm25_seconds", time.perf_counter() - bm25_started)
    return SearchDocumentUpsertResult(
        document_id=document_id,
        term_rows=term_rows,
        timings=timings,
        field_rows=field_rows,
        term_row_values=term_row_values,
    )


def insert_bm25_rows(cur: Any, *, field_rows: list[tuple[object, ...]], term_rows: list[tuple[object, ...]]) -> None:
    """Bulk insert BM25 field and term statistics."""

    bulk_insert_rows(
        cur,
        copy_sql=(
            "COPY document_fields(document_id, field_name, field_text, field_weight, field_length) "
            "FROM STDIN"
        ),
        insert_sql=(
            "INSERT INTO document_fields(document_id, field_name, field_text, field_weight, field_length) "
            "VALUES (%s, %s, %s, %s, %s)"
        ),
        rows=field_rows,
    )
    bulk_insert_rows(
        cur,
        copy_sql=(
            "COPY document_terms(term, document_id, field_name, tf, field_weight, field_length) "
            "FROM STDIN"
        ),
        insert_sql=(
            "INSERT INTO document_terms(term, document_id, field_name, tf, field_weight, field_length) "
            "VALUES (%s, %s, %s, %s, %s, %s)"
        ),
        rows=term_rows,
    )


def bulk_insert_rows(cur: Any, *, copy_sql: str, insert_sql: str, rows: list[tuple[object, ...]]) -> None:
    """Bulk insert simple rows with PostgreSQL COPY, falling back to executemany for test doubles."""

    if not rows:
        return
    copy_method = getattr(cur, "copy", None)
    if callable(copy_method):
        with copy_method(copy_sql) as copy:
            for row in rows:
                copy.write_row(row)
        return
    cur.executemany(insert_sql, rows)


def maybe_store_embedding(
    cur: Any,
    document_id: int,
    document: IndexDocument,
    config: AppConfig,
    provider: CommandEmbeddingProvider | OpenAICompatibleEmbeddingProvider | None,
    report: PgIngestReport | PgEmbeddingReport,
) -> bool:
    """Store an optional embedding for one document when configured."""

    return store_embedding_batch(cur, [(document_id, document)], config, provider, report) == 1


def count_missing_embeddings(cur: Any, config: AppConfig, *, limit: int | None = None) -> int:
    """Count documents that do not yet have embeddings for the configured provider/model."""

    params: list[object] = [config.embedding.provider, config.embedding.model]
    limit_clause = ""
    if limit is not None:
        limit_clause = "LIMIT %s"
        params.append(max(limit, 0))
    cur.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT d.id
            FROM documents d
            LEFT JOIN document_embeddings e
                ON e.document_id = d.id AND e.provider = %s AND e.model = %s
            WHERE e.document_id IS NULL
            ORDER BY d.id ASC
            {limit_clause}
        ) missing
        """,
        params,
    )
    row = cur.fetchone()
    return int(row[0]) if row else 0


def fetch_missing_embedding_documents(
    cur: Any,
    config: AppConfig,
    *,
    limit: int,
) -> list[tuple[int, IndexDocument]]:
    """Fetch documents missing embeddings for the configured provider/model."""

    cur.execute(
        """
        SELECT d.id, d.doc_id, d.doc_type, d.title, d.body, d.work_order_id, d.source_path, d.metadata
        FROM documents d
        LEFT JOIN document_embeddings e
            ON e.document_id = d.id AND e.provider = %s AND e.model = %s
        WHERE e.document_id IS NULL
        ORDER BY d.id ASC
        LIMIT %s
        """,
        (config.embedding.provider, config.embedding.model, max(limit, 1)),
    )
    documents: list[tuple[int, IndexDocument]] = []
    for row in cur.fetchall():
        document_id, doc_id, doc_type, title, body, work_order_id, source_path, metadata = row
        document_metadata = dict(metadata or {}) if isinstance(metadata, dict) else {}
        if work_order_id and "work_order_id" not in document_metadata:
            document_metadata["work_order_id"] = work_order_id
        documents.append(
            (
                int(document_id),
                IndexDocument(
                    doc_id=str(doc_id),
                    doc_type=str(doc_type),
                    title=str(title or ""),
                    text=str(body or ""),
                    fields={"body": str(body or "")},
                    metadata=document_metadata,
                    source_path=str(source_path or ""),
                ),
            )
        )
    return documents


def store_embedding_batch(
    cur: Any,
    items: list[tuple[int, IndexDocument]],
    config: AppConfig,
    provider: CommandEmbeddingProvider | OpenAICompatibleEmbeddingProvider | None,
    report: PgIngestReport | PgEmbeddingReport,
) -> int:
    """Embed and store a batch of documents, falling back to single-item retries on provider errors."""

    if provider is None:
        return 0
    if not items:
        return 0
    texts = [f"{document.title}\n{document.text}" for _, document in items]
    try:
        embedding_started = time.perf_counter()
        vectors = provider.embed_texts(texts)
        add_timing(report, "embedding_seconds", time.perf_counter() - embedding_started)
    except EmbeddingProviderError as exc:
        report.warnings.append(f"embedding_batch_failed:{len(items)}: {exc}")
        return store_embedding_items_individually(cur, items, config, provider, report)

    pg_started = time.perf_counter()
    for (document_id, _document), vector in zip(items, vectors, strict=True):
        insert_document_embedding(cur, document_id, config, vector)
    add_timing(report, "pg_write_seconds", time.perf_counter() - pg_started)
    return len(items)


def store_embedding_items_individually(
    cur: Any,
    items: list[tuple[int, IndexDocument]],
    config: AppConfig,
    provider: CommandEmbeddingProvider | OpenAICompatibleEmbeddingProvider,
    report: PgIngestReport | PgEmbeddingReport,
) -> int:
    """Retry document embeddings one by one after a failed batch request."""

    stored_count = 0
    for document_id, document in items:
        try:
            embedding_started = time.perf_counter()
            vector = provider.embed_texts([f"{document.title}\n{document.text}"])[0]
            add_timing(report, "embedding_seconds", time.perf_counter() - embedding_started)
        except EmbeddingProviderError as exc:
            report.warnings.append(f"embedding_failed:{document.doc_id}: {exc}")
            continue
        pg_started = time.perf_counter()
        insert_document_embedding(cur, document_id, config, vector)
        add_timing(report, "pg_write_seconds", time.perf_counter() - pg_started)
        stored_count += 1
    return stored_count


def insert_document_embedding(cur: Any, document_id: int, config: AppConfig, vector: list[float]) -> None:
    """Insert or update one pgvector embedding row."""

    cur.execute(
        """
        INSERT INTO document_embeddings(document_id, provider, model, dimensions, embedding)
        VALUES (%s, %s, %s, %s, %s::vector)
        ON CONFLICT(document_id, provider, model) DO UPDATE SET
            dimensions = EXCLUDED.dimensions,
            embedding = EXCLUDED.embedding,
            created_at = now()
        """,
        (document_id, config.embedding.provider, config.embedding.model, len(vector), vector_literal(vector)),
    )


def search_bm25(
    cur: Any,
    terms: list[str],
    *,
    top_k: int,
    doc_types: list[str] | None = None,
) -> list[RetrievalHit]:
    """Search documents with the standard BM25 scoring formula in SQL."""

    if not terms:
        return []
    values_sql = ", ".join(["(%s)"] * len(terms))
    doc_type_clause = " AND d.doc_type = ANY(%s::text[])" if doc_types else ""
    params: list[object] = [*terms]
    if doc_types:
        params.extend([doc_types, doc_types, doc_types])
    params.extend([DEFAULT_BM25_K1, DEFAULT_BM25_K1, DEFAULT_BM25_B, DEFAULT_BM25_B, top_k])
    cur.execute(
        f"""
        WITH query_terms(term) AS (VALUES {values_sql}),
        doc_count AS (
            SELECT GREATEST(COUNT(*), 1)::double precision AS n
            FROM documents d
            WHERE 1 = 1 {doc_type_clause}
        ),
        avg_len AS (
            SELECT field_name, GREATEST(AVG(GREATEST(field_length, 1)), 1)::double precision AS avgdl
            FROM document_fields
            GROUP BY field_name
        ),
        df AS (
            SELECT dt.term, GREATEST(COUNT(DISTINCT dt.document_id), 1)::double precision AS df
            FROM document_terms dt
            JOIN documents d ON d.id = dt.document_id
            WHERE dt.term IN (SELECT term FROM query_terms) {doc_type_clause}
            GROUP BY dt.term
        ),
        matched AS (
            SELECT
                dt.document_id,
                dt.term,
                dt.field_name,
                dt.tf::double precision AS tf,
                dt.field_weight::double precision AS field_weight,
                GREATEST(dt.field_length, 1)::double precision AS field_length,
                COALESCE(avg_len.avgdl, 1)::double precision AS avgdl,
                df.df
            FROM document_terms dt
            JOIN query_terms q ON q.term = dt.term
            JOIN documents d ON d.id = dt.document_id
            JOIN df ON df.term = dt.term
            LEFT JOIN avg_len ON avg_len.field_name = dt.field_name
            WHERE 1 = 1 {doc_type_clause}
        )
        SELECT
            d.id, d.doc_id, d.doc_type, d.title, d.body, d.work_order_id, d.source_path, d.metadata,
            SUM(
                m.field_weight
                * LN(1 + ((doc_count.n - m.df + 0.5) / (m.df + 0.5)))
                * ((m.tf * (%s + 1)) / (m.tf + %s * (1 - %s + %s * (m.field_length / m.avgdl))))
            ) AS score,
            JSONB_AGG(DISTINCT JSONB_BUILD_OBJECT(
                'term', m.term,
                'field', m.field_name,
                'tf', m.tf,
                'field_weight', m.field_weight
            )) AS matched_terms
        FROM matched m
        JOIN documents d ON d.id = m.document_id
        CROSS JOIN doc_count
        GROUP BY d.id, d.doc_id, d.doc_type, d.title, d.body, d.work_order_id, d.source_path, d.metadata
        ORDER BY score DESC, d.id ASC
        LIMIT %s
        """,
        params,
    )
    return [row_to_hit(row) for row in cur.fetchall()]


def search_vectors(
    cur: Any,
    query_vector: list[float],
    *,
    provider: str,
    model: str,
    top_k: int,
    doc_types: list[str] | None = None,
) -> list[RetrievalHit]:
    """Search pgvector embeddings by cosine distance."""

    doc_type_clause = " AND d.doc_type = ANY(%s::text[])" if doc_types else ""
    query_vector_literal = vector_literal(query_vector)
    params: list[object] = [query_vector_literal, query_vector_literal, provider, model, len(query_vector)]
    if doc_types:
        params.append(doc_types)
    params.extend([query_vector_literal, top_k])
    cur.execute(
        f"""
        SELECT
            d.id, d.doc_id, d.doc_type, d.title, d.body, d.work_order_id, d.source_path, d.metadata,
            1.0 / (1.0 + (e.embedding <=> %s::vector)) AS score,
            '[]'::jsonb AS matched_terms,
            (e.embedding <=> %s::vector) AS vector_distance
        FROM document_embeddings e
        JOIN documents d ON d.id = e.document_id
        WHERE e.provider = %s AND e.model = %s AND e.dimensions = %s {doc_type_clause}
        ORDER BY e.embedding <=> %s::vector ASC
        LIMIT %s
        """,
        params,
    )
    return [row_to_hit(row, has_vector_distance=True) for row in cur.fetchall()]


def search_fault_codes_exact(cur: Any, codes: list[str], *, top_k: int) -> list[RetrievalHit]:
    """Search fault-code manual chunks by exact code before broad BM25 fallback."""

    cur.execute(
        """
        SELECT
            id, doc_id, doc_type, title, body, work_order_id, source_path, metadata,
            100.0 AS score,
            JSONB_BUILD_ARRAY(JSONB_BUILD_OBJECT('term', metadata->>'fault_code', 'field', 'fault_code', 'tf', 1)) AS matched_terms
        FROM documents
        WHERE doc_type = 'manual_fault_code'
          AND UPPER(metadata->>'fault_code') = ANY(%s::text[])
        ORDER BY id ASC
        LIMIT %s
        """,
        (codes, top_k),
    )
    return [row_to_hit(row) for row in cur.fetchall()]


def fetch_part_candidates(cur: Any, work_order_ids: list[str]) -> list[dict[str, object]]:
    """Fetch every part row linked to retrieved work orders."""

    if not work_order_ids:
        return []
    cur.execute(
        """
        SELECT
            work_order_id,
            reported_issue,
            part_number_name,
            part_number,
            part_name,
            part_code,
            quantity,
            source_path
        FROM part_evidence
        WHERE work_order_id = ANY(%s::text[])
        ORDER BY array_position(%s::text[], work_order_id), id ASC
        """,
        (work_order_ids, work_order_ids),
    )
    keys = (
        "work_order_id",
        "reported_issue",
        "part_number_name",
        "part_number",
        "part_name",
        "part_code",
        "quantity",
        "source_path",
    )
    return [part_candidate_payload(dict(zip(keys, row, strict=False))) for row in cur.fetchall()]


def part_candidate_payload(part: dict[str, object]) -> dict[str, object]:
    """Return a structured part payload shared by candidate and route displays."""

    payload = dict(part)
    payload["doc_type"] = "part_evidence"
    payload["channel"] = "part_evidence"
    payload["metadata"] = {
        "work_order_id": clean_string(part.get("work_order_id")),
        "reported_issue": clean_string(part.get("reported_issue")),
        "part_number_name": clean_string(part.get("part_number_name")),
        "part_number": clean_string(part.get("part_number")),
        "part_name": clean_string(part.get("part_name")),
        "part_code": clean_string(part.get("part_code")),
        "quantity": clean_string(part.get("quantity")),
    }
    return payload


def part_candidate_to_channel_hit(part: dict[str, object], *, rank: int) -> dict[str, object]:
    """Convert a linked part candidate into a retrieval-channel evidence item."""

    work_order_id = clean_string(part.get("work_order_id"))
    part_name = clean_string(part.get("part_name") or part.get("part_number_name"))
    part_code = clean_string(part.get("part_code"))
    quantity = clean_string(part.get("quantity"))
    title = part_name or part_code or f"备件证据 {rank}"
    body_preview = join_text(
        [
            f"新件备件名称: {part_name}" if part_name else "",
            f"新件物料编码: {part_code}" if part_code else "",
            f"新件数量: {quantity}" if quantity else "",
            f"来源工单: {work_order_id}" if work_order_id else "",
            f"用户报修内容: {clean_string(part.get('reported_issue'))}" if clean_string(part.get("reported_issue")) else "",
        ]
    )
    payload = part_candidate_payload(part)
    payload.update(
        {
            "doc_id": f"part:{work_order_id or 'unknown'}:{rank}",
            "title": title,
            "score": None,
            "body_preview": body_preview,
            "matched_terms": [],
        }
    )
    return payload


def enrich_part_hit_fields(cur: Any, hits: list[RetrievalHit]) -> None:
    """Merge structured part fields into retrieved part-evidence hit metadata."""

    part_hits = [hit for hit in hits if hit.doc_type == "part_evidence"]
    if not part_hits:
        return
    document_ids = [hit.document_id for hit in part_hits]
    cur.execute(
        """
        SELECT document_id, field_name, field_text
        FROM document_fields
        WHERE document_id = ANY(%s::bigint[])
          AND field_name = ANY(%s::text[])
        """,
        (document_ids, ["part_number_name", "part_number", "part_name", "part_code", "quantity"]),
    )
    fields_by_document: dict[int, dict[str, str]] = {}
    for document_id, field_name, field_text in cur.fetchall():
        text = clean_string(field_text)
        if text:
            fields_by_document.setdefault(int(document_id), {})[str(field_name)] = text
    for hit in part_hits:
        fields = fields_by_document.get(hit.document_id, {})
        for key, value in fields.items():
            if value and not hit.metadata.get(key):
                hit.metadata[key] = value


def merge_hybrid_hits(
    bm25_hits: list[RetrievalHit],
    vector_hits: list[RetrievalHit],
    *,
    top_k: int,
    bm25_weight: float,
) -> list[RetrievalHit]:
    """Merge BM25 and vector rankings with min-max normalized scores."""

    bm25_weight = min(max(float(bm25_weight), 0.0), 1.0)
    vector_weight = 1.0 - bm25_weight
    max_bm25 = max((hit.score for hit in bm25_hits), default=0.0) or 1.0
    max_vector = max((hit.score for hit in vector_hits), default=0.0) or 1.0
    merged: dict[str, RetrievalHit] = {}
    scores: dict[str, float] = {}

    for hit in bm25_hits:
        merged[hit.doc_id] = hit
        scores[hit.doc_id] = scores.get(hit.doc_id, 0.0) + bm25_weight * (hit.score / max_bm25)
    for hit in vector_hits:
        if hit.doc_id not in merged:
            merged[hit.doc_id] = hit
        else:
            merged[hit.doc_id].vector_distance = hit.vector_distance
        scores[hit.doc_id] = scores.get(hit.doc_id, 0.0) + vector_weight * (hit.score / max_vector)

    ordered = sorted(merged.values(), key=lambda hit: (-scores.get(hit.doc_id, 0.0), hit.document_id))
    for hit in ordered:
        hit.score = round(scores.get(hit.doc_id, 0.0), 6)
    return ordered[:top_k]


def row_to_hit(row: Sequence[Any], *, has_vector_distance: bool = False) -> RetrievalHit:
    """Convert one database row into a retrieval hit."""

    metadata = row[7] if isinstance(row[7], dict) else {}
    matched_terms = row[9] if isinstance(row[9], list) else []
    vector_distance = float(row[10]) if has_vector_distance and row[10] is not None else None
    return RetrievalHit(
        document_id=int(row[0]),
        doc_id=str(row[1]),
        doc_type=str(row[2]),
        title=str(row[3] or ""),
        score=round(float(row[8] or 0.0), 6),
        body_preview=preview_text(str(row[4] or "")),
        work_order_id=str(row[5]) if row[5] else None,
        source_path=str(row[6]) if row[6] else None,
        metadata=metadata,
        matched_terms=matched_terms,
        vector_distance=vector_distance,
    )


def collect_work_order_ids(*hit_groups: list[RetrievalHit]) -> list[str]:
    """Collect stable work-order IDs from retrieved hits."""

    ids: list[str] = []
    seen: set[str] = set()
    for hits in hit_groups:
        for hit in hits:
            work_order_id = clean_string(hit.work_order_id or hit.metadata.get("work_order_id"))
            if work_order_id and work_order_id not in seen:
                seen.add(work_order_id)
                ids.append(work_order_id)
    return ids


def filter_work_order_hits_by_threshold(
    hits: list[RetrievalHit],
    *,
    min_relative_score: float,
    max_hits: int,
) -> list[RetrievalHit]:
    """Keep work-order hits whose scores are close enough to the best candidate."""

    return filter_hits_by_relative_threshold(hits, min_relative_score=min_relative_score, max_hits=max_hits)


def filter_hits_by_relative_threshold(
    hits: list[RetrievalHit],
    *,
    min_relative_score: float,
    max_hits: int,
) -> list[RetrievalHit]:
    """Keep ranked hits whose scores are close enough to the best candidate."""

    if not hits or max_hits <= 0:
        return []
    top_score = hits[0].score
    if top_score <= 0:
        return []
    safe_relative_score = min(max(float(min_relative_score), 0.0), 1.0)
    score_threshold = top_score * safe_relative_score
    return [hit for hit in hits if hit.score >= score_threshold][:max_hits]


def list_payload(value: object) -> list[dict[str, object]]:
    """Return a list of dict payloads from JSON-like data."""

    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def dict_payload(value: object) -> dict[str, object]:
    """Return a dict payload from JSON-like data."""

    return value if isinstance(value, dict) else {}


def retrieval_channel_items(payload: dict[str, object], channel_name: str) -> list[dict[str, object]]:
    """Return copied hit dictionaries for one retrieval channel."""

    channels = payload.get("channels")
    if not isinstance(channels, dict):
        return []
    hits = channels.get(channel_name)
    if not isinstance(hits, list):
        return []
    return [dict(item) for item in hits if isinstance(item, dict)]


def work_order_id_from_hit(hit: dict[str, object]) -> str:
    """Extract a stable work-order ID from a retrieval hit."""

    metadata = hit.get("metadata") if isinstance(hit.get("metadata"), dict) else {}
    return clean_string(hit.get("work_order_id") or metadata.get("work_order_id") or hit.get("doc_id"))


def parts_for_work_order(parts: list[dict[str, object]], work_order_id: str) -> list[dict[str, object]]:
    """Return part candidates linked to one work order."""

    if not work_order_id:
        return []
    linked: list[dict[str, object]] = []
    for part in parts:
        metadata = part.get("metadata") if isinstance(part.get("metadata"), dict) else {}
        part_order_id = clean_string(part.get("work_order_id") or metadata.get("work_order_id"))
        if part_order_id == work_order_id:
            linked.append(part)
    return linked


def parts_for_filtered_orders(parts: list[dict[str, object]], accepted_orders: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return all parts linked to accepted work orders in work-order order."""

    selected: list[dict[str, object]] = []
    seen_rows: set[tuple[str, str, str, str]] = set()
    for order in accepted_orders:
        work_order_id = clean_string(order.get("work_order_id"))
        for part in parts_for_work_order(parts, work_order_id):
            compact = part_for_answer(part)
            row_key = (
                clean_string(compact.get("source_work_order_id") or compact.get("work_order_id")),
                clean_string(compact.get("name")),
                clean_string(compact.get("code")),
                clean_string(compact.get("quantity")),
            )
            if row_key in seen_rows:
                continue
            seen_rows.add(row_key)
            selected.append(part)
    return selected


def part_for_answer(part: dict[str, object]) -> dict[str, object]:
    """Normalize one part row for answer and harness displays."""

    metadata = part.get("metadata") if isinstance(part.get("metadata"), dict) else {}
    name = clean_string(
        part.get("name")
        or part.get("part_name")
        or part.get("part_number_name")
        or metadata.get("part_name")
        or metadata.get("part_number_name")
    )
    code = clean_string(part.get("code") or part.get("part_code") or part.get("part_number") or metadata.get("part_code") or metadata.get("part_number"))
    quantity = clean_string(part.get("quantity") or metadata.get("quantity"))
    work_order_id = clean_string(part.get("source_work_order_id") or part.get("work_order_id") or metadata.get("work_order_id"))
    return {
        "name": name,
        "code": code,
        "quantity": quantity,
        "source_work_order_id": work_order_id,
        "work_order_id": work_order_id,
        "source_path": clean_string(part.get("source_path")),
    }


def parts_from_accepted_orders(accepted_orders: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return usable parts selected by work-order filtering."""

    selected: list[dict[str, object]] = []
    seen_rows: set[tuple[str, str, str, str]] = set()
    for order in accepted_orders:
        work_order_id = clean_string(order.get("work_order_id"))
        source_path = clean_string(order.get("source_path"))
        for item in list_payload(order.get("usable_parts")):
            normalized = part_for_answer(item)
            if not normalized.get("source_work_order_id"):
                normalized["source_work_order_id"] = work_order_id
                normalized["work_order_id"] = work_order_id
            if source_path and not normalized.get("source_path"):
                normalized["source_path"] = source_path
            row_key = (
                clean_string(normalized.get("source_work_order_id")),
                clean_string(normalized.get("name")),
                clean_string(normalized.get("code")),
                clean_string(normalized.get("quantity")),
            )
            if row_key in seen_rows or not any(row_key):
                continue
            seen_rows.add(row_key)
            selected.append(normalized)
    return selected


def coded_parts_from_parts(parts: list[dict[str, object]]) -> list[dict[str, object]]:
    """Build a deduplicated coded-part list from selected structured parts."""

    coded: list[dict[str, object]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for part in parts:
        normalized = part_for_answer(part)
        if not normalized.get("code"):
            continue
        row_key = (
            clean_string(normalized.get("name")),
            clean_string(normalized.get("code")),
            clean_string(normalized.get("quantity")),
            clean_string(normalized.get("source_work_order_id")),
        )
        if row_key in seen:
            continue
        seen.add(row_key)
        coded.append(normalized)
    return coded


def work_order_filter_payload(
    hit: dict[str, object],
    linked_parts: list[dict[str, object]],
    *,
    relevance_level: str,
    matched_reason: str,
    repair_actions: list[str] | None = None,
) -> dict[str, object]:
    """Build one normalized work-order filter item."""

    level = normalize_relevance_level(relevance_level)
    work_order_id = work_order_id_from_hit(hit)
    usable_parts = [part_for_answer(part) for part in linked_parts if any(part_for_answer(part).values())]
    return {
        "work_order_id": work_order_id,
        "related": level in {"high", "medium"},
        "relevance_level": level,
        "matched_reason": matched_reason,
        "repair_actions": repair_actions or [],
        "usable_parts": usable_parts,
        "source_path": clean_string(hit.get("source_path")),
        "title": clean_string(hit.get("title")),
        "score": hit.get("score"),
        "body_preview": clean_string(hit.get("body_preview")),
        "hit": hit,
    }


def work_order_unknown_payload(hit: dict[str, object], *, error: str, raw_text: str = "") -> dict[str, object]:
    """Build an unknown work-order filter item for failed model calls."""

    return {
        "work_order_id": work_order_id_from_hit(hit),
        "related": False,
        "relevance_level": "unknown",
        "matched_reason": "工单筛选失败，默认不进入最终答案。",
        "repair_actions": [],
        "usable_parts": [],
        "source_path": clean_string(hit.get("source_path")),
        "title": clean_string(hit.get("title")),
        "score": hit.get("score"),
        "body_preview": clean_string(hit.get("body_preview")),
        "hit": hit,
        "error": error,
        "raw_text_preview": raw_text[:800],
    }


def normalize_work_order_filter_result(
    parsed: dict[str, object],
    hit: dict[str, object],
    linked_parts: list[dict[str, object]],
    *,
    debug: dict[str, object],
) -> dict[str, object]:
    """Normalize one work-order LLM JSON response to the harness schema."""

    level = normalize_relevance_level(parsed.get("relevance_level"))
    related = parsed.get("related")
    if not isinstance(related, bool):
        related = level in {"high", "medium"}
    if level not in {"high", "medium"}:
        related = False

    usable_parts = normalize_llm_parts(parsed.get("usable_parts"))
    if related and not usable_parts:
        usable_parts = [part_for_answer(part) for part in linked_parts]
    return {
        "work_order_id": clean_string(parsed.get("work_order_id") or work_order_id_from_hit(hit)),
        "related": related,
        "relevance_level": level,
        "matched_reason": clean_string(parsed.get("matched_reason")) or "模型未给出原因。",
        "repair_actions": string_list(parsed.get("repair_actions")),
        "usable_parts": usable_parts,
        "source_path": clean_string(parsed.get("source_path") or hit.get("source_path")),
        "title": clean_string(hit.get("title")),
        "score": hit.get("score"),
        "body_preview": clean_string(hit.get("body_preview")),
        "hit": hit,
        "debug": debug,
    }


def normalize_relevance_level(value: object) -> str:
    """Normalize LLM relevance level labels."""

    level = clean_string(value).lower()
    if level in {"high", "medium", "low", "unrelated", "unknown"}:
        return level
    if level in {"相关", "高度相关"}:
        return "high"
    if level in {"中等相关", "可能相关"}:
        return "medium"
    if level in {"不相关", "无关"}:
        return "unrelated"
    return "unknown"


def normalize_llm_parts(value: object) -> list[dict[str, object]]:
    """Normalize part-like JSON items returned by the LLM."""

    parts: list[dict[str, object]] = []
    for item in list_payload(value):
        part = {
            "name": clean_string(item.get("name") or item.get("part_name") or item.get("part_number_name")),
            "code": clean_string(item.get("code") or item.get("part_code") or item.get("part_number")),
            "quantity": clean_string(item.get("quantity")),
            "source_work_order_id": clean_string(item.get("source_work_order_id") or item.get("work_order_id")),
        }
        if any(part.values()):
            parts.append(part)
    return parts


def string_list(value: object) -> list[str]:
    """Normalize an arbitrary JSON value to a compact string list."""

    if isinstance(value, list):
        return [clean_string(item) for item in value if clean_string(item)]
    text = clean_string(value)
    return [text] if text else []


def manual_selection_payload(hit: dict[str, object], *, relevance_level: str, reason: str) -> dict[str, object]:
    """Build one normalized manual selection item."""

    return {
        "doc_id": clean_string(hit.get("doc_id")),
        "title": clean_string(hit.get("title")),
        "doc_type": clean_string(hit.get("doc_type")),
        "relevance_level": normalize_relevance_level(relevance_level) if relevance_level != "medium" else "medium",
        "reason": reason,
        "source_path": clean_string(hit.get("source_path")),
        "score": hit.get("score"),
        "body_preview": clean_string(hit.get("body_preview")),
        "hit": hit,
    }


def normalize_manual_filter_result(parsed: dict[str, object], manual_hits: list[dict[str, object]]) -> dict[str, object]:
    """Normalize manual-title selection JSON and map IDs back to original hits."""

    hits_by_doc_id = {clean_string(hit.get("doc_id")): hit for hit in manual_hits}
    selected_ids: set[str] = set()
    rejected_ids: set[str] = set()
    selected: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []

    for item in list_payload(parsed.get("selected")):
        doc_id = clean_string(item.get("doc_id"))
        hit = hits_by_doc_id.get(doc_id)
        if not hit:
            continue
        level = normalize_relevance_level(item.get("relevance_level") or "medium")
        if level not in {"high", "medium"}:
            rejected.append(manual_selection_payload(hit, relevance_level="unrelated", reason=clean_string(item.get("reason")) or "模型未判定为相关。"))
            rejected_ids.add(doc_id)
            continue
        selected.append(manual_selection_payload(hit, relevance_level=level, reason=clean_string(item.get("reason")) or "模型判定相关。"))
        selected_ids.add(doc_id)

    for item in list_payload(parsed.get("rejected")):
        doc_id = clean_string(item.get("doc_id"))
        hit = hits_by_doc_id.get(doc_id)
        if not hit or doc_id in selected_ids or doc_id in rejected_ids:
            continue
        rejected.append(manual_selection_payload(hit, relevance_level="unrelated", reason=clean_string(item.get("reason")) or "模型判定不相关。"))
        rejected_ids.add(doc_id)

    for hit in manual_hits:
        doc_id = clean_string(hit.get("doc_id"))
        if doc_id in selected_ids or doc_id in rejected_ids:
            continue
        rejected.append(manual_selection_payload(hit, relevance_level="unrelated", reason="模型未选择该手册标题。"))

    return {"selected": selected, "rejected": rejected}


def build_deterministic_facts(
    *,
    selected_evidence: dict[str, object],
    selected_parts: list[dict[str, object]],
) -> dict[str, object]:
    """Build harness facts deterministically from selected evidence."""

    fault_code_hits = list_payload(selected_evidence.get("fault_code_evidence"))
    selected_work_orders = list_payload(selected_evidence.get("work_orders"))
    selected_manuals = list_payload(selected_evidence.get("manuals"))
    return {
        "fault_code_facts": [
            {
                "doc_id": hit.get("doc_id"),
                "title": hit.get("title"),
                "summary": clean_string(hit.get("body_preview"))[:260],
                "source_path": hit.get("source_path"),
                "mentioned_parts": [],
            }
            for hit in fault_code_hits
        ],
        "work_order_groups": [
            {
                "summary": clean_string(order.get("matched_reason"))
                or "; ".join(string_list(order.get("repair_actions")))
                or clean_string(order.get("body_preview"))[:220],
                "repair_actions": string_list(order.get("repair_actions")),
                "source_work_orders": [order.get("work_order_id")],
                "parts": order.get("usable_parts") if isinstance(order.get("usable_parts"), list) else [],
            }
            for order in selected_work_orders
        ],
        "manual_summaries": [
            {
                "doc_id": manual.get("doc_id"),
                "title": manual.get("title"),
                "summary": clean_string(manual.get("body_preview"))[:260],
                "source_path": manual.get("source_path"),
            }
            for manual in selected_manuals
        ],
        "coded_parts": coded_parts_from_parts(selected_parts),
        "uncoded_possible_parts": [],
    }


def normalize_answer_facts(parsed: dict[str, object], *, fallback: dict[str, object]) -> dict[str, object]:
    """Normalize LLM-extracted facts while preserving deterministic fallbacks."""

    facts = {
        "fault_code_facts": list_payload(parsed.get("fault_code_facts")) or list_payload(fallback.get("fault_code_facts")),
        "work_order_groups": list_payload(parsed.get("work_order_groups")) or list_payload(fallback.get("work_order_groups")),
        "manual_summaries": list_payload(parsed.get("manual_summaries")) or list_payload(fallback.get("manual_summaries")),
        "coded_parts": merge_coded_parts(
            list_payload(fallback.get("coded_parts")),
            list_payload(parsed.get("coded_parts")),
        ),
        "uncoded_possible_parts": list_payload(parsed.get("uncoded_possible_parts")),
    }
    return facts


def merge_coded_parts(primary: list[dict[str, object]], secondary: list[dict[str, object]]) -> list[dict[str, object]]:
    """Merge coded-part facts without dropping structured work-order parts."""

    merged: list[dict[str, object]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for part in [*primary, *secondary]:
        normalized = {
            "name": clean_string(part.get("name") or part.get("part_name")),
            "code": clean_string(part.get("code") or part.get("part_code")),
            "quantity": clean_string(part.get("quantity")),
            "source_work_order_id": clean_string(part.get("source_work_order_id") or part.get("work_order_id")),
        }
        if not normalized.get("code"):
            continue
        row_key = (
            normalized["name"],
            normalized["code"],
            normalized["quantity"],
            normalized["source_work_order_id"],
        )
        if row_key in seen:
            continue
        seen.add(row_key)
        merged.append(normalized)
    return merged


def build_final_answer_context(
    *,
    query: str,
    fault_code_hits: list[dict[str, object]],
    accepted_orders: list[dict[str, object]],
    selected_manuals: list[dict[str, object]],
    facts: dict[str, object],
    selected_parts: list[dict[str, object]],
) -> dict[str, object]:
    """Build the final context passed to answer generation."""

    selected_evidence: list[dict[str, object]] = []
    for hit in fault_code_hits:
        evidence = dict(hit)
        evidence["channel"] = "manual_fault_codes"
        selected_evidence.append(evidence)
    for order in accepted_orders:
        hit = dict_payload(order.get("hit"))
        evidence = dict(hit or order)
        evidence["channel"] = "work_orders"
        evidence["filter"] = {
            "relevance_level": order.get("relevance_level"),
            "matched_reason": order.get("matched_reason"),
            "repair_actions": order.get("repair_actions"),
        }
        selected_evidence.append(evidence)
    for manual in selected_manuals:
        hit = dict_payload(manual.get("hit"))
        evidence = dict(hit or manual)
        evidence["channel"] = "manual_typical_faults"
        evidence["filter"] = {
            "relevance_level": manual.get("relevance_level"),
            "reason": manual.get("reason"),
        }
        selected_evidence.append(evidence)

    return {
        "query": query,
        "fault_code_evidence": fault_code_hits,
        "selected_work_orders": accepted_orders,
        "selected_manuals": selected_manuals,
        "selected_parts": selected_parts,
        "facts": facts,
        "selected_evidence": selected_evidence,
    }


def flatten_evidence(payload: dict[str, object], *, limit: int) -> list[dict[str, object]]:
    """Flatten channel hits into a single evidence list for rerank and answers."""

    channels = payload.get("channels")
    if not isinstance(channels, dict):
        return []
    evidence: list[dict[str, object]] = []
    for channel_name in ("work_orders", "manual_typical_faults", "manual_fault_codes", "part_evidence"):
        hits = channels.get(channel_name)
        if not isinstance(hits, list):
            continue
        for rank, hit in enumerate(hits, start=1):
            if not isinstance(hit, dict):
                continue
            evidence.append(
                {
                    "channel": channel_name,
                    "rank": rank,
                    "doc_id": hit.get("doc_id"),
                    "doc_type": hit.get("doc_type"),
                    "title": hit.get("title"),
                    "score": hit.get("score"),
                    "source_path": hit.get("source_path"),
                    "work_order_id": hit.get("work_order_id"),
                    "body_preview": hit.get("body_preview"),
                    "metadata": hit.get("metadata") or {},
                    "matched_terms": (hit.get("matched_terms") or [])[:20],
                }
            )
    evidence.sort(key=lambda item: (channel_priority(str(item.get("channel"))), int(item.get("rank") or 0)))
    return evidence[: max(limit, 1)]


def channel_priority(channel: str) -> int:
    """Return channel priority for answer packaging."""

    priorities = {
        "work_orders": 0,
        "part_evidence": 1,
        "manual_fault_codes": 2,
        "manual_typical_faults": 3,
    }
    return priorities.get(channel, 9)


def build_rerank_document(item: dict[str, object], *, char_limit: int) -> str:
    """Build a compact rerank document string from one evidence item."""

    text = "\n".join(
        clean_string(value)
        for value in (
            f"channel: {item.get('channel')}",
            f"title: {item.get('title')}",
            f"doc_type: {item.get('doc_type')}",
            f"work_order_id: {item.get('work_order_id')}",
            f"source_path: {item.get('source_path')}",
            item.get("body_preview"),
        )
        if clean_string(value)
    )
    return text[:char_limit]


def stage_event(stage: str, status: str, details: dict[str, object] | None = None) -> dict[str, object]:
    """Create one structured pipeline trace event."""

    return {
        "stage": stage,
        "status": status,
        "details": redact_secrets(details or {}),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def elapsed_ms(started_at: float) -> int:
    """Return elapsed milliseconds from a start time."""

    return int((time.time() - started_at) * 1000)


def iter_manual_files(root: Path) -> Iterable[Path]:
    """Yield manual HTML and Markdown files under a root directory."""

    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in MANUAL_SUFFIXES:
            yield path


def limited_paths(paths: list[Path], limit: int | None) -> list[Path]:
    """Return paths clipped to a non-negative optional limit."""

    if limit is None:
        return paths
    return paths[: max(limit, 0)]


def unique_terms(terms: Iterable[str], *, limit: int = 200) -> list[str]:
    """Deduplicate query terms while preserving order."""

    unique: list[str] = []
    seen: set[str] = set()
    for term in terms:
        cleaned = term.strip().lower()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        unique.append(cleaned)
        if len(unique) >= limit:
            break
    return unique


def build_embedding_provider(config: AppConfig) -> CommandEmbeddingProvider | OpenAICompatibleEmbeddingProvider | None:
    """Build the configured embedding provider when available."""

    if not config.embedding.is_available():
        return None
    if config.embedding.provider.lower().strip() == "command":
        return CommandEmbeddingProvider(config.embedding)
    return OpenAICompatibleEmbeddingProvider(config.embedding)


def json_param(payload: object) -> object:
    """Wrap JSON values for psycopg adaptation."""

    try:
        from psycopg.types.json import Json
    except ImportError as exc:  # pragma: no cover - depends on environment.
        raise RuntimeError("psycopg is required. Install with: pip install -e .") from exc
    return Json(payload)


def vector_literal(vector: list[float]) -> str:
    """Return a pgvector-compatible vector literal."""

    if not vector:
        raise ValueError("vector must not be empty")
    if any(not math.isfinite(float(value)) for value in vector):
        raise ValueError("vector contains non-finite values")
    return "[" + ",".join(f"{float(value):.12g}" for value in vector) + "]"


def preview_text(text: str, *, limit: int = 420) -> str:
    """Return a compact text preview for CLI and web output."""

    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def redact_database_url(database_url: str) -> str:
    """Redact passwords from a database URL for reports."""

    return re.sub(r"(postgres(?:ql)?://[^:/@\s]+:)([^@/\s]+)(@)", r"\1***\3", database_url)


def format_ingest_report_summary(report: PgIngestReport) -> str:
    """Return a compact ingest summary."""

    return (
        f"work_orders={report.work_orders}, part_records={report.part_records}, "
        f"manual_files={report.manual_files}, manual_chunks={report.manual_chunks}, "
        f"documents={report.total_documents}, term_rows={report.term_rows}, "
        f"embeddings={report.embeddings}, skipped_files={report.skipped_files}, failed_items={len(report.failed_items)}, "
        f"elapsed={report.elapsed_seconds}s"
    )


def format_embedding_report_summary(report: PgEmbeddingReport) -> str:
    """Return a compact embedding backfill summary."""

    return (
        f"candidates={report.total_candidates}, processed={report.processed_documents}, "
        f"embeddings={report.embeddings}, failed_items={len(report.failed_items)}, "
        f"elapsed={report.elapsed_seconds}s"
    )


def format_search_summary(payload: dict[str, object]) -> str:
    """Return a compact retrieval summary."""

    channels = payload.get("channels") if isinstance(payload, dict) else {}
    if not isinstance(channels, dict):
        return "channels=0"
    counts = ", ".join(f"{name}={len(items) if isinstance(items, list) else 0}" for name, items in channels.items())
    return f"mode={payload.get('mode')}, {counts}, part_candidates={len(payload.get('part_candidates') or [])}"


def write_json(payload: object, path: Path) -> None:
    """Write JSON output to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
