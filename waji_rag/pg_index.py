"""PostgreSQL and pgvector backed indexing and retrieval."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections import Counter
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
    build_fallback_answer,
    generate_diagnostic_answer,
    parse_diagnostic_query_constraints,
)
from waji_rag.work_order import WorkOrderParser, WorkOrderRecord, iter_txt_files


DEFAULT_DATABASE_URL = "postgresql://waji:waji@127.0.0.1:55432/waji_rag"
SCHEMA_VERSION = 1
DEFAULT_BM25_K1 = 1.5
DEFAULT_BM25_B = 0.75
DEFAULT_BM25_BATCH_TERM_ROWS = 100_000
DEFAULT_SOURCE_CHECKPOINT_FILES = 50
MANUAL_SUFFIXES = {".html", ".htm", ".md", ".markdown"}
FAULT_CODE_IN_QUERY = re.compile(r"\b[A-Za-z]\d{3,}[A-Za-z0-9_-]*\b")
FAULT_PHRASE_SPLIT_PATTERN = re.compile(r"请|回答|如何|怎么|有哪些|哪些|相应|需要|（|\(")
QUERY_LABEL_PATTERN = re.compile(r"^(问题|用户问题|用户报修内容|报修内容)\s*[:：]?\s*")
REPORT_PREFIX_PATTERN = re.compile(r"^(用户|客户|司机)?\s*(报修|保修|反馈|反映)\s*(机器|设备|挖机|挖掘机|该机)?\s*")
LEADING_MACHINE_PATTERN = re.compile(r"^(机器|设备|挖机|挖掘机|该机|车辆|车)\s*")
SYMPTOM_TERMS = {
    "异响",
    "噪声",
    "噪音",
    "响声",
    "尖叫",
    "尖叫声",
    "异常声音",
    "摩擦声",
    "慢",
    "单边慢",
    "无力",
    "高温",
    "漏油",
    "渗油",
    "报警",
    "报错",
    "开裂",
    "损坏",
    "不制冷",
    "不启动",
    "不起动",
    "不工作",
    "不动作",
    "无法动作",
    "无法行走",
}
GENERIC_QUERY_TERMS = {
    "用户",
    "客户",
    "司机",
    "报修",
    "保修",
    "反馈",
    "机器",
    "设备",
    "挖机",
    "挖掘机",
    "该机",
    "故障",
    "原因",
    "可能",
    "导致",
    "如何",
    "解决",
    "需要",
    "更换",
    "备件",
    "详细",
    "信息",
    "编号",
    "名称",
    "编码",
    "数量",
}
COMMON_COMPONENT_TERMS = {
    "风扇",
    "皮带",
    "发动机",
    "张紧轮",
    "皮带轮",
    "空调",
    "压缩机",
    "鼓风机",
    "行走",
    "马达",
    "主泵",
    "液压",
    "动臂",
    "斗杆",
    "铲斗",
    "油缸",
    "回转",
    "电瓶",
    "发电机",
    "gps",
}
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
class QueryConstraints:
    """Deterministic evidence constraints parsed from a diagnostic query."""

    fault_phrase: str
    component_text: str
    component_terms: list[str]
    required_component_terms: list[str]
    symptom_terms: list[str]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable constraints payload."""

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
        retrieval_events: list[dict[str, object]] = []
        terms = unique_terms(tokenize_text(query))
        constraints = resolve_query_constraints(query, terms, self.config, retrieval_events)
        channels: dict[str, list[RetrievalHit]] = {}
        with connect(self.database.database_url) as conn:
            with conn.cursor() as cur:
                work_order_hits = self._search_channel(
                    cur,
                    query,
                    terms,
                    top_k=top_k,
                    doc_types=["work_order"],
                    mode=effective_mode,
                    channel_name="work_orders",
                    constraints=constraints,
                    debug_events=retrieval_events,
                )
                typical_hits = self._search_channel(
                    cur,
                    query,
                    terms,
                    top_k=top_k,
                    doc_types=["manual_typical_fault"],
                    mode=effective_mode,
                    channel_name="manual_typical_faults",
                    constraints=constraints,
                    debug_events=retrieval_events,
                )
                fault_code_hits = self._search_fault_code_channel(
                    cur,
                    query,
                    terms,
                    top_k=top_k,
                    mode=effective_mode,
                    constraints=constraints,
                    debug_events=retrieval_events,
                )
                part_hits = self._search_channel(
                    cur,
                    query,
                    terms,
                    top_k=top_k,
                    doc_types=["part_evidence"],
                    mode=effective_mode,
                    channel_name="part_evidence",
                    constraints=constraints,
                    debug_events=retrieval_events,
                )
                enrich_part_hit_fields(cur, part_hits)
                channels = {
                    "work_orders": work_order_hits,
                    "manual_typical_faults": typical_hits,
                    "manual_fault_codes": fault_code_hits,
                    "part_evidence": part_hits,
                }
                linked_work_order_ids = collect_work_order_ids(work_order_hits, part_hits)
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
            "top_k": top_k,
            "channels": channel_payloads,
            "part_candidates": part_candidates,
            "part_candidate_source": {
                "linked_work_order_ids": linked_work_order_ids,
                "limit_applied": False,
                "search_hit_count": len(channels.get("part_evidence", [])),
            },
        }
        evidence_filter = filter_evidence_for_answer(
            query=query,
            evidence_items=flatten_evidence(payload, limit=max(self.config.answer.evidence_top_k, top_k, 1) * 4),
            constraints=constraints,
        )
        payload["evidence_filter"] = evidence_filter
        payload["filtered_part_candidates"] = filter_part_candidates_by_evidence(part_candidates, evidence_filter)
        fallback_events = [event for event in retrieval_events if event.get("status") == "fallback"]
        if fallback_events:
            payload["warnings"] = [
                f"{event.get('channel')}:{event.get('stage')}:{event.get('reason')}" for event in fallback_events
            ]
        if include_debug:
            payload["debug"] = {
                "query_terms": terms,
                "query_constraints": constraints.to_dict(),
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
        constraints: QueryConstraints,
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
                constraints=constraints,
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
        constraints: QueryConstraints,
        debug_events: list[dict[str, object]],
    ) -> list[RetrievalHit]:
        candidate_top_k = max(top_k, self.config.retrieval.bm25_top_k)
        bm25_hits = search_bm25(cur, terms, top_k=candidate_top_k, doc_types=doc_types)
        if mode != "hybrid" or self.embedding_provider is None:
            return prioritize_hits_by_constraints(bm25_hits, constraints, top_k=top_k)

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
            return prioritize_hits_by_constraints(bm25_hits, constraints, top_k=top_k)
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
            return prioritize_hits_by_constraints(bm25_hits, constraints, top_k=top_k)
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
        return prioritize_hits_by_constraints(merged_hits, constraints, top_k=top_k)


class RagPipeline:
    """Run retrieve, optional rerank, and optional answer generation."""

    def __init__(self, database: DatabaseOptions, config: AppConfig) -> None:
        """Store pipeline dependencies."""

        self.database = database
        self.config = config

    def run(self, query: str, *, top_k: int = 8, include_debug: bool = True) -> dict[str, object]:
        """Run the full RAG pipeline and return answer, evidence, and debug logs."""

        trace: list[dict[str, object]] = []
        started_at = time.time()
        trace.append(stage_event("config", "ok", {"config": self.config.to_dict()}))
        retriever = PgRetriever(self.database, self.config)
        retrieve_started_at = time.time()
        retrieval = retriever.retrieve(query, top_k=top_k, include_debug=include_debug)
        evidence_filter = object_payload_or_empty(retrieval.get("evidence_filter"))
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

        evidence_items = list_payload(evidence_filter.get("accepted"))
        if not evidence_items:
            evidence_items = flatten_evidence(retrieval, limit=max(self.config.answer.evidence_top_k, top_k))
        trace.append(
            stage_event(
                "evidence_filter",
                str(evidence_filter.get("status") or "ok"),
                {
                    "summary": evidence_filter.get("summary"),
                    "accepted_count": len(list_payload(evidence_filter.get("accepted"))),
                    "rejected_count": len(list_payload(evidence_filter.get("rejected"))),
                    "constraints": evidence_filter.get("constraints") or {},
                },
            )
        )
        part_candidates = list_payload(retrieval.get("filtered_part_candidates")) or list_payload(retrieval.get("part_candidates"))
        rerank_payload = self._rerank(query, evidence_items, trace)
        selected_evidence = list(rerank_payload.get("evidence") or evidence_items)
        answer_payload = self._answer(query, selected_evidence, part_candidates, trace)
        payload: dict[str, object] = {
            "query": query,
            "answer": answer_payload,
            "retrieval": retrieval,
            "evidence_filter": evidence_filter,
            "rerank": {
                "enabled": self.config.rerank.enabled,
                "available": self.config.rerank.is_available(),
                **{key: value for key, value in rerank_payload.items() if key != "evidence"},
            },
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
        return payload

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
        evidence_items: list[dict[str, object]],
        part_candidates: list[dict[str, object]],
        trace: list[dict[str, object]],
    ) -> dict[str, object]:
        if not self.config.answer.enabled:
            trace.append(stage_event("answer", "skipped", {"reason": "disabled"}))
            return {"status": "skipped", "text": ""}
        if not self.config.llm.enabled:
            text = build_fallback_answer(query=query, evidence_items=evidence_items, part_candidates=part_candidates)
            trace.append(stage_event("answer", "fallback", {"reason": "llm_disabled"}))
            return {"status": "fallback", "reason": "llm_disabled", "text": text}
        if not self.config.llm.is_available():
            text = build_fallback_answer(query=query, evidence_items=evidence_items, part_candidates=part_candidates)
            trace.append(stage_event("answer", "fallback", {"reason": "missing_llm_config"}))
            return {"status": "fallback", "reason": "missing_llm_config", "text": text}

        started_at = time.time()
        try:
            result = generate_diagnostic_answer(
                query=query,
                evidence_items=evidence_items[: self.config.answer.evidence_top_k],
                part_candidates=part_candidates,
                config=self.config.llm,
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
            return {"status": "ok", "text": result.text, "debug": result.debug}
        except Exception as exc:  # noqa: BLE001 - generation must degrade.
            text = build_fallback_answer(query=query, evidence_items=evidence_items, part_candidates=part_candidates)
            trace.append(stage_event("answer", "fallback", {"elapsed_ms": elapsed_ms(started_at), "error": str(exc)}))
            return {"status": "fallback", "error": str(exc), "text": text}


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


def build_query_constraints(query: str, terms: list[str] | None = None) -> QueryConstraints:
    """Extract high-precision component and symptom anchors from a diagnostic query."""

    fault_phrase = extract_fault_phrase(query)
    symptom_terms = extract_symptom_terms(fault_phrase or query)
    component_text = strip_symptom_terms(fault_phrase, symptom_terms)
    required_component_terms = extract_required_component_terms(component_text)
    component_terms = extract_component_terms(
        component_text,
        terms or unique_terms(tokenize_text(query)),
        required_component_terms=required_component_terms,
    )
    return QueryConstraints(
        fault_phrase=fault_phrase,
        component_text=component_text,
        component_terms=component_terms,
        required_component_terms=required_component_terms,
        symptom_terms=symptom_terms,
    )


def resolve_query_constraints(
    query: str,
    terms: list[str],
    config: AppConfig,
    retrieval_events: list[dict[str, object]],
) -> QueryConstraints:
    """Resolve query constraints through LLM parsing with deterministic fallback."""

    fallback = build_query_constraints(query, terms)
    parser_config = config.query_parser
    if not parser_config.enabled:
        retrieval_events.append(
            {
                "channel": "query_parser",
                "stage": "constraints",
                "status": "fallback",
                "mode": "rules",
                "reason": "query_parser_disabled",
                "constraints": fallback.to_dict(),
            }
        )
        return fallback
    if not parser_config.is_available():
        retrieval_events.append(
            {
                "channel": "query_parser",
                "stage": "constraints",
                "status": "fallback",
                "mode": "rules",
                "reason": "missing_query_parser_config",
                "constraints": fallback.to_dict(),
            }
        )
        return fallback
    try:
        started_at = time.time()
        result = parse_diagnostic_query_constraints(query=query, config=parser_config)
        constraints = query_constraints_from_llm_payload(query, terms, result.payload, fallback=fallback)
        retrieval_events.append(
            {
                "channel": "query_parser",
                "stage": "constraints",
                "status": "ok",
                "mode": "llm",
                "model": parser_config.model,
                "elapsed_ms": elapsed_ms(started_at),
                "constraints": constraints.to_dict(),
                "usage": result.debug.get("usage"),
            }
        )
        return constraints
    except Exception as exc:  # noqa: BLE001 - query parsing must not block retrieval.
        retrieval_events.append(
            {
                "channel": "query_parser",
                "stage": "constraints",
                "status": "fallback",
                "mode": "rules",
                "reason": f"{type(exc).__name__}: {exc}",
                "constraints": fallback.to_dict(),
            }
        )
        return fallback


def query_constraints_from_llm_payload(
    query: str,
    terms: list[str],
    payload: dict[str, object],
    *,
    fallback: QueryConstraints | None = None,
) -> QueryConstraints:
    """Build safe query constraints from an LLM JSON payload."""

    fallback_constraints = fallback or build_query_constraints(query, terms)
    fault_phrase = clean_string(payload.get("fault_phrase")) or fallback_constraints.fault_phrase
    component_text = clean_string(
        payload.get("component_text")
        or payload.get("component")
        or payload.get("component_anchor")
        or payload.get("部件锚点")
    )
    raw_component_terms = string_list_payload(
        payload.get("component_terms")
        or payload.get("component_anchors")
        or payload.get("components")
    )
    raw_required_terms = string_list_payload(payload.get("required_component_terms"))
    raw_symptom_terms = string_list_payload(
        payload.get("symptom_terms")
        or payload.get("abnormal_terms")
        or payload.get("异常词")
    )
    source_text = normalize_for_gate(join_text([query, fault_phrase, component_text]))
    component_terms = safe_llm_terms(raw_component_terms, source_text)
    symptom_terms = safe_llm_terms(raw_symptom_terms, source_text)
    if not component_text and component_terms:
        component_text = component_terms[0]
    if not symptom_terms:
        symptom_terms = fallback_constraints.symptom_terms
    required_component_terms = safe_llm_terms(raw_required_terms, source_text)
    if not required_component_terms:
        required_component_terms = extract_required_component_terms(component_text)
    if not component_terms:
        component_terms = extract_component_terms(
            component_text or fallback_constraints.component_text,
            terms,
            required_component_terms=required_component_terms,
        )
    component_terms = unique_terms(
        term
        for term in component_terms
        if len(term) >= 2 and term not in SYMPTOM_TERMS and term not in GENERIC_QUERY_TERMS
    )[:12]
    required_component_terms = unique_terms(
        term
        for term in required_component_terms
        if len(term) >= 2 and term not in SYMPTOM_TERMS and term not in GENERIC_QUERY_TERMS
    )[:8]
    return QueryConstraints(
        fault_phrase=fault_phrase,
        component_text=component_text or fallback_constraints.component_text,
        component_terms=component_terms or fallback_constraints.component_terms,
        required_component_terms=required_component_terms or fallback_constraints.required_component_terms,
        symptom_terms=unique_terms(symptom_terms)[:8],
    )


def string_list_payload(value: object) -> list[str]:
    """Return a clean string list from model JSON payload values."""

    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        return []
    return [clean_string(item) for item in value if clean_string(item)]


def safe_llm_terms(terms: list[str], source_text: str) -> list[str]:
    """Keep only LLM terms explicitly anchored in the original question text."""

    safe_terms: list[str] = []
    for term in terms:
        normalized = normalize_for_gate(term)
        if len(normalized) >= 2 and normalized in source_text:
            safe_terms.append(normalized)
    return unique_terms(safe_terms)


def extract_fault_phrase(query: str) -> str:
    """Return the most likely user-reported fault phrase from a full question."""

    normalized = clean_string(query)
    if not normalized:
        return ""
    fragment = FAULT_PHRASE_SPLIT_PATTERN.split(normalized, maxsplit=1)[0]
    fragment = QUERY_LABEL_PATTERN.sub("", fragment).strip()
    fragment = REPORT_PREFIX_PATTERN.sub("", fragment).strip()
    fragment = LEADING_MACHINE_PATTERN.sub("", fragment).strip()
    fragment = fragment.strip(" :：，,。；;")
    return fragment or normalized


def extract_symptom_terms(text: str) -> list[str]:
    """Return known symptom words that appear in the query phrase."""

    normalized = normalize_for_gate(text)
    terms = [term for term in sorted(SYMPTOM_TERMS, key=len, reverse=True) if term in normalized]
    return unique_terms(terms)


def strip_symptom_terms(text: str, symptom_terms: list[str]) -> str:
    """Remove known symptom terms from a fault phrase to leave likely components."""

    component_text = normalize_for_gate(text)
    for term in sorted(symptom_terms, key=len, reverse=True):
        component_text = component_text.replace(term, "")
    component_text = REPORT_PREFIX_PATTERN.sub("", component_text).strip()
    component_text = LEADING_MACHINE_PATTERN.sub("", component_text).strip()
    return component_text.strip(" :：，,。；;")


def extract_required_component_terms(component_text: str) -> list[str]:
    """Return component terms that must match together for composite components."""

    normalized_component = normalize_for_gate(component_text)
    if len(normalized_component) < 2:
        return []
    known_terms = sorted(
        (term for term in COMMON_COMPONENT_TERMS if term in normalized_component),
        key=lambda term: (normalized_component.find(term), -len(term)),
    )
    if known_terms:
        return unique_terms(known_terms)
    return unique_terms(tokenize_text(normalized_component))[:2]


def extract_component_terms(
    component_text: str,
    fallback_terms: list[str],
    *,
    required_component_terms: list[str],
) -> list[str]:
    """Return component anchor terms that should dominate evidence filtering."""

    terms: list[str] = []
    normalized_component = normalize_for_gate(component_text)
    if len(normalized_component) >= 2:
        terms.append(normalized_component)
        terms.extend(required_component_terms or tokenize_text(normalized_component))
    if not terms:
        terms.extend(fallback_terms)
    return unique_terms(
        term
        for term in terms
        if len(term) >= 2 and term not in SYMPTOM_TERMS and term not in GENERIC_QUERY_TERMS
    )[:12]


def normalize_for_gate(value: object) -> str:
    """Normalize text for deterministic evidence gate substring checks."""

    return re.sub(r"\s+", "", clean_string(value).lower())


def prioritize_hits_by_constraints(
    hits: list[RetrievalHit],
    constraints: QueryConstraints,
    *,
    top_k: int,
) -> list[RetrievalHit]:
    """Prefer hits that mention component anchors before applying the final limit."""

    if not constraints.component_terms:
        return hits[:top_k]

    def sort_key(hit: RetrievalHit) -> tuple[int, int, float, int]:
        match = match_query_constraints(hit_filter_text(hit), constraints)
        has_component = bool(match["strict_component_match"])
        has_symptom = bool(match["symptom_hits"])
        return (0 if has_component else 1, 0 if has_symptom else 1, -hit.score, hit.document_id)

    return sorted(hits, key=sort_key)[:top_k]


def filter_evidence_for_answer(
    *,
    query: str,
    evidence_items: list[dict[str, object]],
    constraints: QueryConstraints | None = None,
) -> dict[str, object]:
    """Split retrieved evidence into accepted and rejected items before rerank/answer."""

    active_constraints = constraints or build_query_constraints(query)
    decorated: list[dict[str, object]] = []
    for item in evidence_items:
        text = evidence_item_filter_text(item)
        match = match_query_constraints(text, active_constraints)
        decorated_item = dict(item)
        decorated_item["evidence_gate"] = match
        decorated.append(decorated_item)

    component_required = bool(active_constraints.component_terms)
    component_supported = component_required and any(item["evidence_gate"]["strict_component_match"] for item in decorated)
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    for item in decorated:
        gate = dict(item.get("evidence_gate") or {})
        if not component_required:
            gate["decision"] = "accepted"
            gate["reason"] = "no_component_anchor"
            item["evidence_gate"] = gate
            accepted.append(item)
        elif not component_supported:
            gate["decision"] = "accepted"
            gate["reason"] = "no_component_supported_in_retrieval"
            item["evidence_gate"] = gate
            accepted.append(item)
        elif gate.get("strict_component_match"):
            gate["decision"] = "accepted"
            gate["reason"] = "strict_component_anchor_matched"
            item["evidence_gate"] = gate
            accepted.append(item)
        else:
            gate["decision"] = "rejected"
            gate["reason"] = "missing_strict_component_anchor"
            item["evidence_gate"] = gate
            rejected.append(item)

    status = "ok" if not rejected else "filtered"
    return {
        "status": status,
        "query": query,
        "constraints": active_constraints.to_dict(),
        "component_supported": component_supported,
        "summary": f"accepted={len(accepted)}, rejected={len(rejected)}",
        "accepted": accepted,
        "rejected": rejected,
    }


def match_query_constraints(text: str, constraints: QueryConstraints) -> dict[str, object]:
    """Return component and symptom anchor matches for one evidence text."""

    normalized = normalize_for_gate(text)
    token_set = set(tokenize_text(text))
    component_hits = [
        term for term in constraints.component_terms if term and (term in normalized or term in token_set)
    ]
    required_hits = [
        term for term in constraints.required_component_terms if term and (term in normalized or term in token_set)
    ]
    symptom_hits = [
        term for term in constraints.symptom_terms if term and (term in normalized or term in token_set)
    ]
    full_component_match = bool(constraints.component_text and normalize_for_gate(constraints.component_text) in normalized)
    required_component_match = bool(
        constraints.required_component_terms
        and len(unique_terms(required_hits)) >= len(unique_terms(constraints.required_component_terms))
    )
    return {
        "component_hits": unique_terms(component_hits),
        "required_component_hits": unique_terms(required_hits),
        "strict_component_match": full_component_match or required_component_match,
        "symptom_hits": unique_terms(symptom_hits),
    }


def hit_filter_text(hit: RetrievalHit) -> str:
    """Build searchable text for filtering one retrieval hit."""

    return join_text(
        [
            hit.title,
            hit.body_preview,
            hit.source_path,
            metadata_filter_text(hit.metadata),
        ]
    )


def evidence_item_filter_text(item: dict[str, object]) -> str:
    """Build searchable text for filtering one flattened evidence item."""

    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    return join_text(
        [
            item.get("title"),
            item.get("body_preview"),
            item.get("source_path"),
            metadata_filter_text(metadata),
        ]
    )


def metadata_filter_text(metadata: dict[str, object]) -> str:
    """Return compact metadata text useful for evidence gating."""

    keys = ("fault_title", "fault_description", "file_name", "system_dir", "manual_section", "part_name", "part_code")
    return join_text(metadata.get(key) for key in keys)


def filter_part_candidates_by_evidence(
    part_candidates: list[dict[str, object]],
    evidence_filter: dict[str, object],
) -> list[dict[str, object]]:
    """Keep part candidates tied to accepted work orders when the gate has such IDs."""

    accepted = list_payload(evidence_filter.get("accepted"))
    accepted_ids = {
        str(item.get("work_order_id"))
        for item in accepted
        if isinstance(item, dict) and item.get("work_order_id")
    }
    if not accepted_ids:
        return part_candidates
    return [part for part in part_candidates if str(part.get("work_order_id")) in accepted_ids]


def object_payload_or_empty(value: object) -> dict[str, object]:
    """Return a dict payload or an empty dict."""

    return value if isinstance(value, dict) else {}


def list_payload(value: object) -> list[dict[str, object]]:
    """Return a list of dict payloads from JSON-like data."""

    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


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
