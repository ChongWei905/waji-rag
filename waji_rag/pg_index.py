"""PostgreSQL and pgvector backed indexing and retrieval."""

from __future__ import annotations

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
from waji_rag.html_batch import DEFAULT_ENCODINGS, read_text_with_fallback
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
from waji_rag.llm import DashScopeRerankClient, build_fallback_answer, generate_diagnostic_answer
from waji_rag.work_order import WorkOrderParser, WorkOrderRecord, iter_txt_files


DEFAULT_DATABASE_URL = "postgresql://waji:waji@127.0.0.1:55432/waji_rag"
SCHEMA_VERSION = 1
DEFAULT_BM25_K1 = 1.5
DEFAULT_BM25_B = 0.75
MANUAL_SUFFIXES = {".html", ".htm", ".md", ".markdown"}
FAULT_CODE_IN_QUERY = re.compile(r"\b[A-Za-z]\d{3,}[A-Za-z0-9_-]*\b")


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
    config_path: Path | None = None
    config_overrides: dict[str, Any] | None = None
    env_path: Path | None = None
    reset: bool = False
    work_order_limit: int | None = None
    manual_limit: int | None = None
    max_manual_chars: int = 1800
    encodings: tuple[str, ...] = DEFAULT_ENCODINGS
    progress_callback: Callable[[dict[str, object]], None] | None = None


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
    failed_items: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable report dictionary."""

        return asdict(self)


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

    def ingest(self) -> PgIngestReport:
        """Run schema initialization, parse files, and write searchable records."""

        start_time = time.time()
        self._emit_progress({"phase": "init", "message": "初始化 PostgreSQL schema", "percent": 0})
        PgSchemaManager(self.options.database).initialize(reset=self.options.reset)
        work_order_dir = self.options.work_order_dir.resolve() if self.options.work_order_dir else None
        manual_dir = self.options.manual_dir.resolve() if self.options.manual_dir else None
        validate_ingest_inputs(work_order_dir, manual_dir)
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
            report.work_order_files = len(limited_paths(list(iter_txt_files(work_order_dir)), self.options.work_order_limit))
        if manual_dir is not None:
            report.manual_files = len(limited_paths(list(iter_manual_files(manual_dir)), self.options.manual_limit))
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
                if work_order_dir is not None:
                    self._ingest_work_orders(cur, work_order_dir, report)
                if manual_dir is not None:
                    self._ingest_manuals(cur, manual_dir, report)
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

    def _ingest_work_orders(self, cur: Any, root: Path, report: PgIngestReport) -> None:
        txt_files = limited_paths(list(iter_txt_files(root)), self.options.work_order_limit)
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
            try:
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
                record = self.parser.parse(text, source_path=txt_path, encoding=encoding)
                delete_source_records(cur, str(txt_path))
                upsert_work_order(cur, record)
                report.work_orders += 1
                if record.parse_warnings:
                    report.warnings.append(f"{txt_path}: {', '.join(record.parse_warnings)}")

                documents = build_documents_for_work_order(record)
                for document in documents:
                    document_id, term_rows = upsert_search_document(cur, document)
                    report.total_documents += 1
                    report.term_rows += term_rows
                    if maybe_store_embedding(cur, document_id, document, self.config, self.embedding_provider, report):
                        report.embeddings += 1
                report.part_records += len(record.parts)
            except Exception as exc:  # noqa: BLE001 - keep per-file diagnostics.
                report.failed_items.append(
                    {"stage": "work_order", "input": str(txt_path), "error": f"{type(exc).__name__}: {exc}"}
                )
            finally:
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

    def _ingest_manuals(self, cur: Any, root: Path, report: PgIngestReport) -> None:
        manual_files = limited_paths(list(iter_manual_files(root)), self.options.manual_limit)
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
            try:
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
                metadata = infer_manual_metadata(manual_path, root)
                chunks = chunk_markdown(markdown_text, max_chars=max(self.options.max_manual_chars, 200))
                if not chunks:
                    report.warnings.append(f"empty_manual_file: {manual_path}")
                    continue

                delete_source_records(cur, str(manual_path))
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
                        root=root,
                        metadata=chunk_metadata,
                        chunk_text=chunk_text,
                        chunk_index=chunk_index,
                    )
                    insert_manual_chunk(cur, document, chunk_metadata)
                    document_id, term_rows = upsert_search_document(cur, document)
                    report.manual_chunks += 1
                    report.total_documents += 1
                    report.term_rows += term_rows
                    if converted_from_html:
                        report.html_converted_in_memory += 1 if chunk_index == 0 else 0
                    if maybe_store_embedding(cur, document_id, document, self.config, self.embedding_provider, report):
                        report.embeddings += 1
            except Exception as exc:  # noqa: BLE001 - keep per-file diagnostics.
                report.failed_items.append(
                    {"stage": "manual", "input": str(manual_path), "error": f"{type(exc).__name__}: {exc}"}
                )
            finally:
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
            "failed_count": len(report.failed_items),
            "recent_failures": report.failed_items[-5:],
            "warnings": report.warnings[-5:],
            "elapsed_seconds": elapsed_seconds,
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

        terms = unique_terms(tokenize_text(query))
        effective_mode = self.config.retrieval_mode()
        retrieval_events: list[dict[str, object]] = []
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
                    debug_events=retrieval_events,
                )
                fault_code_hits = self._search_fault_code_channel(
                    cur,
                    query,
                    terms,
                    top_k=top_k,
                    mode=effective_mode,
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
                    debug_events=retrieval_events,
                )
                channels = {
                    "work_orders": work_order_hits,
                    "manual_typical_faults": typical_hits,
                    "manual_fault_codes": fault_code_hits,
                    "part_evidence": part_hits,
                }
                part_candidates = fetch_part_candidates(cur, collect_work_order_ids(work_order_hits, part_hits), top_k=top_k * 3)

        payload: dict[str, object] = {
            "query": query,
            "mode": effective_mode,
            "top_k": top_k,
            "channels": {name: [hit.to_dict() for hit in hits] for name, hits in channels.items()},
            "part_candidates": part_candidates,
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
        bm25_hits = search_bm25(cur, terms, top_k=top_k, doc_types=doc_types)
        if mode != "hybrid" or self.embedding_provider is None:
            return bm25_hits

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
            return bm25_hits
        try:
            vector_hits = search_vectors(
                cur,
                query_vector,
                provider=self.config.embedding.provider,
                model=self.config.embedding.model,
                top_k=max(top_k, self.config.retrieval.vector_top_k),
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
            return bm25_hits
        debug_events.append(
            {
                "channel": channel_name,
                "stage": "vector_search",
                "status": "ok",
                "bm25_hits": len(bm25_hits),
                "vector_hits": len(vector_hits),
            }
        )
        return merge_hybrid_hits(
            bm25_hits,
            vector_hits,
            top_k=top_k,
            bm25_weight=self.config.retrieval.hybrid_alpha,
        )


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
        rerank_payload = self._rerank(query, evidence_items, trace)
        selected_evidence = list(rerank_payload.get("evidence") or evidence_items)
        answer_payload = self._answer(query, selected_evidence, list(retrieval.get("part_candidates") or []), trace)
        payload: dict[str, object] = {
            "query": query,
            "answer": answer_payload,
            "retrieval": retrieval,
            "rerank": {
                "enabled": self.config.rerank.enabled,
                "available": self.config.rerank.is_available(),
                **{key: value for key, value in rerank_payload.items() if key != "evidence"},
            },
            "selected_evidence": selected_evidence,
            "part_candidates": retrieval.get("part_candidates") or [],
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


def validate_ingest_inputs(work_order_dir: Path | None, manual_dir: Path | None) -> None:
    """Validate ingest input directories."""

    if work_order_dir is None and manual_dir is None:
        raise ValueError("at least one of work_order_dir or manual_dir is required")
    for label, path in (("work_order_dir", work_order_dir), ("manual_dir", manual_dir)):
        if path is None:
            continue
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"{label} is not a directory: {path}")


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
        "failed_items": len(report.failed_items),
    }


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
                metadata={"work_order_id": work_order_id, "quantity": part.quantity, "part_index": index},
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


def upsert_search_document(cur: Any, document: IndexDocument) -> tuple[int, int]:
    """Insert one searchable document and its BM25 term statistics."""

    work_order_id = clean_string(document.metadata.get("work_order_id")) or None
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

    term_rows = 0
    for field_name, field_text in document.fields.items():
        terms = tokenize_text(field_text)
        field_length = len(terms)
        field_weight = float(FIELD_WEIGHTS.get(field_name, 1.0))
        cur.execute(
            """
            INSERT INTO document_fields(document_id, field_name, field_text, field_weight, field_length)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (document_id, field_name, clean_string(field_text), field_weight, field_length),
        )
        for term, tf in Counter(terms).items():
            cur.execute(
                """
                INSERT INTO document_terms(term, document_id, field_name, tf, field_weight, field_length)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (term, document_id, field_name, tf, field_weight, field_length),
            )
            term_rows += 1
    return document_id, term_rows


def maybe_store_embedding(
    cur: Any,
    document_id: int,
    document: IndexDocument,
    config: AppConfig,
    provider: CommandEmbeddingProvider | OpenAICompatibleEmbeddingProvider | None,
    report: PgIngestReport,
) -> bool:
    """Store an optional embedding for one document when configured."""

    if provider is None:
        return False
    try:
        vector = provider.embed_texts([f"{document.title}\n{document.text}"])[0]
    except EmbeddingProviderError as exc:
        report.warnings.append(f"embedding_failed:{document.doc_id}: {exc}")
        return False
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
    return True


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


def fetch_part_candidates(cur: Any, work_order_ids: list[str], *, top_k: int) -> list[dict[str, object]]:
    """Fetch part rows linked to retrieved work orders."""

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
        ORDER BY id ASC
        LIMIT %s
        """,
        (work_order_ids, top_k),
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
    return [dict(zip(keys, row, strict=False)) for row in cur.fetchall()]


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
