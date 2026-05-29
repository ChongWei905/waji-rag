"""Command-line entry points for local RAG debugging."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from waji_rag import __version__
from waji_rag.html_batch import (
    ConvertOptions,
    HtmlToMarkdownBatch,
    failure_trace,
    format_report_summary,
    write_report,
)
from waji_rag.index_build import (
    IndexBuildOptions,
    LocalIndexBuilder,
    format_report_summary as format_index_report_summary,
    write_report as write_index_report,
)
from waji_rag.config import write_default_config
from waji_rag.pg_index import (
    DatabaseOptions,
    PgIngestBuilder,
    PgIngestOptions,
    PgPipelineOptions,
    PgSchemaManager,
    PgSearchOptions,
    format_ingest_report_summary,
    format_search_summary,
    run_pg_pipeline,
    run_pg_search,
    write_json,
)
from waji_rag.work_order import (
    WorkOrderBatchOptions,
    WorkOrderBatchParser,
    format_report_summary as format_work_order_report_summary,
    write_report as write_work_order_report,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level command parser."""

    parser = argparse.ArgumentParser(
        prog="waji-rag",
        description="Local RAG debugging tools for excavator after-sales diagnosis.",
    )
    parser.add_argument("--version", action="version", version=f"waji-rag {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Print environment diagnostics.")
    doctor.set_defaults(func=run_doctor)

    config = subparsers.add_parser("write-default-config", help="Write a default retrieval config JSON file.")
    config.add_argument("--out", required=True, help="Config JSON path to write.")
    config.set_defaults(func=run_write_default_config)

    html_to_md = subparsers.add_parser("html-to-md", help="Convert a directory of HTML files to Markdown.")
    html_to_md.add_argument("--input-dir", required=True, help="Directory containing .html/.htm files.")
    html_to_md.add_argument("--out-dir", required=True, help="Directory where .md files will be written.")
    html_to_md.add_argument("--limit", type=int, help="Optional maximum number of files to convert.")
    html_to_md.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Skip existing Markdown files instead of overwriting them.",
    )
    html_to_md.add_argument(
        "--reference-links",
        action="store_true",
        help="Render Markdown links as reference-style links.",
    )
    html_to_md.add_argument("--report-json", help="Optional JSON report output path.")
    html_to_md.add_argument("--debug", action="store_true", help="Print failed file details.")
    html_to_md.set_defaults(func=run_html_to_md)

    work_orders = subparsers.add_parser("parse-workorders", help="Parse work-order TXT files.")
    work_orders.add_argument("--input-dir", required=True, help="Directory containing .txt work-order files.")
    work_orders.add_argument("--out-dir", required=True, help="Directory where parsed JSONL/CSV files will be written.")
    work_orders.add_argument("--limit", type=int, help="Optional maximum number of files to parse.")
    work_orders.add_argument("--report-json", help="Optional JSON report output path.")
    work_orders.add_argument("--debug", action="store_true", help="Print failed file details and warnings.")
    work_orders.set_defaults(func=run_parse_workorders)

    build_index = subparsers.add_parser("build-index", help="Build a local keyword index.")
    build_index.add_argument("--work-orders-jsonl", required=True, help="Parsed work_orders.jsonl path.")
    build_index.add_argument("--parts-jsonl", required=True, help="Parsed parts_evidence.jsonl path.")
    build_index.add_argument("--manual-md-dir", required=True, help="Directory containing cleaned Markdown manuals.")
    build_index.add_argument("--out-dir", required=True, help="Directory where index files will be written.")
    build_index.add_argument("--manual-limit", type=int, help="Optional maximum number of Markdown files to index.")
    build_index.add_argument(
        "--max-manual-chars",
        type=int,
        default=1800,
        help="Maximum characters per manual chunk. Defaults to 1800.",
    )
    build_index.add_argument("--report-json", help="Optional extra JSON report output path.")
    build_index.add_argument("--debug", action="store_true", help="Print failed item details and warnings.")
    build_index.set_defaults(func=run_build_index)

    init_db = subparsers.add_parser("init-db", help="Create PostgreSQL/pgvector schema.")
    init_db.add_argument("--database-url", help="PostgreSQL URL. Defaults to WAJI_DATABASE_URL or local Docker default.")
    init_db.add_argument("--reset", action="store_true", help="Drop and recreate application tables.")
    init_db.set_defaults(func=run_init_db)

    ingest_db = subparsers.add_parser("ingest-db", help="Ingest raw TXT/HTML/Markdown evidence into PostgreSQL.")
    ingest_db.add_argument("--database-url", help="PostgreSQL URL. Defaults to WAJI_DATABASE_URL or local Docker default.")
    ingest_db.add_argument("--work-order-dir", help="Directory containing work-order .txt files.")
    ingest_db.add_argument("--manual-dir", help="Directory containing manual .html/.htm/.md files.")
    ingest_db.add_argument("--config", help="Optional retrieval config JSON path.")
    ingest_db.add_argument("--env-file", help="Optional dotenv path for API keys and default models.")
    ingest_db.add_argument("--enable-embedding", action="store_true", help="Enable configured embeddings during ingest.")
    ingest_db.add_argument("--embedding-model", help="Embedding model name.")
    ingest_db.add_argument("--embedding-dimensions", type=int, help="Embedding dimensions.")
    ingest_db.add_argument("--reset", action="store_true", help="Reset schema before ingesting.")
    ingest_db.add_argument("--work-order-limit", type=int, help="Optional maximum number of work-order files.")
    ingest_db.add_argument("--manual-limit", type=int, help="Optional maximum number of manual files.")
    ingest_db.add_argument(
        "--max-manual-chars",
        type=int,
        default=1800,
        help="Maximum characters per manual chunk. Defaults to 1800.",
    )
    ingest_db.add_argument("--report-json", help="Optional JSON report output path.")
    ingest_db.add_argument("--debug", action="store_true", help="Print failed item details and warnings.")
    ingest_db.set_defaults(func=run_ingest_db)

    search_db = subparsers.add_parser("search-db", help="Retrieve an evidence package from PostgreSQL.")
    search_db.add_argument("--database-url", help="PostgreSQL URL. Defaults to WAJI_DATABASE_URL or local Docker default.")
    search_db.add_argument("--query", required=True, help="User diagnostic question.")
    search_db.add_argument("--config", help="Optional retrieval config JSON path.")
    search_db.add_argument("--env-file", help="Optional dotenv path for API keys and default models.")
    search_db.add_argument("--enable-embedding", action="store_true", help="Enable hybrid retrieval when embeddings exist.")
    search_db.add_argument("--enable-rerank", action="store_true", help="Reserved for full ask-db; search-db returns retrieval only.")
    search_db.add_argument("--top-k", type=int, default=8, help="Hits per channel. Defaults to 8.")
    search_db.add_argument("--out-json", help="Optional JSON output path.")
    search_db.add_argument("--debug", action="store_true", help="Include query terms and scoring settings.")
    search_db.set_defaults(func=run_search_db)

    ask_db = subparsers.add_parser("ask-db", help="Retrieve evidence, optionally rerank it, and generate an answer.")
    ask_db.add_argument("--database-url", help="PostgreSQL URL. Defaults to WAJI_DATABASE_URL or local Docker default.")
    ask_db.add_argument("--query", required=True, help="User diagnostic question.")
    ask_db.add_argument("--config", help="Optional retrieval config JSON path.")
    ask_db.add_argument("--env-file", help="Optional dotenv path for API keys and default models.")
    ask_db.add_argument("--top-k", type=int, default=8, help="Hits per channel. Defaults to 8.")
    ask_db.add_argument("--out-json", help="Optional JSON output path.")
    ask_db.add_argument("--enable-embedding", action="store_true", help="Enable hybrid retrieval when embeddings exist.")
    ask_db.add_argument("--enable-rerank", action="store_true", help="Enable configured reranker.")
    ask_db.add_argument("--enable-llm", action="store_true", help="Enable configured LLM answer generation.")
    ask_db.add_argument("--embedding-model", help="Embedding model name.")
    ask_db.add_argument("--embedding-dimensions", type=int, help="Embedding dimensions.")
    ask_db.add_argument("--rerank-model", help="Rerank model name.")
    ask_db.add_argument("--llm-model", help="LLM model name.")
    ask_db.add_argument("--debug", action="store_true", help="Include trace and scoring settings.")
    ask_db.set_defaults(func=run_ask_db)

    serve = subparsers.add_parser("serve", help="Start the local web debugging UI.")
    serve.add_argument("--host", default="127.0.0.1", help="Host to bind. Defaults to 127.0.0.1.")
    serve.add_argument("--port", type=int, default=8765, help="Port to bind. Defaults to 8765.")
    serve.set_defaults(func=run_serve)

    return parser


def run_doctor(_args: argparse.Namespace) -> int:
    """Print basic environment information for Windows feedback loops."""

    payload = {
        "waji_rag_version": __version__,
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "cwd": str(Path.cwd()),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def run_write_default_config(args: argparse.Namespace) -> int:
    """Write a default config JSON file."""

    output_path = Path(args.out)
    write_default_config(output_path)
    print(f"config={output_path.resolve()}")
    return 0


def run_html_to_md(args: argparse.Namespace) -> int:
    """Run the batch HTML-to-Markdown conversion command."""

    options = ConvertOptions(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.out_dir),
        limit=args.limit,
        overwrite=not args.no_overwrite,
        reference_links=args.reference_links,
    )
    try:
        report = HtmlToMarkdownBatch(options).convert_directory()
    except Exception as exc:  # noqa: BLE001 - top-level CLI diagnostics.
        print(f"html-to-md failed: {exc}", file=sys.stderr)
        if args.debug:
            print(failure_trace(exc), file=sys.stderr)
        return 1

    print(format_report_summary(report))
    if args.report_json:
        write_report(report, Path(args.report_json))
        print(f"report_json={Path(args.report_json).resolve()}")

    if args.debug:
        for result in report.results:
            if result.status == "failed":
                print(f"FAILED {result.input_path}: {result.error}", file=sys.stderr)
    return 1 if report.failed_files else 0


def run_build_index(args: argparse.Namespace) -> int:
    """Run the local keyword index build command."""

    options = IndexBuildOptions(
        work_orders_jsonl=Path(args.work_orders_jsonl),
        parts_jsonl=Path(args.parts_jsonl),
        manual_md_dir=Path(args.manual_md_dir),
        output_dir=Path(args.out_dir),
        manual_limit=args.manual_limit,
        max_manual_chars=args.max_manual_chars,
    )
    try:
        report = LocalIndexBuilder(options).build()
    except Exception as exc:  # noqa: BLE001 - top-level CLI diagnostics.
        print(f"build-index failed: {exc}", file=sys.stderr)
        if args.debug:
            print(failure_trace(exc), file=sys.stderr)
        return 1

    print(format_index_report_summary(report))
    for key, output_path in report.output_paths.items():
        print(f"{key}={Path(output_path).resolve()}")
    if args.report_json:
        write_index_report(report, Path(args.report_json))
        print(f"report_json={Path(args.report_json).resolve()}")

    if args.debug:
        for warning in report.warnings:
            print(f"WARN {warning}", file=sys.stderr)
        for failed_item in report.failed_items:
            print(
                f"FAILED {failed_item.get('stage')} {failed_item.get('input')}: {failed_item.get('error')}",
                file=sys.stderr,
            )
    return 1 if report.failed_items else 0


def run_init_db(args: argparse.Namespace) -> int:
    """Initialize the PostgreSQL schema."""

    try:
        payload = PgSchemaManager(DatabaseOptions.from_env(args.database_url)).initialize(reset=args.reset)
    except Exception as exc:  # noqa: BLE001 - top-level CLI diagnostics.
        print(f"init-db failed: {exc}", file=sys.stderr)
        if getattr(args, "debug", False):
            print(failure_trace(exc), file=sys.stderr)
        return 1
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def run_ingest_db(args: argparse.Namespace) -> int:
    """Ingest raw evidence into PostgreSQL."""

    options = PgIngestOptions(
        database=DatabaseOptions.from_env(args.database_url),
        work_order_dir=Path(args.work_order_dir) if args.work_order_dir else None,
        manual_dir=Path(args.manual_dir) if args.manual_dir else None,
        config_path=Path(args.config) if args.config else None,
        config_overrides=config_overrides_from_args(args),
        env_path=Path(args.env_file) if args.env_file else None,
        reset=args.reset,
        work_order_limit=args.work_order_limit,
        manual_limit=args.manual_limit,
        max_manual_chars=args.max_manual_chars,
    )
    try:
        report = PgIngestBuilder(options).ingest()
    except Exception as exc:  # noqa: BLE001 - top-level CLI diagnostics.
        print(f"ingest-db failed: {exc}", file=sys.stderr)
        if args.debug:
            print(failure_trace(exc), file=sys.stderr)
        return 1

    print(format_ingest_report_summary(report))
    if args.report_json:
        write_json(report.to_dict(), Path(args.report_json))
        print(f"report_json={Path(args.report_json).resolve()}")

    if args.debug:
        for warning in report.warnings:
            print(f"WARN {warning}", file=sys.stderr)
        for failed_item in report.failed_items:
            print(
                f"FAILED {failed_item.get('stage')} {failed_item.get('input')}: {failed_item.get('error')}",
                file=sys.stderr,
            )
    return 1 if report.failed_items else 0


def run_search_db(args: argparse.Namespace) -> int:
    """Search PostgreSQL and print a retrieval evidence package."""

    options = PgSearchOptions(
        database=DatabaseOptions.from_env(args.database_url),
        query=args.query,
        config_path=Path(args.config) if args.config else None,
        config_overrides=config_overrides_from_args(args),
        env_path=Path(args.env_file) if args.env_file else None,
        top_k=args.top_k,
        include_debug=args.debug,
    )
    try:
        payload = run_pg_search(options)
    except Exception as exc:  # noqa: BLE001 - top-level CLI diagnostics.
        print(f"search-db failed: {exc}", file=sys.stderr)
        if args.debug:
            print(failure_trace(exc), file=sys.stderr)
        return 1

    print(format_search_summary(payload))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.out_json:
        write_json(payload, Path(args.out_json))
        print(f"out_json={Path(args.out_json).resolve()}")
    return 0


def run_ask_db(args: argparse.Namespace) -> int:
    """Run full retrieve-rerank-answer pipeline."""

    options = PgPipelineOptions(
        database=DatabaseOptions.from_env(args.database_url),
        query=args.query,
        config_path=Path(args.config) if args.config else None,
        config_overrides=config_overrides_from_args(args),
        env_path=Path(args.env_file) if args.env_file else None,
        top_k=args.top_k,
        include_debug=args.debug,
    )
    try:
        payload = run_pg_pipeline(options)
    except Exception as exc:  # noqa: BLE001 - top-level CLI diagnostics.
        print(f"ask-db failed: {exc}", file=sys.stderr)
        if args.debug:
            print(failure_trace(exc), file=sys.stderr)
        return 1

    answer = payload.get("answer") if isinstance(payload.get("answer"), dict) else {}
    print(str(answer.get("text") or ""))
    if args.out_json:
        write_json(payload, Path(args.out_json))
        print(f"out_json={Path(args.out_json).resolve()}")
    elif args.debug:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def config_overrides_from_args(args: argparse.Namespace) -> dict[str, object]:
    """Build config overrides from common CLI flags."""

    overrides: dict[str, object] = {}
    embedding: dict[str, object] = {}
    if getattr(args, "enable_embedding", False):
        embedding["enabled"] = True
    if getattr(args, "embedding_model", None):
        embedding["model"] = args.embedding_model
    if getattr(args, "embedding_dimensions", None):
        embedding["dimensions"] = args.embedding_dimensions
    if embedding:
        overrides["embedding"] = embedding

    rerank: dict[str, object] = {}
    if getattr(args, "enable_rerank", False):
        rerank["enabled"] = True
    if getattr(args, "rerank_model", None):
        rerank["model"] = args.rerank_model
    if rerank:
        overrides["rerank"] = rerank

    llm: dict[str, object] = {}
    if getattr(args, "enable_llm", False):
        llm["enabled"] = True
    if getattr(args, "llm_model", None):
        llm["model"] = args.llm_model
    if llm:
        overrides["llm"] = llm
    return overrides


def run_parse_workorders(args: argparse.Namespace) -> int:
    """Run the work-order TXT parsing command."""

    options = WorkOrderBatchOptions(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.out_dir),
        limit=args.limit,
    )
    try:
        report = WorkOrderBatchParser(options).parse_directory()
    except Exception as exc:  # noqa: BLE001 - top-level CLI diagnostics.
        print(f"parse-workorders failed: {exc}", file=sys.stderr)
        if args.debug:
            print(failure_trace(exc), file=sys.stderr)
        return 1

    print(format_work_order_report_summary(report))
    print(f"work_orders_jsonl={Path(report.work_orders_jsonl or '').resolve()}")
    print(f"parts_jsonl={Path(report.parts_jsonl or '').resolve()}")
    print(f"parts_csv={Path(report.parts_csv or '').resolve()}")
    if args.report_json:
        write_work_order_report(report, Path(args.report_json))
        print(f"report_json={Path(args.report_json).resolve()}")

    if args.debug:
        for result in report.results:
            if result.get("status") == "failed":
                print(f"FAILED {result.get('input_path')}: {result.get('error')}", file=sys.stderr)
            warnings = result.get("warnings") or []
            if warnings:
                print(f"WARN {result.get('input_path')}: {', '.join(warnings)}", file=sys.stderr)
    return 1 if report.failed_files else 0


def run_serve(args: argparse.Namespace) -> int:
    """Start the local web debugging server."""

    from waji_rag.web import serve

    serve(host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface."""

    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
