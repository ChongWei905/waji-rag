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
