"""Batch HTML-to-Markdown conversion with Windows-friendly diagnostics."""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from html_to_markdown import MarkdownConverter, detect_table_count


DEFAULT_ENCODINGS = ("utf-8", "utf-8-sig", "gb18030", "gbk")


@dataclass(slots=True)
class ConvertOptions:
    """Configuration for one batch conversion run."""

    input_dir: Path
    output_dir: Path
    limit: int | None = None
    overwrite: bool = True
    reference_links: bool = False
    encodings: tuple[str, ...] = DEFAULT_ENCODINGS


@dataclass(slots=True)
class FileConvertResult:
    """Conversion result for a single HTML file."""

    input_path: str
    output_path: str | None
    status: str
    encoding: str | None = None
    input_chars: int = 0
    output_chars: int = 0
    table_count: int = 0
    error: str | None = None


@dataclass(slots=True)
class BatchReport:
    """Summary report for a batch conversion run."""

    started_at: str
    elapsed_seconds: float
    input_dir: str
    output_dir: str
    scanned_files: int = 0
    converted_files: int = 0
    skipped_files: int = 0
    failed_files: int = 0
    results: list[FileConvertResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable report dictionary."""

        return asdict(self)


class HtmlToMarkdownBatch:
    """Convert many HTML files to Markdown while preserving relative paths."""

    def __init__(self, options: ConvertOptions) -> None:
        """Store normalized conversion options."""

        self.options = options
        self.converter = MarkdownConverter(reference_links=options.reference_links)

    def convert_directory(self) -> BatchReport:
        """Convert all HTML files under the configured input directory."""

        start_time = time.time()
        input_dir = self.options.input_dir.resolve()
        output_dir = self.options.output_dir.resolve()
        if not input_dir.exists():
            raise FileNotFoundError(f"input directory does not exist: {input_dir}")
        if not input_dir.is_dir():
            raise NotADirectoryError(f"input path is not a directory: {input_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        html_files = list(iter_html_files(input_dir))
        if self.options.limit is not None:
            html_files = html_files[: max(self.options.limit, 0)]

        report = BatchReport(
            started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            elapsed_seconds=0.0,
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            scanned_files=len(html_files),
        )

        for html_path in html_files:
            result = self.convert_file(html_path, input_dir=input_dir, output_dir=output_dir)
            report.results.append(result)
            if result.status == "converted":
                report.converted_files += 1
            elif result.status == "skipped":
                report.skipped_files += 1
            else:
                report.failed_files += 1

        report.elapsed_seconds = round(time.time() - start_time, 3)
        return report

    def convert_file(self, html_path: Path, *, input_dir: Path, output_dir: Path) -> FileConvertResult:
        """Convert one HTML file and return a detailed result object."""

        relative_path = html_path.relative_to(input_dir)
        markdown_path = output_dir / relative_path.with_suffix(".md")
        if markdown_path.exists() and not self.options.overwrite:
            return FileConvertResult(
                input_path=str(html_path),
                output_path=str(markdown_path),
                status="skipped",
            )

        try:
            html, encoding = read_text_with_fallback(html_path, self.options.encodings)
            markdown = self.converter.convert(html)
            markdown_path.parent.mkdir(parents=True, exist_ok=True)
            markdown_path.write_text(markdown, encoding="utf-8", newline="\n")
            return FileConvertResult(
                input_path=str(html_path),
                output_path=str(markdown_path),
                status="converted",
                encoding=encoding,
                input_chars=len(html),
                output_chars=len(markdown),
                table_count=detect_table_count(html),
            )
        except Exception as exc:  # noqa: BLE001 - report all per-file failures.
            return FileConvertResult(
                input_path=str(html_path),
                output_path=str(markdown_path),
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )


def iter_html_files(root: Path) -> Iterable[Path]:
    """Yield HTML files under root in deterministic order."""

    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".html", ".htm"}:
            yield path


def read_text_with_fallback(path: Path, encodings: tuple[str, ...]) -> tuple[str, str]:
    """Read text with several encodings common on Windows Chinese datasets."""

    errors: list[str] = []
    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding), encoding
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    error_preview = "; ".join(errors[:3])
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"failed encodings: {error_preview}")


def write_report(report: BatchReport, path: Path) -> None:
    """Write a batch report as UTF-8 JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def format_report_summary(report: BatchReport) -> str:
    """Return a compact human-readable report summary."""

    return (
        f"scanned={report.scanned_files}, converted={report.converted_files}, "
        f"skipped={report.skipped_files}, failed={report.failed_files}, "
        f"elapsed={report.elapsed_seconds}s"
    )


def failure_trace(exc: BaseException) -> str:
    """Return a traceback string for top-level diagnostics."""

    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
