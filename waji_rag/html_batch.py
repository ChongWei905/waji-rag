"""Batch HTML-to-Markdown conversion with Windows-friendly diagnostics."""

from __future__ import annotations

import json
import re
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from html_to_markdown import MarkdownConverter, detect_table_count


DEFAULT_ENCODINGS = ("utf-8", "utf-8-sig", "gb18030", "gbk", "cp936", "big5", "cp950")
HTML_CHARSET_PATTERN = re.compile(br"charset\s*=\s*['\"]?\s*([A-Za-z0-9._:-]+)", re.IGNORECASE)
BOM_ENCODINGS = (
    (b"\xff\xfe\x00\x00", "utf-32-le"),
    (b"\x00\x00\xfe\xff", "utf-32-be"),
    (b"\xef\xbb\xbf", "utf-8-sig"),
    (b"\xff\xfe", "utf-16-le"),
    (b"\xfe\xff", "utf-16-be"),
)
ENCODING_ALIASES = {
    "gb2312": "gb18030",
    "gb_2312-80": "gb18030",
    "gb-2312": "gb18030",
    "x-gbk": "gbk",
    "unicode": "utf-16",
}


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

    raw = path.read_bytes()
    errors: list[str] = []
    for encoding in candidate_encodings(raw, encodings):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
        except LookupError as exc:
            errors.append(f"{encoding}: {exc}")

    text = raw.decode("gb18030", errors="replace")
    return text, "gb18030-replace"


def candidate_encodings(raw: bytes, encodings: tuple[str, ...]) -> tuple[str, ...]:
    """Return ordered encoding candidates detected from bytes and configuration."""

    candidates: list[str] = []
    for marker, encoding in BOM_ENCODINGS:
        if raw.startswith(marker):
            candidates.append(encoding)

    utf16_guess = guess_utf16_encoding(raw)
    if utf16_guess:
        candidates.append(utf16_guess)

    charset = detect_declared_charset(raw)
    if charset:
        candidates.append(charset)

    candidates.extend(encodings)
    return unique_encoding_candidates(candidates)


def detect_declared_charset(raw: bytes) -> str | None:
    """Read a declared HTML charset from the first bytes when present."""

    head = raw[:4096]
    match = HTML_CHARSET_PATTERN.search(head)
    if not match:
        return None
    charset = match.group(1).decode("ascii", errors="ignore").strip().lower()
    return ENCODING_ALIASES.get(charset, charset) or None


def guess_utf16_encoding(raw: bytes) -> str | None:
    """Guess UTF-16 endianness for BOM-less HTML exported by Windows tools."""

    sample = raw[:4096]
    if len(sample) < 4:
        return None
    even_positions = sample[0::2]
    odd_positions = sample[1::2]
    if not even_positions or not odd_positions:
        return None
    even_null_ratio = even_positions.count(0) / len(even_positions)
    odd_null_ratio = odd_positions.count(0) / len(odd_positions)
    if odd_null_ratio > 0.25 and even_null_ratio < 0.05:
        return "utf-16-le"
    if even_null_ratio > 0.25 and odd_null_ratio < 0.05:
        return "utf-16-be"
    return None


def unique_encoding_candidates(encodings: Iterable[str]) -> tuple[str, ...]:
    """Deduplicate encoding names while preserving order."""

    seen: set[str] = set()
    result: list[str] = []
    for encoding in encodings:
        normalized = ENCODING_ALIASES.get(encoding.strip().lower(), encoding.strip().lower())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return tuple(result)


def is_lossy_encoding(encoding: str | None) -> bool:
    """Return true when a decoded file used replacement characters."""

    return bool(encoding and encoding.endswith("-replace"))


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
