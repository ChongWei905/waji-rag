"""Work-order TXT parsing and evidence export."""

from __future__ import annotations

import csv
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from waji_rag.html_batch import DEFAULT_ENCODINGS, read_text_with_fallback


FIELD_LABELS = {
    "work_order_id": (
        "工单ID",
        "工单编号",
        "工单号",
        "服务单号",
        "单号",
    ),
    "reported_issue": (
        "用户报修内容",
        "用户保修内容",
        "报修内容",
        "报修故障",
        "故障现象",
        "故障描述",
        "客户反映",
        "客户反馈",
        "用户反馈",
        "反馈问题",
    ),
    "solution": (
        "人员落实及解决方法",
        "人员落实",
        "解决方法",
        "解决方案",
        "处理措施",
        "处理结果",
        "维修过程",
        "维修方案",
        "落实情况",
    ),
    "parts": (
        "备件信息",
        "备件明细",
        "备件",
        "更换备件",
        "配件信息",
        "配件明细",
        "物料信息",
    ),
}

PART_LABELS = {
    "part_number_name": ("备件编号及名称", "配件编号及名称", "零件编号及名称"),
    "part_number": ("备件编号", "配件编号", "零件号", "零件编号", "图号"),
    "part_name": ("备件名称", "配件名称", "零件名称", "物料名称", "名称"),
    "part_code": ("备件编码", "配件编码", "物料编码", "编码"),
    "quantity": ("备件数量", "配件数量", "数量", "用量"),
}

OLD_NEW_PART_LABELS = {
    "old_part_name": ("旧件备件名称", "旧件配件名称", "旧件物料名称", "旧件名称"),
    "new_part_name": ("新件备件名称", "新件配件名称", "新件物料名称", "新件名称"),
    "old_part_code": ("旧件物料编码", "旧件备件编码", "旧件配件编码", "旧件编码"),
    "new_part_code": ("新件物料编码", "新件备件编码", "新件配件编码", "新件编码"),
    "old_quantity": ("旧件数量",),
    "new_quantity": ("新件数量",),
}

NORMALIZED_OLD_NEW_KEYS = {
    "old_part_name": "part_name",
    "new_part_name": "part_name",
    "old_part_code": "part_code",
    "new_part_code": "part_code",
    "old_quantity": "quantity",
    "new_quantity": "quantity",
}

NO_PART_PATTERNS = ("无", "未更换", "不涉及", "无需", "没有")
PART_RECORD_START_KEYS = {"part_number_name", "part_number", "part_name"}
PART_RECORD_END_KEYS = {"part_code", "quantity"}
NUMBERED_PART_PATTERN = re.compile(r"(?m)^\s*\d+\s*[.、)]\s*")


@dataclass(slots=True)
class PartRecord:
    """Structured part evidence parsed from a work order."""

    part_number_name: str | None = None
    part_number: str | None = None
    part_name: str | None = None
    part_code: str | None = None
    quantity: str | None = None
    raw_text: str = ""


@dataclass(slots=True)
class WorkOrderRecord:
    """Structured representation of one work-order TXT file."""

    doc_type: str = "work_order"
    work_order_id: str | None = None
    reported_issue: str | None = None
    solution: str | None = None
    parts: list[PartRecord] = field(default_factory=list)
    raw_text: str = ""
    source_path: str = ""
    encoding: str | None = None
    parse_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable work-order dictionary."""

        return asdict(self)


@dataclass(slots=True)
class WorkOrderParseResult:
    """Parse result for one TXT file."""

    input_path: str
    status: str
    record: WorkOrderRecord | None = None
    error: str | None = None


@dataclass(slots=True)
class WorkOrderBatchReport:
    """Summary report for a work-order parsing run."""

    started_at: str
    elapsed_seconds: float
    input_dir: str
    output_dir: str
    scanned_files: int = 0
    parsed_files: int = 0
    failed_files: int = 0
    records_with_parts: int = 0
    part_records: int = 0
    missing_reported_issue: int = 0
    missing_solution: int = 0
    work_orders_jsonl: str | None = None
    parts_jsonl: str | None = None
    parts_csv: str | None = None
    results: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable report dictionary."""

        return asdict(self)


class WorkOrderParser:
    """Parse short after-sales work-order TXT documents."""

    def parse(self, text: str, *, source_path: Path, encoding: str | None = None) -> WorkOrderRecord:
        """Parse one work-order text into stable fields and warnings."""

        normalized = normalize_text(text)
        sections = extract_sections(normalized)
        work_order_id = first_non_empty(
            sections.get("work_order_id"),
            extract_label_value(normalized, FIELD_LABELS["work_order_id"]),
            infer_id_from_filename(source_path),
        )
        reported_issue = first_non_empty(
            sections.get("reported_issue"),
            extract_label_value(normalized, FIELD_LABELS["reported_issue"]),
        )
        solution = first_non_empty(
            sections.get("solution"),
            extract_label_value(normalized, FIELD_LABELS["solution"]),
        )
        parts_text = first_non_empty(
            sections.get("parts"),
            extract_label_value(normalized, FIELD_LABELS["parts"]),
        )
        parts = parse_parts(parts_text or "")

        warnings: list[str] = []
        if not reported_issue:
            warnings.append("missing_reported_issue")
        if not solution:
            warnings.append("missing_solution")
        if parts_text and not parts and not is_no_part_text(parts_text) and not is_empty_part_template(parts_text):
            warnings.append("parts_section_without_structured_parts")
        if not work_order_id:
            warnings.append("missing_work_order_id")

        return WorkOrderRecord(
            work_order_id=work_order_id,
            reported_issue=reported_issue,
            solution=solution,
            parts=parts,
            raw_text=normalized,
            source_path=str(source_path),
            encoding=encoding,
            parse_warnings=warnings,
        )


@dataclass(slots=True)
class WorkOrderBatchOptions:
    """Configuration for a work-order parsing batch."""

    input_dir: Path
    output_dir: Path
    limit: int | None = None
    encodings: tuple[str, ...] = DEFAULT_ENCODINGS


class WorkOrderBatchParser:
    """Parse many work-order TXT files and export evidence files."""

    def __init__(self, options: WorkOrderBatchOptions) -> None:
        """Store parser options."""

        self.options = options
        self.parser = WorkOrderParser()

    def parse_directory(self) -> WorkOrderBatchReport:
        """Parse all TXT files under the configured input directory."""

        start_time = time.time()
        input_dir = self.options.input_dir.resolve()
        output_dir = self.options.output_dir.resolve()
        if not input_dir.exists():
            raise FileNotFoundError(f"input directory does not exist: {input_dir}")
        if not input_dir.is_dir():
            raise NotADirectoryError(f"input path is not a directory: {input_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        txt_files = list(iter_txt_files(input_dir))
        if self.options.limit is not None:
            txt_files = txt_files[: max(self.options.limit, 0)]

        report = WorkOrderBatchReport(
            started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            elapsed_seconds=0.0,
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            scanned_files=len(txt_files),
        )
        records: list[WorkOrderRecord] = []
        results: list[WorkOrderParseResult] = []

        for txt_path in txt_files:
            result = self.parse_file(txt_path)
            results.append(result)
            if result.status == "parsed" and result.record is not None:
                records.append(result.record)
                report.parsed_files += 1
                if result.record.parts:
                    report.records_with_parts += 1
                    report.part_records += len(result.record.parts)
                if not result.record.reported_issue:
                    report.missing_reported_issue += 1
                if not result.record.solution:
                    report.missing_solution += 1
            else:
                report.failed_files += 1

        work_orders_jsonl = output_dir / "work_orders.jsonl"
        parts_jsonl = output_dir / "parts_evidence.jsonl"
        parts_csv = output_dir / "parts_evidence.csv"
        write_work_orders_jsonl(records, work_orders_jsonl)
        write_parts_jsonl(records, parts_jsonl)
        write_parts_csv(records, parts_csv)

        report.work_orders_jsonl = str(work_orders_jsonl)
        report.parts_jsonl = str(parts_jsonl)
        report.parts_csv = str(parts_csv)
        report.results = summarize_results(results)
        report.elapsed_seconds = round(time.time() - start_time, 3)
        return report

    def parse_file(self, txt_path: Path) -> WorkOrderParseResult:
        """Parse a single TXT file with encoding fallback."""

        try:
            text, encoding = read_text_with_fallback(txt_path, self.options.encodings)
            record = self.parser.parse(text, source_path=txt_path, encoding=encoding)
            return WorkOrderParseResult(input_path=str(txt_path), status="parsed", record=record)
        except Exception as exc:  # noqa: BLE001 - keep batch parsing resilient.
            return WorkOrderParseResult(
                input_path=str(txt_path),
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )


def normalize_text(text: str) -> str:
    """Normalize whitespace and common full-width punctuation."""

    normalized = text.replace("\ufeff", "")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("：", ":")
    normalized = re.sub(r"[ \t\u3000]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def extract_sections(text: str) -> dict[str, str]:
    """Extract top-level work-order sections by known Chinese labels."""

    label_to_key = {label: key for key, labels in FIELD_LABELS.items() for label in labels}
    labels_pattern = "|".join(re.escape(label) for label in sorted(label_to_key, key=len, reverse=True))
    pattern = re.compile(rf"(?P<label>{labels_pattern})\s*[:：]\s*", flags=re.IGNORECASE)
    matches = list(pattern.finditer(text))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        key = label_to_key[match.group("label")]
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        value = clean_value(text[start:end])
        if value:
            sections[key] = value
    return sections


def extract_label_value(text: str, labels: tuple[str, ...]) -> str | None:
    """Extract a single value after the first matching label."""

    label_pattern = "|".join(re.escape(label) for label in sorted(labels, key=len, reverse=True))
    other_labels = tuple(label for labels_ in FIELD_LABELS.values() for label in labels_)
    stop_pattern = "|".join(re.escape(label) for label in sorted(other_labels, key=len, reverse=True))
    pattern = re.compile(
        rf"(?:{label_pattern})\s*[:：]\s*(?P<value>.*?)(?=\n?\s*(?:{stop_pattern})\s*[:：]|$)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        return None
    return clean_value(match.group("value"))


def parse_parts(parts_text: str) -> list[PartRecord]:
    """Parse part records from a work-order parts section."""

    parts_text = clean_value(parts_text)
    if not parts_text or is_no_part_text(parts_text):
        return []

    numbered_parts = parse_numbered_block_parts(parts_text)
    if numbered_parts:
        return numbered_parts

    table_parts = parse_table_like_parts(parts_text)
    if table_parts:
        return table_parts

    chunks = split_part_chunks(parts_text)
    parts: list[PartRecord] = []
    for chunk in chunks:
        part = parse_part_chunk(chunk)
        if has_part_signal(part):
            parts.append(part)
    if not parts and parts_text and not is_empty_part_template(parts_text):
        parts.append(PartRecord(raw_text=parts_text))
    return parts


def parse_table_like_parts(parts_text: str) -> list[PartRecord]:
    """Parse simple tabular part sections with header and row lines."""

    lines = [line.strip() for line in parts_text.splitlines() if line.strip()]
    if len(lines) < 2:
        return []

    header_index = next((index for index, line in enumerate(lines) if looks_like_part_header(line)), None)
    if header_index is None or header_index + 1 >= len(lines):
        return []

    headers = split_columns(lines[header_index])
    if len(headers) < 2:
        return []

    header_keys = [part_key_from_header(header) for header in headers]
    if not any(header_keys):
        return []

    parts: list[PartRecord] = []
    for line in lines[header_index + 1 :]:
        if is_no_part_text(line):
            continue
        cells = split_columns(line)
        cells = align_cells_to_headers(cells, header_keys)
        if len(cells) < 2:
            continue
        part = PartRecord(raw_text=line)
        for key, value in zip(header_keys, cells, strict=False):
            if key and value:
                setattr(part, key, clean_value(value))
        if has_part_signal(part):
            parts.append(part)
    return parts


def parse_part_chunk(chunk: str) -> PartRecord:
    """Parse one part chunk with label-based extraction."""

    old_new_part = parse_old_new_part_block(chunk)
    if has_part_identity(old_new_part):
        return old_new_part

    return PartRecord(
        part_number_name=extract_part_value(chunk, PART_LABELS["part_number_name"]),
        part_number=extract_part_value(chunk, PART_LABELS["part_number"]),
        part_name=extract_part_value(chunk, PART_LABELS["part_name"]),
        part_code=extract_part_value(chunk, PART_LABELS["part_code"]),
        quantity=extract_part_value(chunk, PART_LABELS["quantity"]),
        raw_text=clean_value(chunk),
    )


def extract_part_value(text: str, labels: tuple[str, ...]) -> str | None:
    """Extract one part field value by label."""

    label_pattern = "|".join(re.escape(label) for label in sorted(labels, key=len, reverse=True))
    all_labels = tuple(label for labels_ in all_part_label_sets() for label in labels_)
    stop_pattern = "|".join(re.escape(label) for label in sorted(all_labels, key=len, reverse=True))
    pattern = re.compile(
        rf"(?:^|(?<=[;；,\n\s]))(?:{label_pattern})\s*[:：]\s*"
        rf"(?P<value>.*?)(?=\s*(?:{stop_pattern})(?:\s*[:：]|$)|[;；,\n]|$)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        return None
    return clean_value(match.group("value"))


def parse_numbered_block_parts(parts_text: str) -> list[PartRecord]:
    """Parse numbered old/new part blocks such as ``1. 新件备件名称: ...``."""

    blocks = split_numbered_part_blocks(parts_text)
    parts: list[PartRecord] = []
    for block in blocks:
        part = parse_old_new_part_block(block)
        if has_part_identity(part):
            parts.append(part)
    return parts


def parse_old_new_part_block(block: str) -> PartRecord:
    """Parse one old/new part block and prefer new-part fields over old-part fields."""

    old_part_name = extract_part_value(block, OLD_NEW_PART_LABELS["old_part_name"])
    new_part_name = extract_part_value(block, OLD_NEW_PART_LABELS["new_part_name"])
    old_part_code = extract_part_value(block, OLD_NEW_PART_LABELS["old_part_code"])
    new_part_code = extract_part_value(block, OLD_NEW_PART_LABELS["new_part_code"])
    old_quantity = extract_part_value(block, OLD_NEW_PART_LABELS["old_quantity"])
    new_quantity = extract_part_value(block, OLD_NEW_PART_LABELS["new_quantity"])

    return PartRecord(
        part_name=first_non_empty(new_part_name, old_part_name),
        part_code=first_non_empty(new_part_code, old_part_code),
        quantity=first_non_empty(new_quantity, old_quantity),
        raw_text=clean_value(block),
    )


def split_numbered_part_blocks(parts_text: str) -> list[str]:
    """Split a parts section into numbered blocks."""

    matches = list(NUMBERED_PART_PATTERN.finditer(parts_text))
    blocks: list[str] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(parts_text)
        block = clean_value(parts_text[start:end])
        if block:
            blocks.append(block)
    return blocks


def split_part_chunks(parts_text: str) -> list[str]:
    """Split a parts section into likely per-part chunks."""

    lines = [line.strip() for line in parts_text.splitlines() if line.strip()]
    if len(lines) > 1:
        return lines
    return split_inline_part_records(parts_text)


def split_inline_part_records(parts_text: str) -> list[str]:
    """Split one-line part text while keeping fields for one part together."""

    segments = [segment.strip() for segment in re.split(r"[;；]\s*", parts_text) if segment.strip()]
    if len(segments) <= 1:
        return [parts_text.strip()] if parts_text.strip() else []

    chunks: list[str] = []
    current: list[str] = []
    seen_keys: set[str] = set()
    for segment in segments:
        key = first_part_key_in_segment(segment)
        starts_new_record = bool(
            current
            and key
            and (
                key in seen_keys
                or (key in PART_RECORD_START_KEYS and bool(seen_keys & PART_RECORD_END_KEYS))
            )
        )
        if starts_new_record:
            chunks.append("; ".join(current))
            current = [segment]
            seen_keys = {key}
            continue

        current.append(segment)
        if key:
            seen_keys.add(key)

    if current:
        chunks.append("; ".join(current))
    return chunks


def first_part_key_in_segment(segment: str) -> str | None:
    """Return the first labeled part field key found in a text segment."""

    for key, labels in OLD_NEW_PART_LABELS.items():
        label_pattern = "|".join(re.escape(label) for label in sorted(labels, key=len, reverse=True))
        if re.search(rf"(?:^|(?<=[;；,\n\s]))(?:{label_pattern})\s*[:：]", segment, flags=re.IGNORECASE):
            return NORMALIZED_OLD_NEW_KEYS[key]

    for key, labels in PART_LABELS.items():
        label_pattern = "|".join(re.escape(label) for label in sorted(labels, key=len, reverse=True))
        if re.search(rf"(?:^|(?<=[;；,\n\s]))(?:{label_pattern})\s*[:：]", segment, flags=re.IGNORECASE):
            return key
    return None


def split_columns(line: str) -> list[str]:
    """Split a simple table row by common text delimiters."""

    stripped = line.strip().strip("|")
    if "|" in stripped:
        return [cell.strip() for cell in stripped.split("|")]
    if "\t" in stripped:
        return [cell.strip() for cell in stripped.split("\t")]
    if re.search(r"\s{2,}", stripped):
        return [cell.strip() for cell in re.split(r"\s{2,}", stripped) if cell.strip()]
    return stripped.split()


def align_cells_to_headers(cells: list[str], header_keys: list[str | None]) -> list[str]:
    """Merge extra leading cells so row values align with parsed table headers."""

    if len(cells) <= len(header_keys):
        return cells
    extra_cell_count = len(cells) - len(header_keys)
    first_cell = " ".join(cells[: extra_cell_count + 1])
    return [first_cell, *cells[extra_cell_count + 1 :]]


def looks_like_part_header(line: str) -> bool:
    """Return whether a line looks like a part table header."""

    return sum(1 for labels in PART_LABELS.values() for label in labels if label in line) >= 2


def part_key_from_header(header: str) -> str | None:
    """Map a table header cell to a part field key."""

    for key, labels in PART_LABELS.items():
        if any(label in header for label in labels):
            return key
    return None


def has_part_signal(part: PartRecord) -> bool:
    """Return whether a parsed part record contains useful evidence."""

    return any(
        (
            part.part_number_name,
            part.part_number,
            part.part_name,
            part.part_code,
            part.quantity,
        )
    )


def has_part_identity(part: PartRecord) -> bool:
    """Return whether a part record has an identifying name, number, or code."""

    return any((part.part_number_name, part.part_number, part.part_name, part.part_code))


def is_no_part_text(text: str) -> bool:
    """Return whether the parts text means no parts were used."""

    compact = re.sub(r"\s+", "", text)
    return compact in NO_PART_PATTERNS or any(pattern == compact for pattern in NO_PART_PATTERNS)


def is_empty_part_template(text: str) -> bool:
    """Return whether text looks like an empty structured parts template."""

    blocks = split_numbered_part_blocks(clean_value(text))
    if not blocks:
        return False
    return all(not has_any_labeled_part_value(block) for block in blocks)


def has_any_labeled_part_value(text: str) -> bool:
    """Return whether any known part label has a non-empty value."""

    return any(extract_part_value(text, labels) for labels in all_part_label_sets())


def all_part_label_sets() -> tuple[tuple[str, ...], ...]:
    """Return all label sets that can terminate a part-field value."""

    return (*PART_LABELS.values(), *OLD_NEW_PART_LABELS.values())


def first_non_empty(*values: str | None) -> str | None:
    """Return the first non-empty cleaned string."""

    for value in values:
        cleaned = clean_value(value or "")
        if cleaned:
            return cleaned
    return None


def clean_value(value: str) -> str:
    """Clean a parsed field value without changing its meaning."""

    value = value.strip()
    value = re.sub(r"\n\s*", "\n", value)
    value = re.sub(r"[ \t]+", " ", value)
    return value.strip(" \t\n:：,，;；")


def infer_id_from_filename(path: Path) -> str | None:
    """Infer a work-order id from the file stem as a fallback."""

    stem = path.stem.strip()
    if not stem:
        return None
    match = re.search(r"[A-Za-z]*\d[\w-]*", stem)
    return match.group(0) if match else stem


def iter_txt_files(root: Path) -> Iterable[Path]:
    """Yield TXT files under root in deterministic order."""

    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() == ".txt":
            yield path


def write_work_orders_jsonl(records: list[WorkOrderRecord], path: Path) -> None:
    """Write parsed work orders as JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def write_parts_jsonl(records: list[WorkOrderRecord], path: Path) -> None:
    """Write part evidence as JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for evidence in iter_part_evidence(records):
            handle.write(json.dumps(evidence, ensure_ascii=False) + "\n")


def write_parts_csv(records: list[WorkOrderRecord], path: Path) -> None:
    """Write part evidence as a UTF-8-SIG CSV for Excel on Windows."""

    fieldnames = [
        "source",
        "work_order_id",
        "reported_issue",
        "solution",
        "part_number_name",
        "part_number",
        "part_name",
        "part_code",
        "quantity",
        "raw_text",
        "source_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for evidence in iter_part_evidence(records):
            writer.writerow(evidence)


def iter_part_evidence(records: list[WorkOrderRecord]) -> Iterable[dict[str, str | None]]:
    """Yield flattened part evidence dictionaries."""

    for record in records:
        for part in record.parts:
            yield {
                "source": "work_order",
                "work_order_id": record.work_order_id,
                "reported_issue": record.reported_issue,
                "solution": record.solution,
                "part_number_name": part.part_number_name,
                "part_number": part.part_number,
                "part_name": part.part_name,
                "part_code": part.part_code,
                "quantity": part.quantity,
                "raw_text": part.raw_text,
                "source_path": record.source_path,
            }


def summarize_results(results: list[WorkOrderParseResult]) -> list[dict[str, object]]:
    """Return compact per-file parse results for reports."""

    summary: list[dict[str, object]] = []
    for result in results:
        item: dict[str, object] = {"input_path": result.input_path, "status": result.status}
        if result.error:
            item["error"] = result.error
        if result.record is not None:
            item["work_order_id"] = result.record.work_order_id
            item["part_count"] = len(result.record.parts)
            item["warnings"] = result.record.parse_warnings
        summary.append(item)
    return summary


def write_report(report: WorkOrderBatchReport, path: Path) -> None:
    """Write a work-order parse report as UTF-8 JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def format_report_summary(report: WorkOrderBatchReport) -> str:
    """Return a compact human-readable report summary."""

    return (
        f"scanned={report.scanned_files}, parsed={report.parsed_files}, "
        f"failed={report.failed_files}, records_with_parts={report.records_with_parts}, "
        f"part_records={report.part_records}, missing_reported_issue={report.missing_reported_issue}, "
        f"missing_solution={report.missing_solution}, elapsed={report.elapsed_seconds}s"
    )
