"""Build local keyword indexes for RAG retrieval debugging."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from waji_rag.html_batch import DEFAULT_ENCODINGS, read_text_with_fallback


ASCII_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
CHINESE_SEQUENCE_PATTERN = re.compile(r"[\u4e00-\u9fff]+")
FAULT_CODE_PATTERN = re.compile(r"^(?P<code>[A-Za-z]\d{3,}[A-Za-z0-9_-]*)\s*(?P<description>.*)$")
MARKDOWN_HEADING_PATTERN = re.compile(r"^\s{0,3}#{1,6}\s+")

FIELD_WEIGHTS = {
    "reported_issue": 5.0,
    "fault_title": 5.0,
    "file_name": 4.0,
    "part_name": 4.0,
    "part_number_name": 4.0,
    "part_code": 4.0,
    "part_number": 4.0,
    "fault_code": 5.0,
    "fault_description": 4.0,
    "solution": 2.0,
    "remarks": 1.0,
    "chunk_text": 2.0,
    "raw_text": 1.0,
    "path_text": 1.0,
}


@dataclass(slots=True)
class IndexDocument:
    """One searchable document in the local keyword index."""

    doc_id: str
    doc_type: str
    title: str
    text: str
    fields: dict[str, str]
    metadata: dict[str, object] = field(default_factory=dict)
    source_path: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable document dictionary."""

        return asdict(self)


@dataclass(slots=True)
class IndexBuildOptions:
    """Configuration for building the local keyword index."""

    work_orders_jsonl: Path
    parts_jsonl: Path
    manual_md_dir: Path
    output_dir: Path
    manual_limit: int | None = None
    max_manual_chars: int = 1800
    encodings: tuple[str, ...] = DEFAULT_ENCODINGS


@dataclass(slots=True)
class IndexBuildReport:
    """Summary report for a local index build."""

    started_at: str
    elapsed_seconds: float
    inputs: dict[str, str]
    output_dir: str
    work_orders: int = 0
    part_records: int = 0
    manual_files: int = 0
    manual_chunks: int = 0
    total_documents: int = 0
    index_terms: int = 0
    missing_work_order_id: int = 0
    empty_manual_files: int = 0
    failed_items: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    output_paths: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable report dictionary."""

        return asdict(self)


class LocalIndexBuilder:
    """Build a local JSONL document store and JSON inverted index."""

    def __init__(self, options: IndexBuildOptions) -> None:
        """Store index build options."""

        self.options = options

    def build(self) -> IndexBuildReport:
        """Build document files, inverted index, manifest, and report."""

        start_time = time.time()
        work_orders_jsonl = self.options.work_orders_jsonl.resolve()
        parts_jsonl = self.options.parts_jsonl.resolve()
        manual_md_dir = self.options.manual_md_dir.resolve()
        output_dir = self.options.output_dir.resolve()
        validate_inputs(work_orders_jsonl, parts_jsonl, manual_md_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report = IndexBuildReport(
            started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            elapsed_seconds=0.0,
            inputs={
                "work_orders_jsonl": str(work_orders_jsonl),
                "parts_jsonl": str(parts_jsonl),
                "manual_md_dir": str(manual_md_dir),
            },
            output_dir=str(output_dir),
        )

        work_order_docs = build_work_order_documents(work_orders_jsonl, report)
        part_docs = build_part_documents(parts_jsonl, report)
        manual_docs = build_manual_documents(
            manual_md_dir,
            report,
            max_chars=max(self.options.max_manual_chars, 200),
            limit=self.options.manual_limit,
            encodings=self.options.encodings,
        )
        all_docs = ensure_unique_doc_ids([*work_order_docs, *part_docs, *manual_docs])
        inverted_index = build_inverted_index(all_docs)

        output_paths = {
            "index_manifest": output_dir / "index_manifest.json",
            "documents_jsonl": output_dir / "documents.jsonl",
            "work_order_docs_jsonl": output_dir / "work_order_docs.jsonl",
            "part_docs_jsonl": output_dir / "part_docs.jsonl",
            "manual_docs_jsonl": output_dir / "manual_docs.jsonl",
            "inverted_index_json": output_dir / "inverted_index.json",
            "index_report_json": output_dir / "index_report.json",
        }

        write_documents_jsonl(work_order_docs, output_paths["work_order_docs_jsonl"])
        write_documents_jsonl(part_docs, output_paths["part_docs_jsonl"])
        write_documents_jsonl(manual_docs, output_paths["manual_docs_jsonl"])
        write_documents_jsonl(all_docs, output_paths["documents_jsonl"])
        write_json(inverted_index, output_paths["inverted_index_json"])

        report.work_orders = len(work_order_docs)
        report.part_records = len(part_docs)
        report.manual_chunks = len(manual_docs)
        report.total_documents = len(all_docs)
        report.index_terms = len(inverted_index)
        report.output_paths = {key: str(path) for key, path in output_paths.items()}
        report.elapsed_seconds = round(time.time() - start_time, 3)

        manifest = build_manifest(report)
        write_json(manifest, output_paths["index_manifest"])
        write_report(report, output_paths["index_report_json"])
        return report


def validate_inputs(work_orders_jsonl: Path, parts_jsonl: Path, manual_md_dir: Path) -> None:
    """Validate build input paths before doing any output work."""

    if not work_orders_jsonl.exists():
        raise FileNotFoundError(f"work_orders_jsonl does not exist: {work_orders_jsonl}")
    if not work_orders_jsonl.is_file():
        raise FileNotFoundError(f"work_orders_jsonl is not a file: {work_orders_jsonl}")
    if not parts_jsonl.exists():
        raise FileNotFoundError(f"parts_jsonl does not exist: {parts_jsonl}")
    if not parts_jsonl.is_file():
        raise FileNotFoundError(f"parts_jsonl is not a file: {parts_jsonl}")
    if not manual_md_dir.exists():
        raise FileNotFoundError(f"manual_md_dir does not exist: {manual_md_dir}")
    if not manual_md_dir.is_dir():
        raise NotADirectoryError(f"manual_md_dir is not a directory: {manual_md_dir}")


def build_work_order_documents(path: Path, report: IndexBuildReport) -> list[IndexDocument]:
    """Build searchable documents from parsed work-order JSONL."""

    docs: list[IndexDocument] = []
    for line_number, record in read_jsonl_records(path, report, input_name="work_orders_jsonl"):
        work_order_id = clean_string(record.get("work_order_id"))
        if not work_order_id:
            report.missing_work_order_id += 1
            report.warnings.append(f"missing_work_order_id: {path}:{line_number}")

        reported_issue = clean_string(record.get("reported_issue"))
        solution = clean_string(record.get("solution"))
        remarks = clean_string(record.get("remarks"))
        raw_text = clean_string(record.get("raw_text"))
        source_path = clean_string(record.get("source_path"))
        fallback_id = short_hash(f"{path}:{line_number}:{source_path}:{reported_issue}")
        doc_id = f"wo:{work_order_id or fallback_id}"
        fields = {
            "reported_issue": reported_issue,
            "solution": solution,
            "remarks": remarks,
            "raw_text": raw_text,
        }
        docs.append(
            IndexDocument(
                doc_id=doc_id,
                doc_type="work_order",
                title=reported_issue or work_order_id or f"work order {line_number}",
                text=join_text(fields.values()),
                fields=fields,
                metadata={
                    "work_order_id": work_order_id,
                    "line_number": line_number,
                    "part_count": len(record.get("parts") or []),
                },
                source_path=source_path,
            )
        )
    return docs


def build_part_documents(path: Path, report: IndexBuildReport) -> list[IndexDocument]:
    """Build searchable documents from part-evidence JSONL."""

    docs: list[IndexDocument] = []
    for line_number, record in read_jsonl_records(path, report, input_name="parts_jsonl"):
        work_order_id = clean_string(record.get("work_order_id"))
        source_path = clean_string(record.get("source_path"))
        part_number_name = clean_string(record.get("part_number_name"))
        part_number = clean_string(record.get("part_number"))
        part_name = clean_string(record.get("part_name"))
        part_code = clean_string(record.get("part_code"))
        quantity = clean_string(record.get("quantity"))
        raw_text = clean_string(record.get("raw_text"))
        fields = {
            "reported_issue": clean_string(record.get("reported_issue")),
            "solution": clean_string(record.get("solution")),
            "remarks": clean_string(record.get("remarks")),
            "part_number_name": part_number_name,
            "part_number": part_number,
            "part_name": part_name,
            "part_code": part_code,
            "quantity": quantity,
            "raw_text": raw_text,
        }
        base_id = work_order_id or short_hash(f"{path}:{line_number}:{source_path}")
        docs.append(
            IndexDocument(
                doc_id=f"part:{base_id}:{line_number}",
                doc_type="part_evidence",
                title=first_non_empty(part_name, part_number_name, part_code, part_number, f"part {line_number}") or "",
                text=join_text(fields.values()),
                fields=fields,
                metadata={
                    "work_order_id": work_order_id,
                    "line_number": line_number,
                    "quantity": quantity,
                },
                source_path=source_path,
            )
        )
    return docs


def build_manual_documents(
    root: Path,
    report: IndexBuildReport,
    *,
    max_chars: int,
    limit: int | None,
    encodings: tuple[str, ...],
) -> list[IndexDocument]:
    """Build searchable documents from Markdown manual files."""

    docs: list[IndexDocument] = []
    md_files = list(iter_markdown_files(root))
    if limit is not None:
        md_files = md_files[: max(limit, 0)]
    report.manual_files = len(md_files)

    for md_path in md_files:
        try:
            markdown_text, encoding = read_text_with_fallback(md_path, encodings)
            metadata = infer_manual_metadata(md_path, root)
            chunks = chunk_markdown(markdown_text, max_chars=max_chars)
            if not chunks:
                report.empty_manual_files += 1
                report.warnings.append(f"empty_manual_file: {md_path}")
                continue
            for chunk_index, chunk_text in enumerate(chunks):
                relative_path = str(md_path.relative_to(root))
                doc_id = f"manual:{short_hash(relative_path)}:{chunk_index}"
                fields = {
                    "fault_title": clean_string(metadata.get("fault_title")),
                    "fault_code": clean_string(metadata.get("fault_code")),
                    "fault_description": clean_string(metadata.get("fault_description")),
                    "file_name": clean_string(metadata.get("file_name")),
                    "path_text": " ".join(md_path.relative_to(root).parts),
                    "chunk_text": chunk_text,
                }
                docs.append(
                    IndexDocument(
                        doc_id=doc_id,
                        doc_type=str(metadata["doc_type"]),
                        title=first_non_empty(fields["fault_title"], fields["fault_description"], md_path.stem) or "",
                        text=join_text(fields.values()),
                        fields=fields,
                        metadata={
                            **metadata,
                            "chunk_index": chunk_index,
                            "chunk_count": len(chunks),
                            "encoding": encoding,
                        },
                        source_path=str(md_path),
                    )
                )
        except Exception as exc:  # noqa: BLE001 - keep per-file diagnostics.
            report.failed_items.append(
                {
                    "input": str(md_path),
                    "stage": "manual_md",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return docs


def read_jsonl_records(
    path: Path,
    report: IndexBuildReport,
    *,
    input_name: str,
) -> Iterable[tuple[int, dict[str, object]]]:
    """Yield JSON object records from JSONL while recording bad lines."""

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                report.failed_items.append(
                    {
                        "input": f"{path}:{line_number}",
                        "stage": input_name,
                        "error": f"JSONDecodeError: {exc}",
                    }
                )
                continue
            if not isinstance(payload, dict):
                report.failed_items.append(
                    {
                        "input": f"{path}:{line_number}",
                        "stage": input_name,
                        "error": "record is not a JSON object",
                    }
                )
                continue
            yield line_number, payload


def build_inverted_index(documents: list[IndexDocument]) -> dict[str, list[dict[str, object]]]:
    """Build a field-aware inverted index from documents."""

    postings: dict[str, list[dict[str, object]]] = defaultdict(list)
    for document in documents:
        for field_name, field_text in document.fields.items():
            counts = Counter(tokenize_text(field_text))
            for term, term_frequency in sorted(counts.items()):
                postings[term].append(
                    {
                        "doc_id": document.doc_id,
                        "field": field_name,
                        "tf": term_frequency,
                    }
                )

    return {
        term: sorted(term_postings, key=lambda posting: (str(posting["doc_id"]), str(posting["field"])))
        for term, term_postings in sorted(postings.items())
    }


def tokenize_text(text: str) -> list[str]:
    """Tokenize text into ASCII/code tokens and Chinese 2/3-grams."""

    normalized = normalize_index_text(text)
    terms: list[str] = []
    for match in ASCII_TOKEN_PATTERN.finditer(normalized):
        token = match.group(0).strip("._-").lower()
        if token:
            terms.append(token)

    for match in CHINESE_SEQUENCE_PATTERN.finditer(normalized):
        sequence = match.group(0)
        if len(sequence) == 1:
            terms.append(sequence)
            continue
        if len(sequence) <= 12:
            terms.append(sequence)
        for ngram_size in (2, 3):
            if len(sequence) < ngram_size:
                continue
            terms.extend(sequence[index : index + ngram_size] for index in range(0, len(sequence) - ngram_size + 1))
    return terms


def normalize_index_text(text: object) -> str:
    """Normalize index text while preserving diagnostic meaning."""

    normalized = str(text or "")
    normalized = normalized.replace("\ufeff", "")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("：", ":")
    normalized = re.sub(r"[ \t\u3000]+", " ", normalized)
    return normalized.strip().lower()


def infer_manual_metadata(md_path: Path, root: Path) -> dict[str, object]:
    """Infer manual metadata from Markdown path and filename."""

    relative_parts = md_path.relative_to(root).parts
    file_name = md_path.name
    stem = md_path.stem.strip()
    machine_type = relative_parts[0] if len(relative_parts) >= 2 else None
    manual_section = next(
        (part for part in relative_parts if "典型故障解析" in part or "机器故障代码解析" in part),
        None,
    )
    typical_index = next((index for index, part in enumerate(relative_parts) if "典型故障解析" in part), None)
    system_dir = (
        relative_parts[typical_index + 1]
        if typical_index is not None and typical_index + 1 < len(relative_parts) - 1
        else None
    )
    fault_code, fault_description = infer_fault_code(stem)
    fault_title = infer_fault_title(stem, fault_description=fault_description)
    doc_type = "manual_fault_code" if fault_code or manual_section == "机器故障代码解析" else "manual_typical_fault"
    return {
        "machine_type": machine_type,
        "manual_section": manual_section,
        "system_dir": system_dir,
        "file_name": file_name,
        "fault_title": fault_title,
        "fault_code": fault_code,
        "fault_description": fault_description,
        "doc_type": doc_type,
        "relative_path": str(md_path.relative_to(root)),
    }


def infer_fault_code(stem: str) -> tuple[str | None, str | None]:
    """Infer a machine fault code and description from a filename stem."""

    match = FAULT_CODE_PATTERN.match(stem)
    if not match:
        return None, None
    return match.group("code").upper(), clean_string(match.group("description"))


def infer_fault_title(stem: str, *, fault_description: str | None) -> str:
    """Infer the fault title from a manual filename stem."""

    title = re.sub(r"^故障现象\s*[:：]\s*", "", stem).strip()
    if fault_description and title == stem:
        return fault_description
    return title


def chunk_markdown(markdown_text: str, *, max_chars: int) -> list[str]:
    """Chunk Markdown by paragraph and heading boundaries."""

    cleaned = clean_markdown(markdown_text)
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]

    blocks = split_markdown_blocks(cleaned)
    chunks: list[str] = []
    current_blocks: list[str] = []
    current_length = 0
    for block in blocks:
        block_length = len(block)
        if block_length > max_chars:
            if current_blocks:
                chunks.append("\n\n".join(current_blocks))
                current_blocks = []
                current_length = 0
            chunks.extend(split_long_text(block, max_chars=max_chars))
            continue
        next_length = current_length + block_length + (2 if current_blocks else 0)
        if current_blocks and next_length > max_chars:
            chunks.append("\n\n".join(current_blocks))
            current_blocks = [block]
            current_length = block_length
            continue
        current_blocks.append(block)
        current_length = next_length

    if current_blocks:
        chunks.append("\n\n".join(current_blocks))
    return [chunk for chunk in chunks if chunk.strip()]


def clean_markdown(markdown_text: str) -> str:
    """Normalize Markdown whitespace before chunking."""

    cleaned = markdown_text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"\n{4,}", "\n\n\n", cleaned)
    return cleaned.strip()


def split_markdown_blocks(markdown_text: str) -> list[str]:
    """Split Markdown into chunkable blocks."""

    blocks: list[str] = []
    current: list[str] = []
    for line in markdown_text.splitlines():
        if MARKDOWN_HEADING_PATTERN.match(line) and current:
            blocks.append("\n".join(current).strip())
            current = [line]
            continue
        if not line.strip() and current:
            blocks.append("\n".join(current).strip())
            current = []
            continue
        if line.strip():
            current.append(line)
    if current:
        blocks.append("\n".join(current).strip())
    return [block for block in blocks if block]


def split_long_text(text: str, *, max_chars: int) -> list[str]:
    """Split very long text blocks into bounded chunks."""

    return [text[index : index + max_chars].strip() for index in range(0, len(text), max_chars) if text[index : index + max_chars].strip()]


def iter_markdown_files(root: Path) -> Iterable[Path]:
    """Yield Markdown files under root in deterministic order."""

    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".md", ".markdown"}:
            yield path


def ensure_unique_doc_ids(documents: list[IndexDocument]) -> list[IndexDocument]:
    """Make duplicate document IDs unique without changing stable IDs when possible."""

    seen: dict[str, int] = {}
    unique_documents: list[IndexDocument] = []
    for document in documents:
        count = seen.get(document.doc_id, 0)
        seen[document.doc_id] = count + 1
        if count == 0:
            unique_documents.append(document)
            continue
        new_doc = IndexDocument(
            doc_id=f"{document.doc_id}:dup{count}",
            doc_type=document.doc_type,
            title=document.title,
            text=document.text,
            fields=document.fields,
            metadata={**document.metadata, "duplicate_of": document.doc_id, "duplicate_index": count},
            source_path=document.source_path,
        )
        unique_documents.append(new_doc)
    return unique_documents


def build_manifest(report: IndexBuildReport) -> dict[str, object]:
    """Build an index manifest for downstream retrieval commands."""

    return {
        "schema_version": 1,
        "created_at": report.started_at,
        "inputs": report.inputs,
        "output_dir": report.output_dir,
        "outputs": report.output_paths,
        "counts": {
            "work_orders": report.work_orders,
            "part_records": report.part_records,
            "manual_files": report.manual_files,
            "manual_chunks": report.manual_chunks,
            "total_documents": report.total_documents,
            "index_terms": report.index_terms,
        },
        "field_weights": FIELD_WEIGHTS,
        "tokenizer": {
            "ascii_code_tokens": True,
            "chinese_ngrams": [2, 3],
            "whole_chinese_sequence_max_chars": 12,
        },
    }


def write_documents_jsonl(documents: list[IndexDocument], path: Path) -> None:
    """Write index documents as JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for document in documents:
            handle.write(json.dumps(document.to_dict(), ensure_ascii=False) + "\n")


def write_json(payload: object, path: Path) -> None:
    """Write a JSON file with UTF-8 encoding."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_report(report: IndexBuildReport, path: Path) -> None:
    """Write an index build report as JSON."""

    write_json(report.to_dict(), path)


def format_report_summary(report: IndexBuildReport) -> str:
    """Return a compact human-readable index build summary."""

    return (
        f"work_orders={report.work_orders}, part_records={report.part_records}, "
        f"manual_files={report.manual_files}, manual_chunks={report.manual_chunks}, "
        f"total_documents={report.total_documents}, index_terms={report.index_terms}, "
        f"failed_items={len(report.failed_items)}, elapsed={report.elapsed_seconds}s"
    )


def clean_string(value: object) -> str:
    """Convert a value to a stripped string, treating null as empty."""

    if value is None:
        return ""
    return str(value).strip()


def first_non_empty(*values: str | None) -> str | None:
    """Return the first non-empty string."""

    for value in values:
        cleaned = clean_string(value)
        if cleaned:
            return cleaned
    return None


def join_text(values: Iterable[object]) -> str:
    """Join non-empty text values with newlines."""

    return "\n".join(clean_string(value) for value in values if clean_string(value))


def short_hash(value: str) -> str:
    """Return a short stable SHA-1 hash for IDs."""

    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
