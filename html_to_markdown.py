#!/usr/bin/env python3
"""Convert HTML documents to Markdown with careful table handling.

The script intentionally uses only the Python standard library so it can be
copied into a project and run without installing dependencies.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass, field
from html import escape as html_escape
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, TextIO


BLOCK_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "details",
    "dialog",
    "div",
    "dl",
    "fieldset",
    "figcaption",
    "figure",
    "footer",
    "form",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "tbody",
    "td",
    "tfoot",
    "th",
    "thead",
    "tr",
    "ul",
}

SKIP_CONTENT_TAGS = {"script", "style", "template", "noscript", "svg"}

VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}


@dataclass(slots=True)
class Node:
    """A tiny HTML tree node used by the Markdown renderer."""

    tag: str | None = None
    attrs: dict[str, str] = field(default_factory=dict)
    children: list["Node"] = field(default_factory=list)
    text: str = ""

    @property
    def is_text(self) -> bool:
        """Return whether this node represents text content."""

        return self.tag is None

    def append_child(self, node: "Node") -> None:
        """Append a parsed child node."""

        self.children.append(node)


class TreeBuilder(HTMLParser):
    """Build a forgiving HTML tree using Python's built-in parser."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = Node("document")
        self.stack: list[Node] = [self.root]
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Handle opening HTML tags."""

        tag = tag.lower()
        if self._skip_depth:
            if tag in SKIP_CONTENT_TAGS:
                self._skip_depth += 1
            return

        if tag in SKIP_CONTENT_TAGS:
            self._skip_depth = 1
            return

        node = Node(tag=tag, attrs={key.lower(): value or "" for key, value in attrs})
        self.stack[-1].append_child(node)
        if tag not in VOID_TAGS:
            self.stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Handle self-closing HTML tags."""

        tag = tag.lower()
        if self._skip_depth or tag in SKIP_CONTENT_TAGS:
            return

        node = Node(tag=tag, attrs={key.lower(): value or "" for key, value in attrs})
        self.stack[-1].append_child(node)

    def handle_endtag(self, tag: str) -> None:
        """Handle closing HTML tags, tolerating malformed nesting."""

        tag = tag.lower()
        if self._skip_depth:
            if tag in SKIP_CONTENT_TAGS:
                self._skip_depth -= 1
            return

        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                return

    def handle_data(self, data: str) -> None:
        """Handle text data inside the current element."""

        if self._skip_depth or not data:
            return
        self.stack[-1].append_child(Node(text=data))


class MarkdownConverter:
    """Render a parsed HTML tree as readable GitHub-flavored Markdown."""

    def __init__(self, *, reference_links: bool = False) -> None:
        self.reference_links = reference_links
        self._link_refs: dict[str, int] = {}
        self._link_ref_order: list[tuple[str, int]] = []

    def convert(self, html: str) -> str:
        """Convert an HTML string to Markdown."""

        parser = TreeBuilder()
        parser.feed(html)
        parser.close()

        content_root = self._content_root(parser.root)
        markdown = self._render_children(content_root, mode="block")
        markdown = self._normalize_markdown(markdown)
        if self.reference_links and self._link_ref_order:
            refs = "\n".join(
                f"[{number}]: {url}" for url, number in self._link_ref_order
            )
            markdown = f"{markdown}\n\n{refs}" if markdown else refs
        return markdown.rstrip() + "\n"

    @staticmethod
    def _attr_contains(node: Node, names: set[str], needles: tuple[str, ...]) -> bool:
        """Return whether any selected attribute contains one of the needles."""

        values = " ".join(node.attrs.get(name, "") for name in names).lower()
        return any(needle in values for needle in needles)

    @staticmethod
    def _content_root(root: Node) -> Node:
        """Return the most likely document body to avoid indexing template chrome."""

        candidates = MarkdownConverter._collect_descendants(root, {"main"})
        if candidates:
            return candidates[0]

        semantic_candidates = MarkdownConverter._collect_descendants(root, {"article"})
        if semantic_candidates:
            return semantic_candidates[0]

        all_nodes = MarkdownConverter._collect_descendants(root, {"div", "section"})
        for node in all_nodes:
            if MarkdownConverter._attr_contains(
                node,
                {"id", "class", "role"},
                ("main", "content", "article", "document", "manual"),
            ):
                return node

        body = MarkdownConverter._collect_descendants(root, {"body"})
        if body:
            return body[0]
        return root

    @staticmethod
    def _alignment_marker(align: str | None) -> str:
        if align == "left":
            return ":---"
        if align == "center":
            return ":---:"
        if align == "right":
            return "---:"
        return "---"

    @staticmethod
    def _cell_text(markdown: str) -> str:
        markdown = markdown.strip()
        markdown = re.sub(r"\n\s*\n+", "<br>", markdown)
        markdown = re.sub(r"\s*\n\s*", "<br>", markdown)
        markdown = re.sub(r"[ \t]+", " ", markdown)
        markdown = markdown.replace("\\|", "&#124;")
        markdown = markdown.replace("|", "\\|")
        return markdown

    @staticmethod
    def _collect_descendants(node: Node, tags: set[str]) -> list[Node]:
        found: list[Node] = []
        for child in node.children:
            if child.tag in tags:
                found.append(child)
            found.extend(MarkdownConverter._collect_descendants(child, tags))
        return found

    @staticmethod
    def _escape_markdown_text(text: str) -> str:
        text = text.replace("\\", "\\\\")
        for char in ("`", "*", "_", "[", "]"):
            text = text.replace(char, f"\\{char}")
        return text

    @staticmethod
    def _extract_align(node: Node) -> str | None:
        align = node.attrs.get("align", "").lower().strip()
        if align in {"left", "center", "right"}:
            return align

        style = node.attrs.get("style", "").lower()
        match = re.search(r"text-align\s*:\s*(left|center|right)", style)
        if match:
            return match.group(1)
        return None

    @staticmethod
    def _find_direct_children(node: Node, tag: str) -> list[Node]:
        return [child for child in node.children if child.tag == tag]

    @staticmethod
    def _normalize_markdown(markdown: str) -> str:
        lines = [line.rstrip() for line in markdown.replace("\r\n", "\n").split("\n")]
        text = "\n".join(lines)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _parse_span(value: str | None) -> int:
        try:
            parsed = int(value or "1")
        except ValueError:
            return 1
        return max(parsed, 1)

    @staticmethod
    def _table_row_nodes(table: Node) -> list[Node]:
        rows: list[Node] = []
        for child in table.children:
            if child.tag == "tr":
                rows.append(child)
            elif child.tag in {"thead", "tbody", "tfoot"}:
                rows.extend(MarkdownConverter._find_direct_children(child, "tr"))
        if not rows:
            rows = MarkdownConverter._collect_descendants(table, {"tr"})
        return rows

    def _add_reference_link(self, href: str, text: str) -> str:
        if href not in self._link_refs:
            self._link_refs[href] = len(self._link_refs) + 1
            self._link_ref_order.append((href, self._link_refs[href]))
        return f"[{text}][{self._link_refs[href]}]"

    def _expand_table(
        self, rows: list[list[dict[str, object]]]
    ) -> tuple[list[list[str]], list[str | None]]:
        grid: list[list[str]] = []
        aligns: list[str | None] = []
        rowspans: dict[int, tuple[int, str, str | None]] = {}

        for row in rows:
            output_row: list[str] = []
            col_index = 0

            for raw_cell in row:
                while col_index in rowspans:
                    remaining, text, align = rowspans[col_index]
                    output_row.append(text)
                    aligns = self._set_alignment(aligns, col_index, align)
                    if remaining <= 1:
                        del rowspans[col_index]
                    else:
                        rowspans[col_index] = (remaining - 1, text, align)
                    col_index += 1

                text = str(raw_cell["text"])
                align = raw_cell["align"]
                colspan = int(raw_cell["colspan"])
                rowspan = int(raw_cell["rowspan"])

                for span_offset in range(colspan):
                    cell_text = text if span_offset == 0 else ""
                    output_row.append(cell_text)
                    aligns = self._set_alignment(aligns, col_index, align)
                    if rowspan > 1:
                        rowspans[col_index] = (rowspan - 1, cell_text, align)
                    col_index += 1

            while col_index in rowspans:
                remaining, text, align = rowspans[col_index]
                output_row.append(text)
                aligns = self._set_alignment(aligns, col_index, align)
                if remaining <= 1:
                    del rowspans[col_index]
                else:
                    rowspans[col_index] = (remaining - 1, text, align)
                col_index += 1

            grid.append(output_row)

        width = max((len(row) for row in grid), default=0)
        for row in grid:
            row.extend([""] * (width - len(row)))
        aligns.extend([None] * (width - len(aligns)))
        return grid, aligns

    def _format_table_row(self, row: Iterable[str]) -> str:
        return "| " + " | ".join(row) + " |"

    def _render_blockquote(self, node: Node) -> str:
        content = self._render_children(node, mode="block").strip()
        if not content:
            return ""
        return "\n".join(f"> {line}" if line else ">" for line in content.splitlines())

    def _render_children(self, node: Node, *, mode: str) -> str:
        pieces = [self._render_node(child, mode=mode) for child in node.children]
        if mode == "inline":
            return "".join(pieces)
        return "".join(pieces)

    def _render_code_block(self, node: Node) -> str:
        code_node = next((child for child in node.children if child.tag == "code"), node)
        code = self._plain_text(code_node).strip("\n")
        fence = "```"
        while fence in code:
            fence += "`"
        return f"{fence}\n{code}\n{fence}"

    def _render_inline_container(self, node: Node) -> str:
        return self._render_children(node, mode="inline")

    def _render_list(self, node: Node, *, ordered: bool, indent: int = 0) -> str:
        lines: list[str] = []
        index = 1
        for child in node.children:
            if child.tag != "li":
                continue

            prefix = f"{index}. " if ordered else "- "
            content = self._render_children(child, mode="block").strip()
            nested = []
            direct_text_lines = []
            for line in content.splitlines() or [""]:
                if line.startswith(("- ", "> ")) or re.match(r"\d+\. ", line):
                    nested.append(line)
                else:
                    direct_text_lines.append(line)

            first_line = direct_text_lines[0] if direct_text_lines else ""
            lines.append(" " * indent + prefix + first_line)
            continuation_indent = indent + len(prefix)
            for line in direct_text_lines[1:]:
                lines.append(" " * continuation_indent + line)
            for line in nested:
                lines.append(" " * continuation_indent + line)
            index += 1

        return "\n".join(lines)

    def _render_node(self, node: Node, *, mode: str) -> str:
        if node.is_text:
            return self._render_text(node.text, mode=mode)

        tag = node.tag or ""

        if mode == "inline":
            return self._render_inline_node(node)

        if tag in {"script", "style", "template", "noscript", "svg", "form", "nav", "footer"}:
            return ""
        if tag in {"html", "body", "main", "article", "section", "header", "div"}:
            return self._block(self._render_children(node, mode="block"))
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            level = int(tag[1])
            text = self._render_children(node, mode="inline").strip()
            return self._block(f"{'#' * level} {text}") if text else ""
        if tag == "p":
            return self._block(self._render_children(node, mode="inline").strip())
        if tag == "br":
            return "\n"
        if tag == "hr":
            return self._block("---")
        if tag == "blockquote":
            return self._block(self._render_blockquote(node))
        if tag == "pre":
            return self._block(self._render_code_block(node))
        if tag == "ul":
            return self._block(self._render_list(node, ordered=False))
        if tag == "ol":
            return self._block(self._render_list(node, ordered=True))
        if tag == "table":
            return self._block(self._render_table(node))
        if tag in {"thead", "tbody", "tfoot", "tr", "td", "th"}:
            return self._render_children(node, mode="block")
        if tag in {"figure", "figcaption"}:
            return self._block(self._render_children(node, mode="block"))
        if tag in BLOCK_TAGS:
            return self._block(self._render_children(node, mode="block"))

        rendered = self._render_inline_node(node)
        return rendered

    def _render_inline_node(self, node: Node) -> str:
        tag = node.tag or ""

        if tag == "br":
            return "\n"
        if tag in {"strong", "b"}:
            text = self._render_inline_container(node).strip()
            return f"**{text}**" if text else ""
        if tag in {"em", "i"}:
            text = self._render_inline_container(node).strip()
            return f"*{text}*" if text else ""
        if tag in {"s", "strike", "del"}:
            text = self._render_inline_container(node).strip()
            return f"~~{text}~~" if text else ""
        if tag == "code":
            code = self._plain_text(node).strip()
            ticks = "`"
            while ticks in code:
                ticks += "`"
            return f"{ticks}{code}{ticks}" if code else ""
        if tag == "a":
            text = self._render_inline_container(node).strip() or node.attrs.get("href", "")
            href = node.attrs.get("href", "").strip()
            if not href:
                return text
            title = node.attrs.get("title", "").strip()
            if self.reference_links:
                return self._add_reference_link(href, text)
            if title:
                safe_title = title.replace('"', '\\"')
                return f'[{text}]({href} "{safe_title}")'
            return f"[{text}]({href})"
        if tag == "img":
            alt = node.attrs.get("alt", "").strip()
            src = node.attrs.get("src", "").strip()
            title = node.attrs.get("title", "").strip()
            if not src:
                return alt
            if title:
                safe_title = title.replace('"', '\\"')
                return f'![{alt}]({src} "{safe_title}")'
            return f"![{alt}]({src})"
        if tag in {"sub", "sup"}:
            return html_escape(self._plain_text(node), quote=False)
        if tag in {"script", "style", "template", "noscript"}:
            return ""

        return self._render_inline_container(node)

    def _render_table(self, table: Node) -> str:
        rows = self._parse_table_rows(table)
        if not rows:
            return ""

        grid, aligns = self._expand_table(rows)
        if not grid:
            return ""

        header_index = self._header_index(rows)
        header = grid[header_index]
        body = grid[:header_index] + grid[header_index + 1 :]
        if not any(cell.strip() for cell in header):
            header = [f"Column {index}" for index in range(1, len(header) + 1)]

        separator = [self._alignment_marker(align) for align in aligns[: len(header)]]
        lines = [self._format_table_row(header), self._format_table_row(separator)]
        lines.extend(self._format_table_row(row) for row in body)

        caption = self._table_caption(table)
        markdown_table = "\n".join(lines)
        if caption:
            return f"*{caption}*\n\n{markdown_table}"
        return markdown_table

    def _render_text(self, text: str, *, mode: str) -> str:
        if mode == "inline":
            return self._escape_markdown_text(re.sub(r"\s+", " ", text))
        return text

    def _parse_table_rows(self, table: Node) -> list[list[dict[str, object]]]:
        rows: list[list[dict[str, object]]] = []
        for row in self._table_row_nodes(table):
            cells: list[dict[str, object]] = []
            for cell in row.children:
                if cell.tag not in {"td", "th"}:
                    continue
                cells.append(
                    {
                        "tag": cell.tag,
                        "text": self._cell_text(
                            self._render_children(cell, mode="block")
                            or self._render_children(cell, mode="inline")
                        ),
                        "align": self._extract_align(cell),
                        "colspan": self._parse_span(cell.attrs.get("colspan")),
                        "rowspan": self._parse_span(cell.attrs.get("rowspan")),
                    }
                )
            if cells:
                rows.append(cells)
        return rows

    def _plain_text(self, node: Node) -> str:
        if node.is_text:
            return node.text
        return "".join(self._plain_text(child) for child in node.children)

    def _table_caption(self, table: Node) -> str:
        caption = next((child for child in table.children if child.tag == "caption"), None)
        if not caption:
            return ""
        return self._render_children(caption, mode="inline").strip()

    def _set_alignment(
        self, aligns: list[str | None], index: int, align: object
    ) -> list[str | None]:
        aligns.extend([None] * (index + 1 - len(aligns)))
        if align and aligns[index] is None:
            aligns[index] = str(align)
        return aligns

    def _block(self, text: str) -> str:
        text = text.strip()
        return f"\n\n{text}\n\n" if text else ""

    def _header_index(self, rows: list[list[dict[str, object]]]) -> int:
        for index, row in enumerate(rows):
            if any(cell["tag"] == "th" for cell in row):
                return index
        return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    parser = argparse.ArgumentParser(
        description="Convert HTML to Markdown. Reads stdin when input is omitted."
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="HTML file to convert. If omitted, HTML is read from stdin.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Markdown output file. If omitted, Markdown is written to stdout.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Input/output file encoding. Defaults to utf-8.",
    )
    parser.add_argument(
        "--reference-links",
        action="store_true",
        help="Render links as reference-style Markdown links.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print progress logs to stderr.",
    )
    parser.add_argument(
        "--log-file",
        help="Also write progress logs to this file, useful on Windows terminals.",
    )
    return parser


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments."""

    return build_parser().parse_args(argv)


def log(message: str, *, quiet: bool = False, log_file: TextIO | None = None) -> None:
    """Print a timestamped progress message unless quiet mode is enabled."""

    if quiet and log_file is None:
        return

    timestamp = time.strftime("%H:%M:%S")
    line = f"[{timestamp}] {message}"
    if not quiet:
        print(line, file=sys.stderr, flush=True)
    if log_file is not None:
        print(line, file=log_file, flush=True)


def open_log_file(path: str | None, *, encoding: str) -> TextIO | None:
    """Open an optional log file for progress messages."""

    if not path:
        return None
    return Path(path).open("a", encoding=encoding)


def describe_output_path(path: str | None) -> str:
    """Return a human-readable destination for log messages."""

    return path or "stdout"


def detect_table_count(html: str) -> int:
    """Return a quick count of literal table start tags in the source HTML."""

    return len(re.findall(r"<\s*table\b", html, flags=re.IGNORECASE))


def validate_input_source(args: argparse.Namespace) -> bool:
    """Return whether the requested input source can provide HTML."""

    if args.input:
        return True
    if not sys.stdin.isatty():
        return True

    parser = build_parser()
    parser.print_usage(sys.stderr)
    print(
        "html_to_markdown.py: no input file was provided and stdin is empty.",
        file=sys.stderr,
    )
    print(
        "Example: python html_to_markdown.py input.html -o output.md",
        file=sys.stderr,
    )
    return False


def read_input(path: str | None, *, encoding: str) -> str:
    """Read HTML from a file path or standard input."""

    if path:
        return Path(path).read_text(encoding=encoding)
    return sys.stdin.read()


def write_output(markdown: str, path: str | None, *, encoding: str) -> None:
    """Write Markdown to a file path or standard output."""

    if path:
        Path(path).write_text(markdown, encoding=encoding)
        return
    sys.stdout.write(markdown)


def main(argv: list[str] | None = None) -> int:
    """Run the HTML to Markdown command-line interface."""

    args = parse_args(argv or sys.argv[1:])
    if not validate_input_source(args):
        return 2

    log_file: TextIO | None = None
    try:
        log_file = open_log_file(args.log_file, encoding=args.encoding)
        source = args.input or "stdin"
        destination = describe_output_path(args.output)
        log(f"Reading HTML from {source}", quiet=args.quiet, log_file=log_file)
        html = read_input(args.input, encoding=args.encoding)
        if not html and not args.input:
            log(
                "No HTML was received from stdin; pass an input file or pipe HTML in.",
                quiet=False,
                log_file=log_file,
            )
            return 2
        if not html.strip():
            log(
                "Warning: input is empty or whitespace only.",
                quiet=args.quiet,
                log_file=log_file,
            )
        log(
            f"Read {len(html):,} characters; found {detect_table_count(html)} table tag(s)",
            quiet=args.quiet,
            log_file=log_file,
        )
        log("Converting HTML to Markdown", quiet=args.quiet, log_file=log_file)
        markdown = MarkdownConverter(reference_links=args.reference_links).convert(html)
        if not markdown.strip():
            log(
                "Warning: conversion produced empty Markdown.",
                quiet=args.quiet,
                log_file=log_file,
            )
        log(
            f"Generated {len(markdown):,} Markdown characters",
            quiet=args.quiet,
            log_file=log_file,
        )
        log(f"Writing Markdown to {destination}", quiet=args.quiet, log_file=log_file)
        write_output(markdown, args.output, encoding=args.encoding)
        log("Done", quiet=args.quiet, log_file=log_file)
    except OSError as exc:
        log(f"File error: {exc}", quiet=False, log_file=log_file)
        return 1
    except HTMLParserError as exc:
        log(f"HTML parse error: {exc}", quiet=False, log_file=log_file)
        return 1
    finally:
        if log_file is not None:
            log_file.close()
    return 0


class HTMLParserError(Exception):
    """Compatibility exception for unexpected parser failures."""


if __name__ == "__main__":
    raise SystemExit(main())
