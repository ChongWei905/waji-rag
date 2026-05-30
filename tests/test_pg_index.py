from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from waji_rag.config import AppConfig, EmbeddingConfig, LLMConfig
from waji_rag.llm import QueryParseResult
from waji_rag.pg_index import (
    APPLICATION_DATA_TABLES,
    PgIngestReport,
    RetrievalHit,
    build_documents_for_work_order,
    build_query_constraints,
    bulk_insert_rows,
    clear_application_data_with_cursor,
    filter_evidence_for_answer,
    fetch_part_candidates,
    prioritize_hits_by_constraints,
    query_constraints_from_llm_payload,
    resolve_query_constraints,
    store_embedding_batch,
    unique_terms,
    vector_literal,
)
from waji_rag.index_build import IndexDocument
from waji_rag.work_order import PartRecord, WorkOrderRecord


class PgIndexHelpersTests(unittest.TestCase):
    def test_unique_terms_preserves_order(self) -> None:
        terms = unique_terms(["行走", "行走", "马达", "", "马达", "motor"])

        self.assertEqual(terms, ["行走", "马达", "motor"])

    def test_vector_literal_rejects_empty_vectors(self) -> None:
        with self.assertRaises(ValueError):
            vector_literal([])

    def test_build_documents_for_work_order_keeps_part_evidence_searchable(self) -> None:
        record = WorkOrderRecord(
            work_order_id="WO-001",
            reported_issue="风扇皮带异响",
            solution="检查皮带张紧度，更换风扇皮带。",
            parts=[PartRecord(part_name="风扇皮带", part_code="PB-001", quantity="1")],
            raw_text="风扇皮带异响，更换风扇皮带 PB-001",
            source_path=str(Path("WO-001.txt")),
        )

        documents = build_documents_for_work_order(record)

        self.assertEqual([document.doc_type for document in documents], ["work_order", "part_evidence"])
        self.assertEqual(documents[1].fields["part_name"], "风扇皮带")
        self.assertEqual(documents[1].fields["part_code"], "PB-001")
        self.assertEqual(documents[1].metadata["part_name"], "风扇皮带")
        self.assertEqual(documents[1].metadata["part_code"], "PB-001")
        self.assertEqual(documents[1].metadata["quantity"], "1")

    def test_store_embedding_batch_calls_provider_once(self) -> None:
        config = AppConfig(embedding=EmbeddingConfig(enabled=True, provider="vllm", model="demo-embed"))
        report = PgIngestReport(started_at="2026-01-01 00:00:00", elapsed_seconds=0.0, database_url="postgresql://x")
        cursor = FakeCursor()
        provider = FakeEmbeddingProvider()
        documents = [
            IndexDocument(
                doc_id="doc-1",
                doc_type="work_order",
                title="风扇皮带异响",
                text="检查风扇皮带张紧度",
                fields={"body": "检查风扇皮带张紧度"},
                metadata={},
                source_path="doc-1.txt",
            ),
            IndexDocument(
                doc_id="doc-2",
                doc_type="manual_typical_fault",
                title="皮带打滑",
                text="皮带松动会导致尖叫声",
                fields={"body": "皮带松动会导致尖叫声"},
                metadata={},
                source_path="doc-2.md",
            ),
        ]

        stored_count = store_embedding_batch(cursor, [(1, documents[0]), (2, documents[1])], config, provider, report)

        self.assertEqual(stored_count, 2)
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(len(provider.calls[0]), 2)
        self.assertEqual(len(cursor.executions), 2)
        self.assertIn("embedding_seconds", report.timing_seconds)
        self.assertIn("pg_write_seconds", report.timing_seconds)

    def test_clear_application_data_truncates_task_and_index_tables(self) -> None:
        cursor = FakeCursor()

        clear_application_data_with_cursor(cursor)

        sql, _params = cursor.executions[-1]
        self.assertIn("TRUNCATE TABLE", sql)
        self.assertIn("RESTART IDENTITY CASCADE", sql)
        for table_name in APPLICATION_DATA_TABLES:
            self.assertIn(table_name, sql)

    def test_bulk_insert_rows_uses_executemany_when_copy_is_unavailable(self) -> None:
        cursor = FakeCursor()

        bulk_insert_rows(
            cursor,
            copy_sql="COPY demo_table(value) FROM STDIN",
            insert_sql="INSERT INTO demo_table(value) VALUES (%s)",
            rows=[("a",), ("b",)],
        )

        self.assertEqual(cursor.executemany_calls, [("INSERT INTO demo_table(value) VALUES (%s)", [("a",), ("b",)])])

    def test_evidence_filter_rejects_same_symptom_wrong_component(self) -> None:
        constraints = build_query_constraints("用户报修机器风扇皮带异响，请回答可能原因")
        evidence_items = [
            {
                "channel": "manual_typical_faults",
                "doc_id": "manual:fan",
                "title": "风扇皮带异响",
                "body_preview": "风扇皮带松动会出现尖叫声。",
            },
            {
                "channel": "manual_typical_faults",
                "doc_id": "manual:aircon",
                "title": "空调有异响",
                "body_preview": "空调压缩机或鼓风机异常声音。",
            },
            {
                "channel": "manual_typical_faults",
                "doc_id": "manual:belt",
                "title": "发动机皮带异响",
                "body_preview": "发动机附件皮带松动会产生尖叫声。",
            },
        ]

        result = filter_evidence_for_answer(
            query="用户报修机器风扇皮带异响，请回答可能原因",
            evidence_items=evidence_items,
            constraints=constraints,
        )

        self.assertEqual([item["doc_id"] for item in result["accepted"]], ["manual:fan"])
        self.assertEqual([item["doc_id"] for item in result["rejected"]], ["manual:aircon", "manual:belt"])
        self.assertEqual(result["rejected"][0]["evidence_gate"]["reason"], "missing_strict_component_anchor")
        self.assertEqual(result["rejected"][1]["evidence_gate"]["component_hits"], ["皮带"])

    def test_query_constraints_from_llm_payload_keeps_explicit_anchors_only(self) -> None:
        fallback = build_query_constraints("用户报修机器风扇皮带异响，请回答可能原因")

        constraints = query_constraints_from_llm_payload(
            "用户报修机器风扇皮带异响，请回答可能原因",
            ["风扇", "皮带", "异响"],
            {
                "fault_phrase": "风扇皮带异响",
                "component_text": "风扇皮带",
                "component_terms": ["风扇皮带", "发动机附件轮系", "风扇", "皮带"],
                "required_component_terms": ["风扇", "皮带"],
                "symptom_terms": ["异响", "噪声"],
            },
            fallback=fallback,
        )

        self.assertEqual(constraints.fault_phrase, "风扇皮带异响")
        self.assertEqual(constraints.component_text, "风扇皮带")
        self.assertEqual(constraints.component_terms, ["风扇皮带", "风扇", "皮带"])
        self.assertEqual(constraints.required_component_terms, ["风扇", "皮带"])
        self.assertEqual(constraints.symptom_terms, ["异响"])

    def test_resolve_query_constraints_uses_llm_when_available(self) -> None:
        config = AppConfig(
            llm=LLMConfig(
                enabled=True,
                provider="vllm",
                model="demo-chat",
                base_url="http://127.0.0.1:9999/v1",
            )
        )
        events: list[dict[str, object]] = []

        with patch("waji_rag.pg_index.parse_diagnostic_query_constraints") as parser:
            parser.return_value = QueryParseResult(
                payload={
                    "fault_phrase": "风扇皮带异响",
                    "component_text": "风扇皮带",
                    "component_terms": ["风扇皮带", "风扇", "皮带"],
                    "required_component_terms": ["风扇", "皮带"],
                    "symptom_terms": ["异响"],
                },
                debug={"usage": {"total_tokens": 12}},
            )

            constraints = resolve_query_constraints("用户报修机器风扇皮带异响", ["风扇", "皮带", "异响"], config, events)

        self.assertEqual(constraints.component_terms, ["风扇皮带", "风扇", "皮带"])
        self.assertEqual(events[0]["status"], "ok")
        self.assertEqual(events[0]["mode"], "llm")

    def test_prioritize_hits_by_constraints_prefers_component_anchor(self) -> None:
        constraints = build_query_constraints("风扇皮带异响")
        hits = [
            RetrievalHit(
                document_id=1,
                doc_id="aircon",
                doc_type="manual_typical_fault",
                title="空调有异响",
                score=10.0,
                body_preview="空调出风口有异常声音。",
                work_order_id=None,
                source_path="空调系统/故障现象：空调有异响.html",
                metadata={},
            ),
            RetrievalHit(
                document_id=2,
                doc_id="fan",
                doc_type="manual_typical_fault",
                title="风扇皮带异响",
                score=5.0,
                body_preview="风扇皮带张紧不足会产生异响。",
                work_order_id=None,
                source_path="动力系统/故障现象：风扇皮带异响.html",
                metadata={},
            ),
        ]

        prioritized = prioritize_hits_by_constraints(hits, constraints, top_k=1)

        self.assertEqual(prioritized[0].doc_id, "fan")

    def test_fetch_part_candidates_returns_all_parts_for_linked_orders(self) -> None:
        cursor = FakeFetchCursor(
            [
                ("WO-001", "风扇皮带异响", None, None, "风扇皮带", "PB-001", "1", "WO-001.txt"),
                ("WO-001", "风扇皮带异响", None, None, "张紧轮", "TN-002", "1", "WO-001.txt"),
                ("WO-001", "风扇皮带异响", None, None, "皮带轮", "PL-003", "1", "WO-001.txt"),
                ("WO-001", "风扇皮带异响", None, None, "固定螺栓", "BT-004", "4", "WO-001.txt"),
            ]
        )

        parts = fetch_part_candidates(cursor, ["WO-001"])

        self.assertEqual(len(parts), 4)
        self.assertNotIn("LIMIT", cursor.executions[0][0].upper())
        self.assertEqual(cursor.executions[0][1], (["WO-001"], ["WO-001"]))
        self.assertEqual([part["part_code"] for part in parts], ["PB-001", "TN-002", "PL-003", "BT-004"])
        self.assertTrue(all(part["doc_type"] == "part_evidence" for part in parts))
        self.assertTrue(all(part["channel"] == "part_evidence" for part in parts))


class FakeCursor:
    def __init__(self) -> None:
        self.executions: list[tuple[str, object]] = []
        self.executemany_calls: list[tuple[str, list[tuple[object, ...]]]] = []

    def execute(self, sql: str, params: object = None) -> None:
        """Record SQL execution parameters for assertions."""

        self.executions.append((sql, params))

    def executemany(self, sql: str, rows: list[tuple[object, ...]]) -> None:
        """Record bulk SQL execution parameters for assertions."""

        self.executemany_calls.append((sql, rows))


class FakeFetchCursor(FakeCursor):
    def __init__(self, rows: list[tuple[object, ...]]) -> None:
        self.rows = rows
        super().__init__()

    def fetchall(self) -> list[tuple[object, ...]]:
        """Return preloaded rows for query helper assertions."""

        return self.rows


class FakeEmbeddingProvider:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str], *, text_type: str = "document") -> list[list[float]]:
        """Return deterministic vectors while recording batch calls."""

        self.calls.append(texts)
        return [[float(index), float(index + 1)] for index, _text in enumerate(texts, start=1)]


if __name__ == "__main__":
    unittest.main()
