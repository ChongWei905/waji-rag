from __future__ import annotations

import unittest
from unittest.mock import patch

from waji_rag.config import AppConfig, LLMConfig
from waji_rag.llm import ModelCallResult, parse_json_object
from waji_rag.pg_index import (
    DatabaseOptions,
    RagPipeline,
    build_deterministic_facts,
    build_final_answer_context,
    parts_from_accepted_orders,
    work_order_filter_payload,
)


class HarnessJsonTests(unittest.TestCase):
    def test_parse_json_object_accepts_markdown_fence(self) -> None:
        payload = parse_json_object('```json\n{"selected": [], "rejected": []}\n```')

        self.assertEqual(payload, {"selected": [], "rejected": []})

    def test_parse_json_object_rejects_non_object(self) -> None:
        with self.assertRaises(ValueError):
            parse_json_object("[1, 2, 3]")


class AnswerHarnessTests(unittest.TestCase):
    def test_work_order_filter_marks_failed_model_call_unknown(self) -> None:
        config = AppConfig(
            llm=LLMConfig(
                enabled=True,
                provider="vllm",
                model="fake-chat",
                base_url="http://127.0.0.1:9999/v1",
            )
        )
        pipeline = RagPipeline(DatabaseOptions(database_url="postgresql://demo"), config)
        hits = [
            {
                "doc_id": "WO-001",
                "doc_type": "work_order",
                "title": "风扇皮带异响",
                "work_order_id": "WO-001",
                "body_preview": "更换风扇皮带",
            },
            {
                "doc_id": "WO-002",
                "doc_type": "work_order",
                "title": "空调异响",
                "work_order_id": "WO-002",
                "body_preview": "更换鼓风机",
            },
        ]

        def fake_judge(**kwargs: object) -> ModelCallResult:
            work_order = kwargs["work_order_hit"]
            if isinstance(work_order, dict) and work_order.get("work_order_id") == "WO-002":
                raise RuntimeError("model timeout")
            return ModelCallResult(
                text=(
                    '{"work_order_id":"WO-001","related":true,"relevance_level":"high",'
                    '"matched_reason":"同为风扇皮带异响","repair_actions":["更换风扇皮带"],'
                    '"usable_parts":[{"name":"风扇皮带","code":"PB-001","quantity":"1"}],'
                    '"source_path":"WO-001.txt"}'
                ),
                debug={"model": "fake-chat"},
            )

        with patch("waji_rag.pg_index.judge_work_order_relevance", side_effect=fake_judge):
            payload = pipeline._filter_work_orders(query="风扇皮带异响", work_order_hits=hits, part_candidates=[])

        self.assertEqual(payload["status"], "ok")
        self.assertEqual([item["work_order_id"] for item in payload["accepted"]], ["WO-001"])
        self.assertEqual([item["work_order_id"] for item in payload["unknown"]], ["WO-002"])
        self.assertEqual(payload["unknown"][0]["relevance_level"], "unknown")

    def test_final_context_keeps_fault_codes_and_selected_order_parts(self) -> None:
        accepted_order = work_order_filter_payload(
            {
                "doc_id": "WO-001",
                "doc_type": "work_order",
                "title": "风扇皮带异响",
                "work_order_id": "WO-001",
                "source_path": "WO-001.txt",
                "body_preview": "更换风扇皮带",
            },
            [{"work_order_id": "WO-001", "part_name": "风扇皮带", "part_code": "PB-001", "quantity": "1"}],
            relevance_level="high",
            matched_reason="同部件同异常",
        )
        rejected_order = work_order_filter_payload(
            {"doc_id": "WO-002", "work_order_id": "WO-002", "title": "空调异响"},
            [],
            relevance_level="unrelated",
            matched_reason="部件不同",
        )
        selected_parts = parts_from_accepted_orders([accepted_order])
        selected_evidence = {
            "fault_code_evidence": [{"doc_id": "E00131", "title": "GPS 一级锁车", "source_path": "E00131.html"}],
            "work_orders": [accepted_order],
            "manuals": [],
        }
        facts = build_deterministic_facts(selected_evidence=selected_evidence, selected_parts=selected_parts)
        context = build_final_answer_context(
            query="风扇皮带异响",
            fault_code_hits=selected_evidence["fault_code_evidence"],
            accepted_orders=[accepted_order],
            selected_manuals=[],
            facts=facts,
            selected_parts=selected_parts,
        )

        self.assertEqual(context["fault_code_evidence"][0]["doc_id"], "E00131")
        self.assertEqual(context["facts"]["coded_parts"][0]["code"], "PB-001")
        self.assertNotIn(rejected_order["work_order_id"], str(context))


if __name__ == "__main__":
    unittest.main()
