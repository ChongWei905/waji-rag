from __future__ import annotations

import unittest

from waji_rag.web import (
    batch_eval_summary,
    build_failed_items_from_task,
    build_question_tabs_from_tasks,
    is_retryable_task_db_error,
    retry_ingest_payload,
)


class WebRetryTests(unittest.TestCase):
    def test_build_question_tabs_groups_persisted_search_and_answer_tasks(self) -> None:
        tasks = [
            {
                "id": 4,
                "task_type": "answer",
                "status": "completed",
                "query": "风扇皮带异响",
                "summary": "ok",
                "created_at": "2026-05-30T10:04:00",
                "updated_at": "2026-05-30T10:05:00",
            },
            {
                "id": 3,
                "task_type": "search",
                "status": "completed",
                "query": "风扇皮带异响",
                "summary": "检索完成",
                "created_at": "2026-05-30T10:03:00",
                "updated_at": "2026-05-30T10:03:00",
            },
            {
                "id": 2,
                "task_type": "search",
                "status": "completed",
                "query": " 行走   单边慢 ",
                "summary": "检索完成",
                "created_at": "2026-05-30T10:01:00",
                "updated_at": "2026-05-30T10:02:00",
            },
            {"id": 1, "task_type": "build", "status": "completed", "query": None},
        ]

        tabs = build_question_tabs_from_tasks(tasks)

        self.assertEqual(len(tabs), 2)
        self.assertEqual(tabs[0]["query"], "风扇皮带异响")
        self.assertEqual(tabs[0]["answer_task_id"], 4)
        self.assertEqual(tabs[0]["search_task_id"], 3)
        self.assertEqual(tabs[0]["status"], "answered")
        self.assertEqual(tabs[1]["query"], "行走 单边慢")
        self.assertEqual(tabs[1]["search_task_id"], 2)
        self.assertEqual(tabs[1]["status"], "searched")

    def test_build_failed_items_filters_retryable_build_sources(self) -> None:
        task = {
            "result": {
                "report": {
                    "failed_items": [
                        {"stage": "manual", "input": "D:/manual/a.html", "error": "bad encoding"},
                        {"stage": "work_order", "input": "D:/orders/wo.txt", "error": "bad field"},
                        {"stage": "embedding", "input": "doc-1", "error": "timeout"},
                    ]
                }
            }
        }

        failed_items = build_failed_items_from_task(task)

        self.assertEqual(
            failed_items,
            [
                {"stage": "manual", "input": "D:/manual/a.html", "error": "bad encoding"},
                {"stage": "work_order", "input": "D:/orders/wo.txt", "error": "bad field"},
            ],
        )

    def test_retry_ingest_payload_targets_only_failed_stage_paths(self) -> None:
        source_request = {
            "work_order_dir": "D:/orders",
            "manual_dir": "D:/manuals",
            "reset": True,
        }
        request_payload = {
            "task_id": 12,
            "database_url": "postgresql://waji:waji@127.0.0.1:55432/waji_rag",
            "manual_dir": "D:/manuals-current",
        }
        failed_items = [{"stage": "manual", "input": "D:/manuals-current/a.html", "error": "bad encoding"}]

        retry_payload = retry_ingest_payload(request_payload, failed_items, source_request)

        self.assertNotIn("task_id", retry_payload)
        self.assertFalse(retry_payload["reset"])
        self.assertTrue(retry_payload["resume"])
        self.assertIsNone(retry_payload["work_order_dir"])
        self.assertEqual(retry_payload["manual_dir"], "D:/manuals-current")
        self.assertEqual(retry_payload["work_order_paths"], [])
        self.assertEqual(retry_payload["manual_paths"], ["D:/manuals-current/a.html"])

    def test_batch_eval_summary_uses_counts(self) -> None:
        summary = batch_eval_summary(
            {
                "status": "completed",
                "row_count": 3,
                "counts": {"done": 3, "total": 3, "pass": 2, "fail": 1, "error": 0},
            }
        )

        self.assertEqual(summary, "completed · 3/3 · 正确 2 · 失败 1 · 错误 0")

    def test_deadlock_errors_are_retryable(self) -> None:
        class FakeDeadlock(Exception):
            sqlstate = "40P01"

        self.assertTrue(is_retryable_task_db_error(FakeDeadlock("deadlock detected")))


if __name__ == "__main__":
    unittest.main()
