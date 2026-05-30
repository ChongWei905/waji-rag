from __future__ import annotations

import unittest

from waji_rag.web import build_failed_items_from_task, retry_ingest_payload


class WebRetryTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
