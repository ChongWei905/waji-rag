from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from waji_rag.model_call_logging import append_model_api_request_log, redact_headers, redact_url


class ModelCallLoggingTests(unittest.TestCase):
    def test_redact_headers_hides_authorization(self) -> None:
        headers = redact_headers({"Authorization": "Bearer secret", "Content-Type": "application/json"})

        self.assertEqual(headers["Authorization"], "<redacted>")
        self.assertEqual(headers["Content-Type"], "application/json")

    def test_redact_url_hides_secret_query_values(self) -> None:
        url = redact_url("https://example.test/v1/chat?api_key=secret&mode=test")

        self.assertEqual(url, "https://example.test/v1/chat?api_key=%3Credacted%3E&mode=test")

    def test_append_model_api_request_log_writes_jsonl_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = str(Path(tmp_dir) / "model_api_requests.jsonl")

            append_model_api_request_log(
                enabled=True,
                log_path=log_path,
                service="embedding",
                provider="vllm",
                model="",
                method="POST",
                url="http://127.0.0.1:8888/v1/embeddings",
                headers={"Authorization": "Bearer secret", "Content-Type": "application/json"},
                payload={"input": ["你好"]},
                elapsed_ms=12,
                status="ok",
                status_code=200,
                response={"data": [{"index": 0, "embedding": [0.1, 0.2]}]},
            )

            rows = Path(log_path).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 1)
            record = json.loads(rows[0])
            self.assertEqual(record["request"]["payload"], {"input": ["你好"]})
            self.assertEqual(record["request"]["headers"]["Authorization"], "<redacted>")
            self.assertEqual(record["result"]["response_summary"]["first_embedding_dimensions"], 2)


if __name__ == "__main__":
    unittest.main()
