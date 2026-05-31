from __future__ import annotations

import unittest

from waji_rag.config import config_from_payload


class ConfigTests(unittest.TestCase):
    def test_redaction_keeps_non_secret_token_counts_visible(self) -> None:
        config = config_from_payload(
            {
                "llm": {
                    "enabled": True,
                    "api_key": "secret-value",
                    "api_key_env": "DOCARBOR_LLM_API_KEY",
                    "max_tokens": 777,
                }
            }
        )

        payload = config.to_dict()

        self.assertEqual(payload["llm"]["api_key"], "<redacted>")
        self.assertEqual(payload["llm"]["api_key_env"], "DOCARBOR_LLM_API_KEY")
        self.assertEqual(payload["llm"]["max_tokens"], 777)

    def test_embedding_no_proxy_hosts_parse_from_comma_string(self) -> None:
        config = config_from_payload(
            {
                "embedding": {
                    "enabled": True,
                    "provider": "vllm",
                    "base_url": "http://10.30.4.5:8888/v1",
                    "no_proxy_hosts": "10.30.4.5,192.168.0.0/16,*.company.local",
                }
            }
        )

        self.assertEqual(config.embedding.no_proxy_hosts, ["10.30.4.5", "192.168.0.0/16", "*.company.local"])

    def test_llm_and_rerank_no_proxy_hosts_parse_from_comma_string(self) -> None:
        config = config_from_payload(
            {
                "llm": {
                    "enabled": True,
                    "provider": "vllm",
                    "model": "qwen-local",
                    "base_url": "http://10.30.4.5:8000/v1",
                    "no_proxy_hosts": "10.30.4.5,192.168.0.0/16",
                },
                "rerank": {
                    "enabled": True,
                    "api_key": "secret-value",
                    "no_proxy_hosts": "10.30.4.5,*.company.local",
                },
            }
        )

        self.assertTrue(config.llm.is_available())
        self.assertEqual(config.llm.no_proxy_hosts, ["10.30.4.5", "192.168.0.0/16"])
        self.assertEqual(config.rerank.no_proxy_hosts, ["10.30.4.5", "*.company.local"])

    def test_unknown_config_sections_are_ignored(self) -> None:
        config = config_from_payload(
            {
                "legacy_feature": {
                    "enabled": True,
                    "provider": "vllm",
                    "model": "parser-model",
                    "base_url": "http://127.0.0.1:8010/v1",
                },
                "llm": {
                    "enabled": False,
                    "provider": "dashscope",
                    "model": "answer-model",
                },
            }
        )

        self.assertFalse(config.llm.is_available())
        self.assertNotIn("legacy_feature", config.to_dict())
        self.assertEqual(config.llm.model, "answer-model")

    def test_model_request_logging_defaults_to_enabled_and_can_be_disabled(self) -> None:
        default_config = config_from_payload({})

        disabled_config = config_from_payload(
            {
                "embedding": {"log_requests_enabled": False},
                "llm": {"log_requests_enabled": False, "request_log_path": "tmp/model-calls.jsonl"},
            }
        )

        self.assertTrue(default_config.embedding.log_requests_enabled)
        self.assertTrue(default_config.llm.log_requests_enabled)
        self.assertFalse(disabled_config.embedding.log_requests_enabled)
        self.assertFalse(disabled_config.llm.log_requests_enabled)
        self.assertEqual(disabled_config.llm.request_log_path, "tmp/model-calls.jsonl")

    def test_work_order_threshold_config_is_clamped(self) -> None:
        config = config_from_payload(
            {
                "retrieval": {
                    "work_order_candidate_top_k": 80,
                    "work_order_min_relative_score": 1.8,
                    "work_order_max_hits": 12,
                }
            }
        )

        self.assertEqual(config.retrieval.work_order_candidate_top_k, 80)
        self.assertEqual(config.retrieval.work_order_min_relative_score, 1.0)
        self.assertEqual(config.retrieval.work_order_max_hits, 12)


if __name__ == "__main__":
    unittest.main()
