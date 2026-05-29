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


if __name__ == "__main__":
    unittest.main()
