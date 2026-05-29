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


if __name__ == "__main__":
    unittest.main()
