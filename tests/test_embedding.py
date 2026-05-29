from __future__ import annotations

import unittest
from unittest.mock import patch

from waji_rag.config import EmbeddingConfig
from waji_rag.embedding import OpenAICompatibleEmbeddingProvider, is_local_url, should_bypass_proxy


class EmbeddingProviderTests(unittest.TestCase):
    def test_vllm_provider_allows_empty_key_and_model(self) -> None:
        config = EmbeddingConfig(
            enabled=True,
            provider="vllm",
            model="",
            base_url="http://127.0.0.1:8888/v1",
            api_key="",
            dimensions=None,
        )

        self.assertTrue(config.is_available())

    def test_vllm_payload_uses_minimal_openai_compatible_shape(self) -> None:
        config = EmbeddingConfig(
            enabled=True,
            provider="vllm",
            model="",
            base_url="http://127.0.0.1:8888/v1",
            api_key="",
            dimensions=None,
        )
        provider = OpenAICompatibleEmbeddingProvider(config)

        with patch("waji_rag.embedding.post_json", return_value={"data": [{"index": 0, "embedding": [0.1, 0.2]}]}) as mocked:
            vectors = provider.embed_texts(["你好"], text_type="query")

        self.assertEqual(vectors, [[0.1, 0.2]])
        args, kwargs = mocked.call_args
        self.assertEqual(args[0], "http://127.0.0.1:8888/v1/embeddings")
        self.assertEqual(kwargs["payload"], {"input": ["你好"]})
        self.assertEqual(kwargs["api_key"], "")

    def test_local_embedding_urls_bypass_proxy(self) -> None:
        self.assertTrue(is_local_url("http://localhost:8888/v1/embeddings"))
        self.assertTrue(is_local_url("http://127.0.0.1:8888/v1/embeddings"))
        self.assertFalse(is_local_url("https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"))

    def test_configured_no_proxy_hosts_support_ip_cidr_and_wildcard(self) -> None:
        patterns = ["10.30.4.5", "192.168.0.0/16", "*.company.local"]

        self.assertTrue(should_bypass_proxy("http://10.30.4.5:8888/v1/embeddings", patterns))
        self.assertTrue(should_bypass_proxy("http://192.168.8.9:8888/v1/embeddings", patterns))
        self.assertTrue(should_bypass_proxy("http://embed.company.local:8888/v1/embeddings", patterns))
        self.assertFalse(should_bypass_proxy("http://172.16.8.9:8888/v1/embeddings", patterns))


if __name__ == "__main__":
    unittest.main()
