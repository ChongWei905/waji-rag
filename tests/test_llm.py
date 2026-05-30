from __future__ import annotations

import unittest

from waji_rag.llm import parse_query_constraints_json


class LlmHelpersTests(unittest.TestCase):
    def test_parse_query_constraints_json_accepts_fenced_json(self) -> None:
        payload = parse_query_constraints_json(
            """```json
            {
              "fault_phrase": "风扇皮带异响",
              "component_text": "风扇皮带",
              "component_terms": ["风扇皮带", "风扇", "皮带"],
              "required_component_terms": ["风扇", "皮带"],
              "symptom_terms": ["异响"]
            }
            ```"""
        )

        self.assertEqual(payload["fault_phrase"], "风扇皮带异响")
        self.assertEqual(payload["component_terms"], ["风扇皮带", "风扇", "皮带"])


if __name__ == "__main__":
    unittest.main()
