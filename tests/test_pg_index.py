from __future__ import annotations

import unittest
from pathlib import Path

from waji_rag.pg_index import (
    build_documents_for_work_order,
    unique_terms,
    vector_literal,
)
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


if __name__ == "__main__":
    unittest.main()
