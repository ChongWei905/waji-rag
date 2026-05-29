from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from waji_rag.index_build import IndexBuildOptions, LocalIndexBuilder, tokenize_text


class IndexBuildTests(unittest.TestCase):
    def test_tokenize_chinese_and_code_terms(self) -> None:
        terms = tokenize_text("行走单边慢，E00131 GPS一级锁车，物料编码 310705565")

        self.assertIn("行走", terms)
        self.assertIn("单边", terms)
        self.assertIn("行走单", terms)
        self.assertIn("e00131", terms)
        self.assertIn("gps", terms)
        self.assertIn("310705565", terms)

    def test_build_index_outputs_documents_and_terms(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            work_orders_jsonl = root / "work_orders.jsonl"
            parts_jsonl = root / "parts_evidence.jsonl"
            manual_md_dir = root / "manual_md"
            manual_file = manual_md_dir / "类型A" / "典型故障解析" / "液压系统故障解析" / "故障现象：行走单边慢.md"
            output_dir = root / "index"

            write_jsonl(
                work_orders_jsonl,
                [
                    {
                        "work_order_id": "WO-001",
                        "reported_issue": "行走单边慢",
                        "solution": "判断右侧行走马达内泄，更换行走马达总成。",
                        "remarks": "客户报错机号。",
                        "parts": [],
                        "raw_text": "行走单边慢，更换行走马达总成。",
                        "source_path": "WO-001.txt",
                    }
                ],
            )
            write_jsonl(
                parts_jsonl,
                [
                    {
                        "work_order_id": "WO-001",
                        "reported_issue": "行走单边慢",
                        "solution": "更换行走马达总成。",
                        "remarks": "客户报错机号。",
                        "part_name": "行走马达总成",
                        "part_code": "MOTOR-001",
                        "quantity": "1",
                        "raw_text": "行走马达总成 MOTOR-001 1",
                        "source_path": "WO-001.txt",
                    }
                ],
            )
            manual_file.parent.mkdir(parents=True, exist_ok=True)
            manual_file.write_text(
                "# 行走单边慢\n\n## 故障原因\n\n行走马达内泄可能导致单边行走慢。\n\n## 处理\n\n检查主泵压力和行走马达。",
                encoding="utf-8",
            )

            report = LocalIndexBuilder(
                IndexBuildOptions(
                    work_orders_jsonl=work_orders_jsonl,
                    parts_jsonl=parts_jsonl,
                    manual_md_dir=manual_md_dir,
                    output_dir=output_dir,
                )
            ).build()

            self.assertEqual(report.work_orders, 1)
            self.assertEqual(report.part_records, 1)
            self.assertEqual(report.manual_files, 1)
            self.assertEqual(report.total_documents, 3)
            self.assertTrue((output_dir / "index_manifest.json").exists())
            self.assertTrue((output_dir / "documents.jsonl").exists())
            self.assertTrue((output_dir / "inverted_index.json").exists())
            index_payload = json.loads((output_dir / "inverted_index.json").read_text(encoding="utf-8"))
            self.assertIn("行走", index_payload)
            self.assertIn("马达", index_payload)
            self.assertIn("motor-001", index_payload)


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    unittest.main()
