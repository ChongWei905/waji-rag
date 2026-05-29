from __future__ import annotations

import unittest
from pathlib import Path

from waji_rag.work_order import WorkOrderParser, parse_parts


class WorkOrderParserTests(unittest.TestCase):
    def test_old_new_numbered_part_blocks_prefer_new_fields(self) -> None:
        text = """
工单ID: 5DEDB7FE-F071-468B-9725-3E0359EFB3C7
用户报修内容: 1.动臂开裂、2.斗杆开裂——包外（工号：1527）
人员落实及解决方法: 现场更换动臂斗杆补液压油36升。

备件信息:
1. 旧件备件名称: XE215G.02.3 斗杆(2900)
    新件备件名称: XE215G.02.3 斗杆(2900)
    新件数量: 1.0
    旧件物料编码: 310705565
    新件物料编码: 310705565
2. 旧件备件名称: XE215G.02.4II 动臂(5680)
    新件备件名称: XE215G.02.4II 动臂(5680)
    新件数量: 1.0
    旧件物料编码: 310717107
    新件物料编码: 310717107
"""

        record = WorkOrderParser().parse(text, source_path=Path("sample.txt"))

        self.assertEqual(record.work_order_id, "5DEDB7FE-F071-468B-9725-3E0359EFB3C7")
        self.assertEqual(len(record.parts), 2)
        self.assertEqual(record.parts[0].part_name, "XE215G.02.3 斗杆(2900)")
        self.assertEqual(record.parts[0].part_code, "310705565")
        self.assertEqual(record.parts[0].quantity, "1.0")
        self.assertEqual(record.parts[1].part_name, "XE215G.02.4II 动臂(5680)")
        self.assertEqual(record.parts[1].part_code, "310717107")
        self.assertEqual(record.parts[1].quantity, "1.0")
        self.assertNotIn("parts_section_without_structured_parts", record.parse_warnings)

    def test_empty_old_new_part_template_is_ignored(self) -> None:
        parts = parse_parts(
            """
1. 旧件备件名称:
    新件备件名称:
    新件数量:
    新件物料编码:
"""
        )

        self.assertEqual(parts, [])

    def test_remark_field_does_not_override_work_order_id(self) -> None:
        text = """
工单ID: 01A33183-E69E-4B9C-ACEF-5F57FDB926BF
用户报修内容: 行走单边慢
人员落实及解决方法: 用户反馈该机单边行走慢，服务人员现场判断马达内泄过大导致，试机正常，故障排除。
备注：客户报错机号，已做变更，数据变更申请单号：SJBG2023031100001。[2023-03-11 09:28:20]
"""

        record = WorkOrderParser().parse(text, source_path=Path("sample.txt"))

        self.assertEqual(record.work_order_id, "01A33183-E69E-4B9C-ACEF-5F57FDB926BF")
        self.assertEqual(record.reported_issue, "行走单边慢")
        self.assertEqual(
            record.solution,
            "用户反馈该机单边行走慢，服务人员现场判断马达内泄过大导致，试机正常，故障排除。",
        )
        self.assertEqual(
            record.remarks,
            "客户报错机号，已做变更，数据变更申请单号:SJBG2023031100001。[2023-03-11 09:28:20]",
        )

    def test_inline_multiple_parts_still_group_fields(self) -> None:
        parts = parse_parts(
            "备件名称: 风扇皮带; 备件编码: BELT-009; 数量: 1; "
            "备件名称: 张紧轮; 备件编码: TENS-002; 数量: 1"
        )

        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0].part_name, "风扇皮带")
        self.assertEqual(parts[0].part_code, "BELT-009")
        self.assertEqual(parts[0].quantity, "1")
        self.assertEqual(parts[1].part_name, "张紧轮")
        self.assertEqual(parts[1].part_code, "TENS-002")
        self.assertEqual(parts[1].quantity, "1")


if __name__ == "__main__":
    unittest.main()
