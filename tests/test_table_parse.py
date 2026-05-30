from __future__ import annotations

import unittest
import zipfile
from io import BytesIO

from waji_rag.web import parse_table_file, parse_xlsx_table


class TableParseTests(unittest.TestCase):
    def test_parse_xlsx_table_reads_first_sheet_as_batch_eval_rows(self) -> None:
        payload = parse_xlsx_table(
            make_xlsx(
                [
                    ["问题", "新件备件名称", "新件物料编码", "新件数量"],
                    ["风扇皮带异响", "风扇皮带,张紧轮", "3101,3102", "1,2"],
                    ["行走单边慢", "", "", ""],
                ]
            )
        )

        self.assertEqual(payload["format"], "XLSX")
        self.assertEqual(payload["headers"], ["问题", "新件备件名称", "新件物料编码", "新件数量"])
        self.assertEqual(payload["rows"][0], ["风扇皮带异响", "风扇皮带,张紧轮", "3101,3102", "1,2"])
        self.assertEqual(payload["rows"][1], ["行走单边慢", "", "", ""])

    def test_parse_table_file_rejects_legacy_xls(self) -> None:
        with self.assertRaisesRegex(ValueError, "暂不支持旧版 .xls"):
            parse_table_file("cases.xls", b"not an xlsx")


def make_xlsx(rows: list[list[str]]) -> bytes:
    strings: list[str] = []
    string_index: dict[str, int] = {}
    for row in rows:
        for value in row:
            if value not in string_index:
                string_index[value] = len(strings)
                strings.append(value)

    sheet_rows = []
    for row_index, row in enumerate(rows, start=1):
        cells = []
        for column_index, value in enumerate(row):
            cell_ref = f"{column_name(column_index)}{row_index}"
            cells.append(f'<c r="{cell_ref}" t="s"><v>{string_index[value]}</v></c>')
        sheet_rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')

    output = BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as workbook:
        workbook.writestr(
            "[Content_Types].xml",
            """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>
</Types>""",
        )
        workbook.writestr(
            "_rels/.rels",
            """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>""",
        )
        workbook.writestr(
            "xl/workbook.xml",
            """<?xml version="1.0" encoding="UTF-8"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
          xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets>
</workbook>""",
        )
        workbook.writestr(
            "xl/_rels/workbook.xml.rels",
            """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>""",
        )
        workbook.writestr(
            "xl/sharedStrings.xml",
            f"""<?xml version="1.0" encoding="UTF-8"?>
<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="{len(strings)}" uniqueCount="{len(strings)}">
  {"".join(f"<si><t>{escape_xml(value)}</t></si>" for value in strings)}
</sst>""",
        )
        workbook.writestr(
            "xl/worksheets/sheet1.xml",
            f"""<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>{"".join(sheet_rows)}</sheetData>
</worksheet>""",
        )
    return output.getvalue()


def column_name(index: int) -> str:
    name = ""
    current = index + 1
    while current:
        current, remainder = divmod(current - 1, 26)
        name = chr(ord("A") + remainder) + name
    return name


def escape_xml(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


if __name__ == "__main__":
    unittest.main()
