from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from waji_rag.html_batch import is_lossy_encoding, read_text_with_fallback


class HtmlBatchEncodingTests(unittest.TestCase):
    def test_read_text_with_fallback_detects_bomless_utf16_html(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manual.html"
            path.write_bytes("<html><body>风扇皮带异响</body></html>".encode("utf-16-le"))

            text, encoding = read_text_with_fallback(path, ("utf-8", "gb18030"))

            self.assertEqual(encoding, "utf-16-le")
            self.assertIn("风扇皮带异响", text)

    def test_read_text_with_fallback_uses_declared_html_charset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manual.html"
            html = '<html><head><meta charset="gb2312"></head><body>行走单边慢</body></html>'
            path.write_bytes(html.encode("gb18030"))

            text, encoding = read_text_with_fallback(path, ("utf-8",))

            self.assertEqual(encoding, "gb18030")
            self.assertIn("行走单边慢", text)

    def test_read_text_with_fallback_marks_lossy_last_resort(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manual.html"
            path.write_bytes(b"<html>\xae</html>")

            text, encoding = read_text_with_fallback(path, ("utf-8",))

            self.assertEqual(encoding, "gb18030-replace")
            self.assertTrue(is_lossy_encoding(encoding))
            self.assertIn("\ufffd", text)


if __name__ == "__main__":
    unittest.main()
