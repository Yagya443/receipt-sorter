import contextlib
import csv
import io
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import sorter


class ParseReceiptResponseTests(unittest.TestCase):
    def test_strips_thinking_block_and_parses_numeric_total(self):
        result = sorter.parse_receipt_response(
            '<think>Need inspect the total.</think>'
            '{"vendor":"Staples","date":"2026-10-05","total":21.72,"category":"office"}'
        )

        self.assertEqual(result["vendor"], "Staples")
        self.assertEqual(result["date"], "2026-10-05")
        self.assertEqual(result["total"], Decimal("21.72"))
        self.assertEqual(result["category"], "office")

    def test_rejects_non_numeric_total(self):
        with self.assertRaises(sorter.ReceiptParseError):
            sorter.parse_receipt_response(
                '{"vendor":"Shop","date":"","total":"12.50","category":"food"}'
            )

    def test_rejects_category_outside_allowed_list(self):
        with self.assertRaises(sorter.ReceiptParseError):
            sorter.parse_receipt_response(
                '{"vendor":"Shop","date":"","total":12.5,"category":"retail"}'
            )


class ExtractReceiptTests(unittest.TestCase):
    @staticmethod
    def response(content):
        return SimpleNamespace(message=SimpleNamespace(content=content))

    @patch("sorter.ollama.chat")
    def test_retries_invalid_json_once(self, chat):
        chat.side_effect = [
            self.response("not json"),
            self.response(
                '{"vendor":"Staples","date":"2026-10-05","total":21.72,"category":"office"}'
            ),
        ]

        receipt = sorter.extract_receipt("Staples total 21.72")

        self.assertEqual(receipt["vendor"], "Staples")
        self.assertEqual(chat.call_count, 2)
        for call in chat.call_args_list:
            self.assertEqual(call.kwargs["model"], sorter.MODEL)
            self.assertEqual(call.kwargs["format"], "json")
            self.assertIs(call.kwargs["think"], False)

    @patch("sorter.ollama.chat")
    def test_fails_after_second_invalid_response(self, chat):
        chat.return_value = self.response("not json")

        with self.assertRaisesRegex(
            sorter.ReceiptParseError, "after one retry"
        ):
            sorter.extract_receipt("unreadable")

        self.assertEqual(chat.call_count, 2)


class OutputTests(unittest.TestCase):
    def test_writes_plain_numeric_totals_and_category_summaries(self):
        receipts = [
            {
                "vendor": "Staples",
                "date": "2026-10-05",
                "total": Decimal("21.720"),
                "category": "office",
            }
        ]

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "output.csv"
            sorter.write_csv(receipts, output_path)
            with output_path.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(rows[0]["total"], "21.72")
        self.assertEqual(
            list(rows[0]),
            ["vendor", "date", "total", "category"],
        )
        with patch("builtins.print") as print_mock:
            sorter.print_category_totals(receipts)
        self.assertIn(unittest.mock.call("office: 21.72"), print_mock.call_args_list)

    @patch("sorter.ollama.chat")
    def test_main_processes_every_text_receipt(self, chat):
        chat.side_effect = [
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"vendor":"Shop A","date":"","total":10,"category":"food"}'
                )
            ),
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"vendor":"Shop B","date":"","total":5.5,"category":"travel"}'
                )
            ),
        ]

        with tempfile.TemporaryDirectory() as directory:
            receipts_dir = Path(directory) / "receipts"
            receipts_dir.mkdir()
            (receipts_dir / "first.txt").write_text("Receipt A", encoding="utf-8")
            (receipts_dir / "second.txt").write_text("Receipt B", encoding="utf-8")
            output_path = Path(directory) / "output.csv"

            with (
                patch.object(sorter, "RECEIPTS_DIR", receipts_dir),
                patch.object(sorter, "OUTPUT_CSV", output_path),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                sorter.main()

            with output_path.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(chat.call_count, 2)
        self.assertEqual([row["vendor"] for row in rows], ["Shop A", "Shop B"])


if __name__ == "__main__":
    unittest.main()
