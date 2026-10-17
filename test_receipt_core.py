import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import receipt_core as core


class ReceiptCleaningTests(unittest.TestCase):
    def test_clean_normalizes_fields_and_flags_unverified_total(self):
        result = core._clean(
            {
                "vendor": "  Corner Shop ",
                "date": "2026-10-05",
                "currency": "usd",
                "tax": 1.25,
                "total": 12.5,
                "payment_method": " card ",
                "category": "FOOD",
                "items": [{"description": "Snack", "amount": 12.5}],
            },
            "Corner Shop\nTotal: 12.50",
            core.CATEGORIES,
        )

        self.assertEqual(result["vendor"], "Corner Shop")
        self.assertEqual(result["date"], "2026-10-05")
        self.assertEqual(result["currency"], "USD")
        self.assertEqual(result["category"], "food")
        self.assertEqual(result["items"], [{"description": "Snack", "amount": 12.5}])
        self.assertEqual(result["needs_review"], 0)

    def test_clean_flags_total_missing_from_receipt_text(self):
        result = core._clean(
            {"total": 29.99, "date": "2026-10-05", "items": []},
            "Total: 19.99",
            core.CATEGORIES,
        )

        self.assertEqual(result["needs_review"], 1)


class ReceiptDatabaseTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "receipts.db"
        self.db_path_patch = patch.object(core, "DB_PATH", self.db_path)
        self.db_path_patch.start()
        core.init_db()

    def tearDown(self):
        self.db_path_patch.stop()
        self.temp_dir.cleanup()

    def test_save_load_and_detect_duplicate(self):
        receipt = {
            "vendor": "Corner Shop",
            "date": "2026-10-05",
            "currency": "USD",
            "tax": 1.25,
            "total": 12.5,
            "payment_method": "Card",
            "category": "food",
            "items": [{"description": "Snack", "amount": 12.5}],
            "needs_review": 0,
        }

        core.save_receipt("abc123", "receipt.txt", receipt)

        self.assertTrue(core.is_duplicate("abc123"))
        loaded = core.load_receipts()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded.iloc[0]["vendor"], "Corner Shop")
        self.assertEqual(json.loads(loaded.iloc[0]["items"]), receipt["items"])
        self.assertEqual(core.currency_totals(loaded).to_dict(), {"USD": 12.5})

    def test_currency_totals_never_combines_different_currencies(self):
        receipts = pd.DataFrame(
            [
                {"currency": "usd", "total": 12.5},
                {"currency": " USD ", "total": 2.5},
                {"currency": "INR", "total": 100.0},
                {"currency": "", "total": 4.0},
                {"currency": None, "total": 1.0},
            ]
        )

        self.assertEqual(
            core.currency_totals(receipts).to_dict(),
            {"INR": 100.0, "USD": 15.0, "Unknown": 5.0},
        )
        self.assertTrue(core.currency_totals(receipts.iloc[0:0]).empty)

    def test_manual_updates_are_saved(self):
        receipt = {
            "vendor": "Corner Shop",
            "date": "2026-10-05",
            "currency": "USD",
            "tax": 1.25,
            "total": 12.5,
            "payment_method": "Card",
            "category": "food",
            "items": [],
            "needs_review": 1,
        }
        core.save_receipt("abc123", "receipt.txt", receipt)
        edited = core.load_receipts()
        edited["vendor"] = "Corrected Vendor"
        edited["needs_review"] = 0

        core.update_receipts(edited)

        loaded = core.load_receipts()
        self.assertEqual(loaded.iloc[0]["vendor"], "Corrected Vendor")
        self.assertEqual(loaded.iloc[0]["needs_review"], 0)


if __name__ == "__main__":
    unittest.main()
