"""Sort text receipts with a local Ollama model and summarize spending."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import ollama

# Change this value to use a different model available in your Ollama installation.
MODEL = "qwen3:4b"
RECEIPTS_DIR = Path(__file__).resolve().parent / "receipts"
OUTPUT_CSV = Path(__file__).resolve().parent / "output.csv"
CATEGORIES = ("food", "travel", "software", "office", "other")
CSV_COLUMNS = ("vendor", "date", "total", "category")

SYSTEM_PROMPT = f"""Extract the vendor, receipt date, total, and category.
Return one JSON object with exactly these fields:
- vendor: string
- date: string, preferably YYYY-MM-DD; use an empty string if unavailable
- total: a JSON number only, with no currency symbol
- category: one of {", ".join(CATEGORIES)}
Use the final amount paid as the total. Do not include explanations."""


class ReceiptParseError(ValueError):
    """Raised when a model response is not valid receipt JSON."""


def _strip_thinking_text(content: str) -> str:
    """Remove any Qwen thinking blocks before parsing the response."""
    return re.sub(r"<think>.*?</think>", "", content, flags=re.IGNORECASE | re.DOTALL).strip()


def _reject_nonfinite(value: str) -> None:
    """Reject JavaScript-style NaN and Infinity, which Python's JSON parser accepts."""
    raise ReceiptParseError(f"Non-finite JSON number: {value}")


def parse_receipt_response(content: str) -> dict[str, Any]:
    """Parse and validate the required fields in a model response."""
    clean_content = _strip_thinking_text(content)
    try:
        receipt = json.loads(
            clean_content,
            parse_float=Decimal,
            parse_constant=_reject_nonfinite,
        )
    except (json.JSONDecodeError, InvalidOperation) as exc:
        raise ReceiptParseError("The model response is not valid JSON.") from exc

    if not isinstance(receipt, dict):
        raise ReceiptParseError("The model response must be a JSON object.")

    vendor = receipt.get("vendor")
    date = receipt.get("date")
    total = receipt.get("total")
    category = receipt.get("category")

    if not isinstance(vendor, str) or not vendor.strip():
        raise ReceiptParseError("The receipt vendor must be a non-empty string.")
    if not isinstance(date, str):
        raise ReceiptParseError("The receipt date must be a string.")
    if isinstance(total, bool) or not isinstance(total, (int, Decimal)):
        raise ReceiptParseError("The receipt total must be a JSON number.")
    try:
        numeric_total = Decimal(total) if isinstance(total, int) else total
    except (InvalidOperation, ValueError) as exc:
        raise ReceiptParseError("The receipt total must be a finite number.") from exc
    if not numeric_total.is_finite():
        raise ReceiptParseError("The receipt total must be a finite number.")
    if not isinstance(category, str) or category.strip().lower() not in CATEGORIES:
        raise ReceiptParseError(
            f"The receipt category must be one of: {', '.join(CATEGORIES)}."
        )

    return {
        "vendor": vendor.strip(),
        "date": date.strip(),
        "total": numeric_total,
        "category": category.strip().lower(),
    }


def extract_receipt(receipt_text: str) -> dict[str, Any]:
    """Ask Ollama to extract a receipt, retrying once if its response is invalid."""
    for attempt in range(2):
        response = ollama.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": receipt_text},
            ],
            format="json",
            think=False,
        )
        try:
            return parse_receipt_response(response.message.content)
        except ReceiptParseError as exc:
            if attempt == 1:
                raise ReceiptParseError(
                    "Ollama returned invalid receipt data after one retry."
                ) from exc
    raise AssertionError("The receipt extraction retry loop did not return.")


def _format_total(total: Decimal) -> str:
    """Write totals as simple decimal numbers rather than currency-formatted text."""
    formatted = format(total, "f")
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted or "0"


def write_csv(
    receipts: list[dict[str, Any]], output_path: Path | None = None
) -> None:
    """Write the extracted receipt fields to a CSV file."""
    if output_path is None:
        output_path = OUTPUT_CSV
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for receipt in receipts:
            row = dict(receipt)
            row["total"] = _format_total(receipt["total"])
            writer.writerow(row)


def print_category_totals(receipts: list[dict[str, Any]]) -> None:
    """Print total spending for every supported category."""
    totals = defaultdict(Decimal)
    for receipt in receipts:
        totals[receipt["category"]] += receipt["total"]

    print("Total spending per category:")
    for category in CATEGORIES:
        print(f"{category}: {_format_total(totals[category])}")


def main() -> None:
    """Process every .txt receipt in the receipts folder and write output.csv."""
    if not RECEIPTS_DIR.is_dir():
        raise FileNotFoundError(f"Receipts folder not found: {RECEIPTS_DIR}")

    receipt_files = sorted(RECEIPTS_DIR.glob("*.txt"))
    receipts = []
    for receipt_path in receipt_files:
        receipt_text = receipt_path.read_text(encoding="utf-8-sig")
        try:
            receipts.append(extract_receipt(receipt_text))
        except ReceiptParseError as exc:
            raise RuntimeError(f"Could not process {receipt_path.name}: {exc}") from exc

    write_csv(receipts)
    print(f"Saved {len(receipts)} receipt(s) to {OUTPUT_CSV.name}.")
    print_category_totals(receipts)


if __name__ == "__main__":
    main()
