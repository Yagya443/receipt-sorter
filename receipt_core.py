"""Core logic for Receipt Sorter Pro: local LLM extraction, validation, SQLite storage."""

import hashlib
import io
import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import ollama
import pandas as pd
from dateutil import parser as dateparser
from pypdf import PdfReader

DB_PATH = Path(__file__).resolve().parent / "receipts.db"  # local only, keep private
CATEGORIES = ["food", "travel", "software", "office", "utilities", "marketing", "other"]


# ---------- Reading files ----------
def read_file_text(name: str, data: bytes) -> str:
    """Return plain text from a .txt or text-based .pdf upload."""
    if name.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return data.decode("utf-8-sig", errors="replace")


def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------- LLM extraction ----------
def _schema(categories):
    return {
        "type": "object",
        "properties": {
            "vendor": {"type": "string"},
            "date": {"type": "string"},
            "currency": {"type": "string"},
            "tax": {"type": "number"},
            "total": {"type": "number"},
            "payment_method": {"type": "string"},
            "category": {"type": "string", "enum": categories},
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "amount": {"type": "number"},
                    },
                    "required": ["description", "amount"],
                },
            },
        },
        "required": ["vendor", "date", "currency", "tax", "total", "category", "items"],
    }


def _num(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(re.sub(r"[^\d.\-]", "", str(value).replace(",", "")))
    except ValueError:
        return 0.0


def _iso_date(value) -> str:
    try:
        return dateparser.parse(str(value), fuzzy=True).date().isoformat()
    except Exception:
        return ""


def _clean(data: dict, text: str, categories) -> dict:
    total, tax = _num(data.get("total")), _num(data.get("tax"))
    category = str(data.get("category", "other")).strip().lower()
    items = [
        {"description": str(i.get("description", "")), "amount": _num(i.get("amount"))}
        for i in data.get("items", [])
        if isinstance(i, dict)
    ]
    # Hallucination guard: the total must literally appear somewhere in the receipt.
    numbers = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", text.replace(",", ""))]
    needs_review = total <= 0 or not any(abs(n - total) < 0.005 for n in numbers)
    date = _iso_date(data.get("date"))
    return {
        "vendor": str(data.get("vendor", "")).strip() or "Unknown",
        "date": date,
        "currency": str(data.get("currency", "")).strip().upper()[:3],
        "tax": tax,
        "total": total,
        "payment_method": str(data.get("payment_method", "")).strip(),
        "category": category if category in categories else "other",
        "items": items,
        "needs_review": int(needs_review or not date),
    }


def extract_receipt(text: str, model: str, categories=CATEGORIES) -> dict:
    """Ask a local Ollama model for structured receipt data; retry once on bad output."""
    system = (
        "You extract structured data from receipts. Use only information present in the "
        "receipt. total is the final amount paid. Use an empty string or 0 if unknown. "
        f"category must be one of: {', '.join(categories)}."
    )
    last_error = None
    for _ in range(2):
        response = ollama.chat(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": text}],
            format=_schema(categories),
            think=False,
            options={"temperature": 0},
        )
        raw = re.sub(r"<think>.*?</think>", "", response.message.content, flags=re.S).strip()
        try:
            return _clean(json.loads(raw), text, categories)
        except (json.JSONDecodeError, AttributeError, TypeError) as exc:
            last_error = exc
    raise ValueError(f"Model returned invalid data twice: {last_error}")


# ---------- Database ----------
@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with db() as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS receipts (
                id INTEGER PRIMARY KEY AUTOINCREMENT, file_hash TEXT UNIQUE, filename TEXT,
                vendor TEXT, date TEXT, currency TEXT, tax REAL, total REAL,
                payment_method TEXT, category TEXT, items TEXT, needs_review INTEGER,
                added_at TEXT)"""
        )
        c.execute("CREATE TABLE IF NOT EXISTS rules (pattern TEXT PRIMARY KEY, category TEXT)")


def get_rules() -> dict:
    with db() as c:
        return dict(c.execute("SELECT pattern, category FROM rules").fetchall())


def add_rule(pattern: str, category: str):
    with db() as c:
        c.execute("INSERT OR REPLACE INTO rules VALUES (?, ?)", (pattern.strip().lower(), category))


def delete_rule(pattern: str):
    with db() as c:
        c.execute("DELETE FROM rules WHERE pattern = ?", (pattern,))


def apply_rules(vendor: str, category: str) -> str:
    """Vendor rules override the model, so recurring vendors are always categorised the same."""
    for pattern, rule_category in get_rules().items():
        if pattern in vendor.lower():
            return rule_category
    return category


def is_duplicate(h: str) -> bool:
    with db() as c:
        return c.execute("SELECT 1 FROM receipts WHERE file_hash = ?", (h,)).fetchone() is not None


def save_receipt(h: str, filename: str, r: dict):
    r["category"] = apply_rules(r["vendor"], r["category"])
    with db() as c:
        c.execute(
            """INSERT INTO receipts (file_hash, filename, vendor, date, currency, tax, total,
               payment_method, category, items, needs_review, added_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (h, filename, r["vendor"], r["date"], r["currency"], r["tax"], r["total"],
             r["payment_method"], r["category"], json.dumps(r["items"]), r["needs_review"],
             datetime.now().isoformat(timespec="seconds")),
        )


def load_receipts() -> pd.DataFrame:
    with db() as c:
        return pd.read_sql_query("SELECT * FROM receipts ORDER BY date DESC, id DESC", c)


def update_receipts(df: pd.DataFrame):
    """Save manual corrections; rows ticked 'delete' are removed, edited rows are marked reviewed."""
    with db() as c:
        for _, row in df.iterrows():
            if row.get("delete"):
                c.execute("DELETE FROM receipts WHERE id = ?", (int(row["id"]),))
            else:
                c.execute(
                    """UPDATE receipts SET vendor=?, date=?, currency=?, tax=?, total=?,
                       payment_method=?, category=?, needs_review=? WHERE id=?""",
                    (row["vendor"], row["date"], row["currency"], float(row["tax"]),
                     float(row["total"]), row["payment_method"], row["category"],
                     int(bool(row["needs_review"])), int(row["id"])),
                )