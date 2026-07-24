"""Invoice ledger -- the memory that makes duplicate detection possible.

A pipeline that only ever sees one invoice at a time cannot notice that it paid
the same invoice yesterday. This table is that memory: every invoice that
reaches validation is recorded, and every payment is recorded separately.

Two distinct situations, deliberately treated differently:

  Same invoice number, identical content   -> the same document arriving twice
                                              (e.g. PDF and its text version).
                                              Informational, not a fault.
  Same invoice number, different content   -> a revision or a fraud attempt.
                                              Critical: paying both means paying
                                              for the overlapping items twice.

Lives in the same SQLite file as inventory, in its own tables.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import INVENTORY_DB_PATH  # noqa: E402

logger = logging.getLogger(__name__)


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS invoice_ledger ("
        "  invoice_number TEXT NOT NULL,"
        "  fingerprint    TEXT NOT NULL,"
        "  vendor         TEXT,"
        "  amount         REAL,"
        "  source_file    TEXT,"
        "  seen_at        TEXT NOT NULL,"
        "  PRIMARY KEY (invoice_number, fingerprint)"
        ")"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS payment_ledger ("
        "  invoice_number TEXT PRIMARY KEY,"   # the idempotency guarantee
        "  vendor         TEXT,"
        "  amount         REAL,"
        "  paid_at        TEXT NOT NULL"
        ")"
    )
    return conn


def fingerprint(items: List[Dict[str, Any]], amount: Optional[float]) -> str:
    """Content hash of an invoice.

    Built from line items and the stated total, so the same document read from
    two file formats produces the same fingerprint while a revised invoice
    produces a different one.
    """
    body = sorted((str(i.get("name")), i.get("quantity"), i.get("unit_price"))
                  for i in items)
    payload = json.dumps([body, round(amount or 0.0, 2)], default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def check_and_record(invoice_number: Optional[str], fp: str, vendor: Optional[str],
                     amount: Optional[float], source_file: str,
                     db_path: str = INVENTORY_DB_PATH) -> Optional[Dict[str, Any]]:
    """Record this sighting and report any prior one.

    Returns None if unseen, otherwise a dict describing the prior sighting and
    whether the content matches.
    """
    if not invoice_number:
        return None                     # nothing to key on; validation flags this separately

    try:
        with _connect(db_path) as conn:
            rows = conn.execute(
                "SELECT fingerprint, vendor, amount, source_file, seen_at "
                "FROM invoice_ledger WHERE invoice_number = ?",
                (invoice_number,),
            ).fetchall()

            prior = None
            if rows:
                exact = next((r for r in rows if r[0] == fp), None)
                row = exact or rows[0]
                prior = {"identical": exact is not None, "vendor": row[1],
                         "amount": row[2], "source_file": row[3], "seen_at": row[4]}

            conn.execute(
                "INSERT OR IGNORE INTO invoice_ledger "
                "(invoice_number, fingerprint, vendor, amount, source_file, seen_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (invoice_number, fp, vendor, amount, source_file,
                 datetime.now().isoformat(timespec="seconds")),
            )
            return prior
    except sqlite3.Error as exc:
        logger.warning("Ledger check failed for %s: %s", invoice_number, exc)
        return None


def already_paid(invoice_number: Optional[str],
                 db_path: str = INVENTORY_DB_PATH) -> Optional[Dict[str, Any]]:
    """Prior payment record, if any. Payment is the one irreversible step, so it
    is checked immediately before the call, not inferred from earlier state."""
    if not invoice_number:
        return None
    try:
        with _connect(db_path) as conn:
            row = conn.execute(
                "SELECT vendor, amount, paid_at FROM payment_ledger WHERE invoice_number = ?",
                (invoice_number,),
            ).fetchone()
    except sqlite3.Error as exc:
        logger.warning("Payment ledger read failed for %s: %s", invoice_number, exc)
        return None
    return {"vendor": row[0], "amount": row[1], "paid_at": row[2]} if row else None


def record_payment(invoice_number: Optional[str], vendor: Optional[str],
                   amount: float, db_path: str = INVENTORY_DB_PATH) -> None:
    if not invoice_number:
        return
    try:
        with _connect(db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO payment_ledger "
                "(invoice_number, vendor, amount, paid_at) VALUES (?, ?, ?, ?)",
                (invoice_number, vendor, amount,
                 datetime.now().isoformat(timespec="seconds")),
            )
    except sqlite3.Error as exc:
        logger.warning("Payment ledger write failed for %s: %s", invoice_number, exc)


def prior_exposure(invoice_number: Optional[str],
                   db_path: str = INVENTORY_DB_PATH) -> Optional[float]:
    """What has already been committed against this invoice number.

    Used to value a revision correctly. If an invoice is revised upward and both
    copies are paid, the loss is not the revision's face value -- it is the
    amount already paid for the items that appear on both, i.e. the overlap.
    Prefers an actual payment; falls back to the earliest sighting.
    """
    if not invoice_number:
        return None
    paid = already_paid(invoice_number, db_path)
    if paid:
        return paid["amount"]
    try:
        with _connect(db_path) as conn:
            row = conn.execute(
                "SELECT amount FROM invoice_ledger WHERE invoice_number = ? "
                "ORDER BY seen_at LIMIT 1", (invoice_number,)).fetchone()
    except sqlite3.Error as exc:
        logger.warning("Prior exposure lookup failed for %s: %s", invoice_number, exc)
        return None
    return row[0] if row else None


def reset(db_path: str = INVENTORY_DB_PATH) -> None:
    """Clear both ledgers. Used by the batch runner so a demo starts clean."""
    try:
        with _connect(db_path) as conn:
            conn.execute("DELETE FROM invoice_ledger")
            conn.execute("DELETE FROM payment_ledger")
    except sqlite3.Error as exc:
        logger.warning("Ledger reset failed: %s", exc)
