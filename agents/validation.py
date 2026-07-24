"""Validation Agent -- deterministic fact-check against inventory.db.

Reads:  extracted, computed_total
Writes: validation_issues, validation_status

No LLM influences pass/fail. The one optional LLM call writes a prose summary to
the log and nothing else, so an invoice routes identically with or without a
key -- which is what makes the outcome auditable.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
from typing import Any, Dict, List, Optional

import llm_client
from config import (
    INVENTORY_DB_PATH,
    MONEY_TOLERANCE,
    SCRUTINY_THRESHOLD,
    THRESHOLD_HUG_BAND,
)
from db import ledger
from state import InvoiceState, ValidationIssue

logger = logging.getLogger(__name__)

# Wording that appears in invoice-fraud attempts to rush a payment past review.
FRAUD_TERMS = ["urgent", "immediately", "wire transfer", "avoid penalt", "asap"]


def normalise_item_name(name: str) -> str:
    """'Widget A' and 'WidgetA (rush order)' both resolve to 'WidgetA'.

    Deliberately exact-match after cleanup rather than fuzzy: fuzzy matching
    would absorb an unknown 'WidgetC' into 'WidgetA' and hide a real problem.
    """
    without_parens = re.sub(r"\([^)]*\)", "", name or "")
    return re.sub(r"[^A-Za-z0-9]", "", without_parens)


def query_inventory(item_name: str, db_path: str = INVENTORY_DB_PATH) -> Optional[int]:
    """Stock level for an item, or None if it isn't in the catalogue."""
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                "SELECT stock FROM inventory WHERE LOWER(item) = LOWER(?)",
                (normalise_item_name(item_name),),
            ).fetchone()
    except sqlite3.Error as exc:
        logger.warning("Inventory lookup failed for %r: %s", item_name, exc)
        return None
    return row[0] if row else None


def check_quantity(requested: int, stock: int) -> Optional[str]:
    """Return an issue code, or None if the quantity is satisfiable."""
    if requested < 0:
        return "INVALID_QUANTITY"
    if stock == 0:
        return "OUT_OF_STOCK"
    if requested > stock:
        return "INSUFFICIENT_STOCK"
    return None


def flag_issue(severity: str, code: str, message: str) -> ValidationIssue:
    return {"severity": severity, "code": code, "message": message}


def _aggregate_quantities(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Sum quantities per item *before* comparing against stock.

    A bulk invoice can list the same item on several lines, each individually
    within stock while the total is far over. Checking line by line passes it;
    one sample invoice is exactly this case.
    """
    totals: Dict[str, Dict[str, Any]] = {}
    for item in items:
        key = normalise_item_name(item.get("name", ""))
        if not key:
            continue
        entry = totals.setdefault(key, {"display": item.get("name", key), "quantity": 0})
        entry["quantity"] += int(item.get("quantity", 0))
    return totals


def _check_completeness(extracted: Dict[str, Any]) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    if not extracted.get("vendor"):
        issues.append(flag_issue("critical", "MISSING_VENDOR",
                                 "No vendor could be identified on the invoice"))
    if extracted.get("amount") is None:
        issues.append(flag_issue("critical", "MISSING_AMOUNT",
                                 "No invoice total could be identified"))
    if not extracted.get("due_date"):
        issues.append(flag_issue("warning", "MISSING_DUE_DATE",
                                 "No usable due date; payment timing cannot be scheduled"))
    if not extracted.get("items"):
        issues.append(flag_issue("critical", "NO_LINE_ITEMS",
                                 "No line items could be extracted from the document"))
    currency = (extracted.get("currency") or "USD").upper()
    if currency != "USD":
        issues.append(flag_issue("critical", "NON_USD_CURRENCY",
                                 f"Invoice is denominated in {currency}; no exchange rate "
                                 "is available offline, so the payable amount is unknown"))
    return issues


def _check_totals(extracted: Dict[str, Any],
                  computed_total: Optional[float]) -> List[ValidationIssue]:
    """Compare the stated total against one recomputed from line items.

    This is the check that only became possible once LineItem carried a unit
    price -- and it is the one that catches errors a human reviewer never would,
    because the wrong number looks entirely plausible on the page.
    """
    stated = extracted.get("amount")
    if stated is None or computed_total is None:
        return []
    difference = round(stated - computed_total, 2)
    if abs(difference) <= MONEY_TOLERANCE:
        return []
    direction = "overstated" if difference > 0 else "understated"
    return [flag_issue(
        "critical", "AMOUNT_MISMATCH",
        f"Stated total ${stated:,.2f} is {direction} by ${abs(difference):,.2f}; "
        f"line items plus tax come to ${computed_total:,.2f}")]


def _check_fraud_signals(extracted: Dict[str, Any]) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    haystack = f"{' '.join(extracted.get('extraction_notes', []))} " \
               f"{extracted.get('raw_text', '')}".lower()
    hits = [term for term in FRAUD_TERMS if term in haystack]
    if hits:
        issues.append(flag_issue("warning", "SUSPICIOUS_LANGUAGE",
                                 "Payment-pressure wording present: " + ", ".join(hits)))

    amount = extracted.get("amount")
    lower_bound = SCRUTINY_THRESHOLD * (1 - THRESHOLD_HUG_BAND)
    if amount is not None and lower_bound <= amount < SCRUTINY_THRESHOLD:
        # Splitting or shaving an invoice to sit just under an approval limit is
        # a documented AP fraud pattern, and it is invisible to a per-invoice
        # eyeball check.
        issues.append(flag_issue(
            "warning", "THRESHOLD_HUGGING",
            f"${amount:,.2f} sits just below the ${SCRUTINY_THRESHOLD:,} approval "
            f"threshold ({(1 - amount / SCRUTINY_THRESHOLD) * 100:.1f}% under)"))
    return issues


def _check_duplicates(extracted: Dict[str, Any], source_file: str) -> List[ValidationIssue]:
    """Ask the ledger whether this invoice number has been seen before.

    Two situations, deliberately separated:

      identical content  -> the same document arriving twice, e.g. a PDF and its
                            text twin. Informational.
      different content  -> the same invoice number carrying different figures.

    The second case is flagged as INVOICE_REVISION rather than as fraud, because
    the document alone cannot tell you which it is. One sample invoice says
    "Revised invoice - additional items added per PO amendment", which is an
    ordinary business event; an attacker re-submitting at a higher value looks
    identical on the page. Both are held for manual reconciliation and neither is
    auto-superseded -- that policy is published in the impact report as a stated
    assumption rather than left for a reader to infer.
    """
    number = extracted.get("invoice_number")
    fingerprint = ledger.fingerprint(extracted.get("items", []), extracted.get("amount"))
    prior = ledger.check_and_record(number, fingerprint, extracted.get("vendor"),
                                    extracted.get("amount"), source_file)
    if prior is None:
        return []
    if prior["identical"]:
        return [flag_issue("info", "REPROCESSED",
                           f"Identical content already seen from {prior['source_file']}")]
    return [flag_issue(
        "critical", "INVOICE_REVISION",
        f"Invoice {number} was already seen from {os.path.basename(prior['source_file'])} "
        f"at ${prior['amount'] or 0:,.2f}; this copy states "
        f"${extracted.get('amount') or 0:,.2f}. Held for reconciliation -- paying both "
        "would pay twice for the items common to each.")]


def _check_inventory(extracted: Dict[str, Any]) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    for key, entry in sorted(_aggregate_quantities(extracted.get("items", [])).items()):
        requested, display = entry["quantity"], entry["display"]
        stock = query_inventory(key)
        if stock is None:
            issues.append(flag_issue("critical", "UNKNOWN_ITEM",
                                     f"'{display}' is not in the inventory catalogue"))
            continue
        code = check_quantity(requested, stock)
        if code == "INVALID_QUANTITY":
            issues.append(flag_issue("critical", code,
                                     f"'{display}' has a negative quantity ({requested})"))
        elif code == "OUT_OF_STOCK":
            issues.append(flag_issue("critical", code,
                                     f"'{display}' has zero stock on hand"))
        elif code == "INSUFFICIENT_STOCK":
            issues.append(flag_issue("critical", code,
                                     f"'{display}': {requested} requested, {stock} in stock"))
        else:
            issues.append(flag_issue("info", "ITEM_OK",
                                     f"'{display}': {requested} of {stock} available"))
    return issues


def validate(state: InvoiceState) -> Dict[str, Any]:
    """LangGraph node. Returns validation_issues and validation_status."""
    extracted = state.get("extracted") or {}
    log = list(state.get("log", []))

    issues: List[ValidationIssue] = []
    issues += _check_completeness(extracted)
    issues += _check_totals(extracted, state.get("computed_total"))
    issues += _check_inventory(extracted)
    issues += _check_duplicates(extracted, state.get("invoice_path", "unknown"))
    issues += _check_fraud_signals(extracted)

    if any(i["severity"] == "critical" for i in issues):
        status = "failed"
    elif any(i["severity"] == "warning" for i in issues):
        status = "warning"
    else:
        status = "passed"

    critical = sum(1 for i in issues if i["severity"] == "critical")
    warnings = sum(1 for i in issues if i["severity"] == "warning")
    log.append(f"[validate] status={status}, {critical} critical, {warnings} warning")

    notable = [i for i in issues if i["severity"] != "info"]
    if notable:
        summary = llm_client.summarise_findings(
            "; ".join(f"{i['code']} ({i['severity']}): {i['message']}" for i in notable))
        if summary:
            log.append(f"[validate] summary: {summary.strip()}")

    return {"validation_issues": issues, "validation_status": status,
            "stage": "validated", "log": log}
