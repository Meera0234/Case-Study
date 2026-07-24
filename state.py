"""Shared state schema -- the single source of truth for data shape.

Exactly one InvoiceState object flows through every LangGraph node. Each agent
returns a partial update; LangGraph merges it back. Agents import their types
from here rather than redefining shapes, which is what keeps four independently
written agents compatible through one object.

Four fields here go beyond the minimum the brief describes. Each earns its place
by catching a class of error the sample data actually contains:

  LineItem.unit_price   -- without it a stated total cannot be reconciled against
                           the line items, and two sample invoices overstate
                           their totals by amounts a human would never spot.
  invoice_number        -- the key duplicate detection is keyed on.
  subtotal / tax_amount -- separates "this total is wrong" from "this total
  / other_charges          includes tax and shipping", which is the difference
                           between a real finding and a false alarm.
  currency              -- one sample invoice is in EUR; paying it as USD would
                           silently overpay.
"""

from typing import List, Literal, Optional, TypedDict


class LineItem(TypedDict):
    name: str
    quantity: int
    unit_price: Optional[float]


class ExtractedInvoice(TypedDict):
    invoice_number: Optional[str]
    vendor: Optional[str]
    amount: Optional[float]
    subtotal: Optional[float]
    tax_amount: Optional[float]
    other_charges: Optional[float]       # shipping, freight, handling
    currency: str
    items: List[LineItem]
    due_date: Optional[str]
    raw_text: str
    extraction_notes: List[str]
    extraction_method: Literal["llm", "fallback_regex"]


class ValidationIssue(TypedDict):
    severity: Literal["info", "warning", "critical"]
    code: str
    message: str


class ApprovalDecision(TypedDict):
    decision: Literal["approved", "rejected", "pending"]
    reasoning: str
    critique_rounds: int
    requires_scrutiny: bool


class InvoiceState(TypedDict, total=False):
    invoice_path: str
    extracted: ExtractedInvoice
    computed_total: Optional[float]      # recomputed from line items; this is what gets paid
    validation_issues: List[ValidationIssue]
    validation_status: Literal["passed", "failed", "warning"]
    approval: ApprovalDecision
    payment_result: dict
    stage: str
    log: List[str]
    error: Optional[str]


# Human-readable explanations, shown in the dashboard so an AP clerk sees what a
# code means without reading the source.
ISSUE_CODES = {
    "UNKNOWN_ITEM":        "Item is not in the inventory catalogue",
    "OUT_OF_STOCK":        "Item exists but has zero stock",
    "INSUFFICIENT_STOCK":  "Requested quantity exceeds stock on hand",
    "INVALID_QUANTITY":    "Quantity is negative or nonsensical",
    "AMOUNT_MISMATCH":     "Stated total does not match the line items",
    "INVOICE_REVISION":    "Invoice number already seen with different figures",
    "REPROCESSED":         "Same invoice seen again, identical content",
    "THRESHOLD_HUGGING":   "Amount sits just below the approval threshold",
    "MISSING_VENDOR":      "No vendor could be identified",
    "MISSING_AMOUNT":      "No invoice total could be identified",
    "MISSING_DUE_DATE":    "No usable due date",
    "NO_LINE_ITEMS":       "No line items could be extracted",
    "NON_USD_CURRENCY":    "Not in USD and no exchange rate is available offline",
    "SUSPICIOUS_LANGUAGE": "Payment-pressure wording typical of invoice fraud",
    "ITEM_OK":             "Item found with sufficient stock",
}


def empty_extracted(raw_text: str = "") -> ExtractedInvoice:
    """A well-formed but empty ExtractedInvoice.

    Ingestion never raises; on total failure it returns this shape with the
    reason recorded in extraction_notes, so downstream agents can rely on the
    keys existing rather than guarding every access.
    """
    return {
        "invoice_number": None,
        "vendor": None,
        "amount": None,
        "subtotal": None,
        "tax_amount": None,
        "other_charges": None,
        "currency": "USD",
        "items": [],
        "due_date": None,
        "raw_text": raw_text,
        "extraction_notes": [],
        "extraction_method": "fallback_regex",
    }
