"""Ingestion Agent -- turn a raw file into structured fields.

Reads:  invoice_path
Writes: extracted, computed_total

Never raises. Anything ambiguous lands in extraction_notes, so downstream agents
and the audit trail can see what was uncertain rather than inferring it from
silence.

Structured formats (JSON, XML, CSV) are parsed by code, not by the LLM. Sending
well-formed JSON to a language model to "extract" spends tokens and invites a
hallucinated number where an exact one was already available.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import llm_client
from state import ExtractedInvoice, InvoiceState, LineItem, empty_extracted

logger = logging.getLogger(__name__)


# --- file readers -----------------------------------------------------------

def read_pdf(path: str) -> str:
    """Text layer only. Scanned/image PDFs are out of scope -- see README."""
    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber not installed; cannot read %s", path)
        return ""
    try:
        with pdfplumber.open(path) as pdf:
            return "\n".join((page.extract_text() or "") for page in pdf.pages)
    except Exception as exc:            # noqa: BLE001 - a corrupt file must not stop the run
        logger.warning("PDF read failed for %s: %s", path, exc)
        return ""


def read_txt(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Text read failed for %s: %s", path, exc)
        return ""


read_csv = read_txt
read_json = read_txt
read_xml = read_txt

_READERS = {".pdf": read_pdf, ".txt": read_txt, ".csv": read_csv,
            ".json": read_json, ".xml": read_xml}


def read_raw(path: str) -> str:
    """Dispatch by extension. Unknown extensions are read as plain text."""
    return _READERS.get(Path(path).suffix.lower(), read_txt)(path)


# --- normalisation ----------------------------------------------------------

def fix_ocr(text: str) -> str:
    """Repair capital 'O' read as zero inside numeric tokens.

    One sample invoice contains '2O26' and '$3,500.O0'. Only tokens that are
    otherwise entirely numeric are touched, so real words are left alone.
    """
    def fix_part(part: str) -> str:
        core = part.strip("$,.")
        if core and any(c.isdigit() for c in core) and all(c in "0123456789Oo,." for c in core):
            return part.replace("O", "0").replace("o", "0")
        return part

    out: List[str] = []
    for token in re.split(r"(\s+)", text):
        if not token.strip():
            out.append(token)
            continue
        out.append("".join(fix_part(p) for p in re.split(r"([-/:])", token)))
    return "".join(out)


def to_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = re.sub(r"[^0-9.\-]", "", str(value))
    if cleaned in ("", "-", "."):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


_DATE_FORMATS = ["%Y-%m-%d", "%d-%b-%Y", "%d %b %Y", "%b %d %Y", "%B %d, %Y",
                 "%b %d, %Y", "%m/%d/%Y", "%Y/%m/%d", "%d-%B-%Y"]
_JUNK_DATES = {"yesterday", "today", "tomorrow", "asap", "immediate",
               "immediately", "n/a", "tbd", ""}


def normalise_date(value: Any) -> Optional[str]:
    """ISO-format a date, or None if it isn't one.

    Junk words return None so validation flags a missing due date rather than
    trusting the word "yesterday" as a payment deadline.
    """
    if not value:
        return None
    text = str(value).strip().rstrip(".,;")
    if text.lower() in _JUNK_DATES:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def detect_currency(text: str, default: str = "USD") -> str:
    match = re.search(r"\b(USD|EUR|GBP|CAD|AUD|JPY|CHF|INR)\b", text)
    if match:
        return match.group(1)
    if "€" in text:
        return "EUR"
    if "£" in text:
        return "GBP"
    return default


def normalise_invoice_number(value: Any) -> Optional[str]:
    """Canonical form of an invoice number, applied to every extraction path.

    This matters more than it looks. The invoice number is the key for duplicate
    detection and for payment idempotency, and the two extraction paths do not
    naturally agree on it: the regex path yields 'INV-1012' while an LLM reading
    the same document returns 'INV 1012' verbatim, exactly as printed. Two
    spellings mean two ledger entries, which means the same invoice can be paid
    twice -- once per extraction mode.

    Anything with digits collapses to INV-<digits>; anything else is upper-cased
    and stripped of punctuation so it at least compares consistently.
    """
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    digits = re.search(r"(\d{3,})", text)
    if digits:
        return f"INV-{digits.group(1)}"
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").upper() or None


def normalise_vendor(value: Any) -> Optional[str]:
    """Canonical vendor name.

    Same reasoning as invoice numbers: the regex path strips a trailing
    '(formerly FastShip Ltd.)' while an LLM keeps it, so payment records for one
    supplier differ by extraction mode. Vendor is not a key, but an audit trail
    that names the same company two ways is a bad audit trail.
    """
    if not value:
        return None
    text = re.sub(r"\([^)]*\)", "", str(value))
    # PDF extraction collapses column gaps, gluing the next field's label on.
    text = re.split(r"\s+(?:Due|Date|Terms|Invoice|Attn|PO|Total)\b\s*:?",
                    text, flags=re.IGNORECASE)[0]
    # Trailing periods are kept: "Widgets Inc." is the company's name, not a
    # sentence ending.
    return re.sub(r"\s+", " ", text).strip(" ,;:-") or None


# --- deterministic extraction ----------------------------------------------

_STOPWORDS = {"subtotal", "total", "tax", "salestax", "shipping", "grandtotal",
              "amount", "item", "description", "paymentterms", "terms", "notes",
              "qty", "rate", "unitprice", "linetotal", "balance", "due",
              "invoice", "vendor", "date"}

_ITEM_PATTERNS = [
    # "WidgetA qty: 10 unit price: $250.00" / "GadgetX qty 20 @ $750" / "- SuperGizmo x12 $400.00"
    re.compile(r"^\s*[-*]?\s*(?P<name>[A-Za-z][A-Za-z0-9 ()]*?)\s+(?:qty:?\s*|x)"
               r"(?P<qty>-?\d+(?:\.\d+)?)\s+(?:unit\s*price:?\s*|@\s*)?"
               r"\$?(?P<price>[\d,]+(?:\.\d+)?)", re.IGNORECASE),
    # aligned table row. Single \s+ rather than \s{2,} because pdfplumber
    # collapses column padding into one space -- requiring two matched nothing.
    re.compile(r"^\s*[-*]?\s*(?P<name>[A-Za-z][A-Za-z0-9 ()]*?)\s+(?P<qty>-?\d+(?:\.\d+)?)\s+"
               r"\$(?P<price>[\d,]+(?:\.\d+)?)"
               r"(?:\s+\$(?P<amount>[\d,]+(?:\.\d+)?))?(?P<note>[A-Za-z ]*)\s*$"),
]


def _is_stopword(name: str) -> bool:
    return re.sub(r"[^A-Za-z0-9]", "", name or "").lower() in _STOPWORDS


def _find_label(text: str, *labels: str, exclude: str | None = None,
                numeric: bool = False) -> Optional[str]:
    """Find 'Label: value' on a single line.

    Uses [ \\t] rather than \\s deliberately: \\s crosses newlines, which made a
    table header's 'TOTAL' capture the dashed separator line underneath it.
    """
    for label in labels:
        for match in re.finditer(rf"{label}[ \t]*:?[ \t]*(.+)", text, re.IGNORECASE):
            line_start = text.rfind("\n", 0, match.start()) + 1
            if exclude and re.search(exclude, text[line_start:match.start()], re.IGNORECASE):
                continue
            value = match.group(1).strip()
            if not value:
                continue
            if numeric and to_number(value) is None:
                continue
            return value
    return None


def _from_json(raw: str) -> Dict[str, Any]:
    data = json.loads(raw)
    vendor = data.get("vendor")
    if isinstance(vendor, dict):
        vendor = vendor.get("name")
    items = [{"name": li.get("item", ""), "quantity": to_number(li.get("quantity")),
              "unit_price": to_number(li.get("unit_price"))}
             for li in data.get("line_items", [])]
    return {"invoice_number": data.get("invoice_number"), "vendor": vendor or None,
            "amount": to_number(data.get("total")),
            "subtotal": to_number(data.get("subtotal")),
            "tax_amount": to_number(data.get("tax_amount")),
            "other_charges": to_number(data.get("shipping") or data.get("other_charges")),
            "currency": data.get("currency") or "USD",
            "items": items, "due_date": data.get("due_date"),
            "notes": [data["notes"]] if data.get("notes") else []}


def _from_xml(raw: str) -> Dict[str, Any]:
    root = ET.fromstring(raw)

    def text_of(node, tag):
        found = node.find(tag)
        return found.text if found is not None else None

    items = []
    for node in root.findall(".//line_items/item"):
        name = text_of(node, "n") or text_of(node, "name") or ""
        items.append({"name": name, "quantity": to_number(text_of(node, "quantity")),
                      "unit_price": to_number(text_of(node, "unit_price"))})
    return {"invoice_number": text_of(root, "header/invoice_number"),
            "vendor": text_of(root, "header/vendor"),
            "amount": to_number(text_of(root, "totals/total")),
            "subtotal": to_number(text_of(root, "totals/subtotal")),
            "tax_amount": to_number(text_of(root, "totals/tax_amount")),
            "other_charges": to_number(text_of(root, "totals/shipping")),
            "currency": text_of(root, "header/currency") or "USD",
            "items": items, "due_date": text_of(root, "header/due_date"), "notes": []}


def _from_csv(raw: str) -> Dict[str, Any]:
    rows = [r for r in csv.reader(io.StringIO(raw)) if any(c.strip() for c in r)]
    result: Dict[str, Any] = {"invoice_number": None, "vendor": None, "amount": None,
                              "subtotal": None, "tax_amount": None, "other_charges": None,
                              "currency": "USD", "items": [], "due_date": None, "notes": []}
    if not rows:
        return result
    header = [c.strip().lower() for c in rows[0]]

    # Layout A: field,value with 'item' repeated. csv.DictReader would keep only
    # the last occurrence, silently discarding every earlier line item.
    if header[:2] == ["field", "value"]:
        current: Optional[Dict[str, Any]] = None
        for row in rows[1:]:
            if len(row) < 2:
                continue
            key, value = row[0].strip().lower(), row[1].strip()
            if key == "item":
                current = {"name": value, "quantity": None, "unit_price": None}
                result["items"].append(current)
            elif key == "quantity" and current:
                current["quantity"] = to_number(value)
            elif key == "unit_price" and current:
                current["unit_price"] = to_number(value)
            elif key == "invoice_number":
                result["invoice_number"] = value
            elif key == "vendor":
                result["vendor"] = value
            elif key == "due_date":
                result["due_date"] = value
            elif key == "subtotal":
                result["subtotal"] = to_number(value)
            elif key in ("tax", "tax_amount"):
                result["tax_amount"] = to_number(value)
            elif key in ("shipping", "freight", "other_charges"):
                result["other_charges"] = to_number(value)
            elif key == "total":
                result["amount"] = to_number(value)
        return result

    # Layout B: wide table with footer total rows.
    index = {name: i for i, name in enumerate(header)}

    def cell(row: List[str], key: str) -> str:
        i = index.get(key)
        return row[i].strip() if i is not None and i < len(row) else ""

    for row in rows[1:]:
        if row and row[0].strip():
            result["items"].append({"name": cell(row, "item"),
                                    "quantity": to_number(cell(row, "qty")),
                                    "unit_price": to_number(cell(row, "unit price"))})
            result["invoice_number"] = result["invoice_number"] or row[0].strip()
            result["vendor"] = result["vendor"] or cell(row, "vendor")
            result["due_date"] = result["due_date"] or cell(row, "due date")
        else:
            cells = [c.strip() for c in row if c.strip()]
            if len(cells) >= 2:
                label, value = cells[-2].lower(), to_number(cells[-1])
                if "subtotal" in label:          # checked before "total": it contains it
                    result["subtotal"] = value
                elif "tax" in label:
                    result["tax_amount"] = value
                elif "shipping" in label or "freight" in label:
                    result["other_charges"] = value
                elif "total" in label:
                    result["amount"] = value
    return result


def _from_text(raw: str) -> Dict[str, Any]:
    """Regex extraction for free-text and PDF-derived invoices."""
    items = []
    for line in raw.splitlines():
        for pattern in _ITEM_PATTERNS:
            match = pattern.match(line)
            if match and not _is_stopword(match.group("name")):
                items.append({"name": match.group("name").strip(),
                              "quantity": to_number(match.group("qty")),
                              "unit_price": to_number(match.group("price"))})
                break

    number_match = (re.search(r"INV[\s\-#]*(\d{3,})", raw, re.IGNORECASE)
                    or re.search(r"Inv\s*#?\s*:?\s*(\d{3,})", raw, re.IGNORECASE))

    vendor = _find_label(raw, r"\bVendor\b", r"\bVndr\b", r"\bFROM\b")
    if vendor:
        vendor = re.sub(r"\(.*?\)", "", vendor)
        # PDF extraction collapses column gaps, gluing the next column's label
        # onto the vendor name -- cut at any trailing field label.
        vendor = re.split(r"\s+(?:Due|Date|Terms|Invoice|Attn|PO|Total)\b\s*:?",
                          vendor, flags=re.IGNORECASE)[0].strip()

    notes_text = _find_label(raw, r"\bNotes\b")
    return {"invoice_number": f"INV-{number_match.group(1)}" if number_match else None,
            "vendor": vendor or None,
            "amount": to_number(_find_label(raw, r"\bTotal Amount\b", r"\bGrand Total\b",
                                            r"\bTotal\b", r"\bAmt\b", numeric=True)),
            "subtotal": to_number(_find_label(raw, r"\bSubtotal\b", numeric=True)),
            "tax_amount": to_number(_find_label(raw, r"\bTax\b[^:$\n]*", numeric=True)),
            "other_charges": to_number(_find_label(raw, r"\bShipping\b", r"\bFreight\b",
                                                   r"\bHandling\b", numeric=True)),
            "currency": detect_currency(raw),
            "items": items,
            "due_date": _find_label(raw, r"\bDue Date\b", r"\bDue Dt\b", r"\bDue\b"),
            "notes": [notes_text] if notes_text else []}


def extract_fields_fallback(raw_text: str, path: str) -> Dict[str, Any]:
    """Deterministic extraction. Used when no LLM key is present, and whenever
    the file format is structured enough not to need one."""
    suffix = Path(path).suffix.lower()
    try:
        if suffix == ".json":
            return _from_json(raw_text)
        if suffix == ".xml":
            return _from_xml(raw_text)
        if suffix == ".csv":
            return _from_csv(raw_text)
    except Exception as exc:            # noqa: BLE001 - malformed file, not a crash
        logger.warning("Structured parse of %s failed (%s); using regex", path, exc)
    return _from_text(raw_text)


# --- assembly ---------------------------------------------------------------

def _coerce_items(raw_items: Any) -> List[LineItem]:
    items: List[LineItem] = []
    if not isinstance(raw_items, list):
        return items
    for entry in raw_items:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or entry.get("item") or "").strip()
        quantity = to_number(entry.get("quantity"))
        if not name or quantity is None:
            continue
        items.append({"name": name, "quantity": int(quantity),
                      "unit_price": to_number(entry.get("unit_price"))})
    return items


def compute_total(items: List[LineItem], tax_amount: Optional[float],
                  other_charges: Optional[float] = None) -> Optional[float]:
    """Recompute the invoice total from line items, tax and other charges.

    This is the number the payment agent uses. The stated total on the document
    is treated as a claim to be checked, not a fact to be paid.

    Shipping and freight are included deliberately: without them, every invoice
    carrying a delivery charge would raise a false mismatch, and a validator
    that cries wolf on legitimate invoices gets switched off.
    """
    priced = [i for i in items if i.get("unit_price") is not None]
    if not priced or len(priced) != len(items):
        return None
    subtotal = sum(i["quantity"] * (i["unit_price"] or 0.0) for i in priced)
    return round(subtotal + (tax_amount or 0.0) + (other_charges or 0.0), 2)


def ingest(state: InvoiceState) -> Dict[str, Any]:
    """LangGraph node. Returns a partial update: extracted + computed_total."""
    path = state["invoice_path"]
    log = list(state.get("log", []))
    notes: List[str] = []

    if not Path(path).exists():
        extracted = empty_extracted()
        extracted["extraction_notes"] = [f"File not found: {path}"]
        log.append(f"[ingest] file not found: {path}")
        return {"extracted": extracted, "computed_total": None, "stage": "ingested",
                "log": log, "error": f"File not found: {path}"}

    raw_text = read_raw(path)
    repaired = fix_ocr(raw_text)
    if repaired != raw_text:
        notes.append("OCR repair applied: capital 'O' read as zero in numeric tokens")
    raw_text = repaired

    if not raw_text.strip():
        extracted = empty_extracted()
        extracted["extraction_notes"] = ["File was empty or unreadable"]
        log.append(f"[ingest] no text extracted from {Path(path).name}")
        return {"extracted": extracted, "computed_total": None,
                "stage": "ingested", "log": log}

    # Free text is where an LLM earns its keep; structured formats are parsed
    # exactly by code.
    method = "fallback_regex"
    parsed: Optional[Dict[str, Any]] = None
    is_free_text = Path(path).suffix.lower() in (".txt", ".pdf")
    if is_free_text and llm_client.is_available():
        parsed, llm_notes = llm_client.extract_invoice(raw_text)
        notes.extend(llm_notes)
        if parsed is not None:
            method = "llm"
    if parsed is None:
        parsed = extract_fields_fallback(raw_text, path)

    items = _coerce_items(parsed.get("items"))
    amount = to_number(parsed.get("amount"))
    tax_amount = to_number(parsed.get("tax_amount"))
    other_charges = to_number(parsed.get("other_charges"))
    raw_due = parsed.get("due_date")
    due_date = normalise_date(raw_due)
    if raw_due and due_date is None:
        notes.append(f"Due date {raw_due!r} is not a parseable date")

    extra_notes = parsed.get("notes")
    if isinstance(extra_notes, list):
        notes.extend(str(n) for n in extra_notes if n)

    computed = compute_total(items, tax_amount, other_charges)
    if computed is None and items:
        notes.append("Some line items have no unit price; total could not be recomputed")
    if not items:
        notes.append("No line items could be extracted")
    if amount is None:
        notes.append("No invoice total could be extracted")

    extracted: ExtractedInvoice = {
        "invoice_number": normalise_invoice_number(parsed.get("invoice_number")),
        "vendor": normalise_vendor(parsed.get("vendor")),
        "amount": amount,
        "subtotal": to_number(parsed.get("subtotal")),
        "tax_amount": tax_amount,
        "other_charges": other_charges,
        "currency": (parsed.get("currency") or "USD").upper(),
        "items": items,
        "due_date": due_date,
        "raw_text": raw_text,
        "extraction_notes": notes,
        "extraction_method": method,
    }
    log.append(f"[ingest] {Path(path).name} via {method}: {len(items)} item(s), "
               f"stated={amount}, recomputed={computed}")
    return {"extracted": extracted, "computed_total": computed,
            "stage": "ingested", "log": log}
