"""Payment Agent -- pure execution, no LLM.

Reads:  approval, extracted, computed_total
Writes: payment_result

Two deliberate choices worth stating out loud, because both concern money
leaving the building:

1. The amount paid is the total **recomputed from line items**, never the total
   stated on the document. A supplier's arithmetic is a claim, not a fact.
2. Payment is checked against a ledger immediately before the call, so a crashed
   and re-run pipeline cannot pay the same invoice twice.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from config import SCRUTINY_THRESHOLD
from db import ledger
from state import InvoiceState, ValidationIssue

logger = logging.getLogger(__name__)


def mock_payment(vendor: str, amount: float) -> Dict[str, str]:
    print(f"Paid {amount} to {vendor}")
    return {"status": "success"}


def log_rejection(reasoning: str, validation_issues: List[ValidationIssue],
                  vendor: Optional[str] = None,
                  amount: Optional[float] = None) -> Dict[str, Any]:
    """Structured rejection record. No payment call is made."""
    contributing = [i for i in validation_issues if i["severity"] in ("critical", "warning")]
    logger.info("Rejected invoice from %s for %s: %s", vendor, amount, reasoning)
    return {"status": "rejected", "vendor": vendor, "amount": amount,
            "reasoning": reasoning, "contributing_issues": contributing}


def payment_blockers(state: InvoiceState) -> List[str]:
    """Reasons this invoice must not be paid, checked independently of routing.

    Defence in depth. The graph already routes critical failures away from
    payment, and the approval agent already refuses to approve them -- but both
    of those are decisions made elsewhere, and this is the one step that cannot
    be undone. If a future edit reorders the graph or an LLM response slips
    through, this function is what stops the money.
    """
    blockers: List[str] = []
    issues: List[ValidationIssue] = state.get("validation_issues", [])

    critical = [i for i in issues if i["severity"] == "critical"]
    if critical:
        blockers.append("critical validation issues present: "
                        + "; ".join(i["code"] for i in critical))

    if (state.get("approval") or {}).get("decision") != "approved":
        blockers.append("approval decision is not 'approved'")

    extracted = state.get("extracted") or {}
    if not extracted.get("vendor"):
        blockers.append("no payee identified")

    amount = state.get("computed_total")
    if amount is None:
        amount = extracted.get("amount")
    if amount is None:
        blockers.append("no payable amount could be determined")
    elif amount <= 0:
        blockers.append(f"payable amount is not positive ({amount})")

    currency = (extracted.get("currency") or "USD").upper()
    if currency != "USD":
        blockers.append(f"currency is {currency}, not USD")

    return blockers


def pay(state: InvoiceState) -> Dict[str, Any]:
    """LangGraph node `pay`. Only reachable once a decision of 'approved' has
    survived the critique loop -- and even then, re-checked here."""
    extracted = state.get("extracted") or {}
    log = list(state.get("log", []))
    number = extracted.get("invoice_number")
    vendor = extracted.get("vendor") or "UNKNOWN VENDOR"

    blockers = payment_blockers(state)
    if blockers:
        # Reaching here means something upstream is wrong, so it is logged loudly
        # rather than quietly returning.
        logger.error("PAYMENT GATE blocked %s: %s", number, "; ".join(blockers))
        result = {"status": "blocked_by_gate", "vendor": vendor, "amount": None,
                  "invoice_number": number,
                  "reasoning": "Payment gate refused: " + "; ".join(blockers)}
        log.append(f"[pay] GATE BLOCKED: {'; '.join(blockers)}")
        return {"payment_result": result, "stage": "payment_blocked", "log": log}

    prior = ledger.already_paid(number)
    if prior:
        result = {"status": "blocked_duplicate_payment", "vendor": vendor,
                  "amount": prior["amount"], "invoice_number": number,
                  "reasoning": f"{number} was already paid on {prior['paid_at']}"}
        log.append(f"[pay] blocked: {number} already paid on {prior['paid_at']}")
        return {"payment_result": result, "stage": "payment_blocked", "log": log}

    # The recomputed figure is authoritative; fall back to the stated total only
    # when line items carried no prices to recompute from.
    amount = state.get("computed_total")
    if amount is None:
        amount = extracted.get("amount") or 0.0
        log.append("[pay] no recomputed total available; using stated amount")

    result = mock_payment(vendor, amount)
    result.update({"vendor": vendor, "amount": amount, "invoice_number": number})
    ledger.record_payment(number, vendor, amount)
    log.append(f"[pay] paid {amount} to {vendor}")
    return {"payment_result": result, "stage": "paid", "log": log}


def reject_log(state: InvoiceState) -> Dict[str, Any]:
    """LangGraph node `reject_log`. Terminal path for both validation failures
    and approval rejections; the reasoning differs by which one arrived here."""
    extracted = state.get("extracted") or {}
    approval = state.get("approval") or {}
    issues: List[ValidationIssue] = state.get("validation_issues", [])
    log = list(state.get("log", []))

    reasoning = approval.get("reasoning")
    if not reasoning:
        critical = [i for i in issues if i["severity"] == "critical"]
        reasoning = ("Rejected at validation, before approval reasoning. Critical findings: "
                     + "; ".join(i["message"] for i in critical)) if critical else \
                    "Rejected: invoice did not pass validation."

    record = log_rejection(reasoning, issues, vendor=extracted.get("vendor"),
                           amount=extracted.get("amount"))
    record["invoice_number"] = extracted.get("invoice_number")
    log.append(f"[reject_log] rejected: {reasoning}")

    # Record the terminal decision so the final state is unambiguous even when
    # rejection happened before the approval agent ever ran. The scrutiny flag is
    # recomputed rather than defaulted to False: a $22,000 invoice rejected at
    # validation still exceeded the threshold, and an audit trail that says
    # otherwise is wrong.
    final_approval = dict(approval)
    amount = extracted.get("amount")
    final_approval["requires_scrutiny"] = bool(
        amount is not None and amount > SCRUTINY_THRESHOLD)
    final_approval.setdefault("critique_rounds", 0)
    final_approval["decision"] = "rejected"
    final_approval["reasoning"] = reasoning

    return {"payment_result": record, "approval": final_approval,
            "stage": "rejected", "log": log}
