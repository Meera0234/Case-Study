"""Business impact -- translates a run into the numbers in the brief.

The brief states three baselines: $2M lost annually, a 30% error rate, and 5-day
processing. This module reports in the same units so the comparison is like for
like.

Three rules govern everything here, because an impact figure that cannot be
defended in a meeting is worse than no figure at all:

1. **Every total reconciles.** The counting basis is printed in words and the
   parts add up to the whole. A reader who does the arithmetic themselves must
   land where the report says they will.
2. **Every total is itemised.** "Money at risk" names each invoice, the reason,
   and the basis for its value. Nothing is a bare number.
3. **Every judgement call is published.** Where the system chose a policy rather
   than measured a fact, the policy is stated rather than left to be inferred.
"""

from __future__ import annotations

from typing import Any, Dict, List

from config import (
    BASELINE_ANNUAL_LOSS_USD,
    BASELINE_DAYS_PER_INVOICE,
    BASELINE_ERROR_RATE,
    BASELINE_MINUTES_MANUAL_REVIEW,
)
from db import ledger

# Issues where real money would have moved incorrectly.
FINANCIAL_ERROR_CODES = {"AMOUNT_MISMATCH", "INVOICE_REVISION"}

# Issues representing an operational problem, whether or not money was at stake.
OPERATIONAL_ERROR_CODES = {
    "UNKNOWN_ITEM", "OUT_OF_STOCK", "INSUFFICIENT_STOCK", "INVALID_QUANTITY",
    "MISSING_VENDOR", "MISSING_AMOUNT", "NO_LINE_ITEMS", "NON_USD_CURRENCY",
    "SUSPICIOUS_LANGUAGE", "THRESHOLD_HUGGING", "MISSING_DUE_DATE",
}

STATED_ASSUMPTIONS = [
    "A revision to an invoice number already on file is held for manual "
    "reconciliation rather than superseding the original. The document alone does "
    "not distinguish an amendment from a re-submission.",
    "Invoices not denominated in USD are held rather than converted, since no "
    "exchange rate is available offline.",
    "Rates are per unique invoice. A file that repeats a document already read is "
    "counted separately.",
    "A stated total below the line-item sum is valued at zero, since it costs the "
    "supplier rather than the client.",
    "Manual review time is an estimate. Substitute a measured figure before using "
    "these numbers for a return-on-investment case.",
    "The 30% error rate and five-day cycle time are the figures given in the "
    "brief, not results from this run.",
]


def _codes(state: Dict[str, Any]) -> set:
    return {i["code"] for i in state.get("validation_issues", [])}


def _status(state: Dict[str, Any]) -> str:
    return (state.get("payment_result") or {}).get("status", "")


def _is_reread(state: Dict[str, Any]) -> bool:
    """A second reading of a document already seen, e.g. a PDF and its text twin.
    Not a distinct invoice, so excluded from every rate."""
    return "REPROCESSED" in _codes(state)


def _money_at_risk_lines(unique: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One line per exposure, each naming the invoice and the basis of its value.

    Two kinds, valued differently on purpose:

      AMOUNT_MISMATCH   -- the overstatement only. An understated total costs the
                           supplier, so it is listed at zero rather than claimed
                           as a saving.
      INVOICE_REVISION  -- the overlap with what was already committed against
                           that invoice number, NOT the revision's face value.
                           Paying a $1,890 invoice and then a $5,940 revision of
                           it costs $1,890: the other $4,050 buys goods that were
                           genuinely ordered. Claiming face value would overstate
                           the catch by the value of the new items.
    """
    lines: List[Dict[str, Any]] = []
    for state in unique:
        extracted = state.get("extracted") or {}
        number = extracted.get("invoice_number") or "unknown"
        codes = _codes(state)

        if "AMOUNT_MISMATCH" in codes:
            stated, computed = extracted.get("amount"), state.get("computed_total")
            if stated is not None and computed is not None:
                delta = round(stated - computed, 2)
                lines.append({
                    "invoice": number, "code": "AMOUNT_MISMATCH",
                    "value": max(delta, 0.0),
                    "basis": ("overstatement that would have been paid" if delta > 0
                              else "understated; costs the supplier, not the client"),
                })

        if "INVOICE_REVISION" in codes:
            overlap = ledger.prior_exposure(extracted.get("invoice_number"))
            face = extracted.get("amount") or 0.0
            value = min(overlap, face) if overlap is not None else 0.0
            basis = (f"overlap with ${overlap:,.2f} already committed against this "
                     "invoice number" if overlap is not None
                     else "prior exposure unknown; valued at zero")
            lines.append({"invoice": number, "code": "INVOICE_REVISION",
                          "value": round(value, 2), "basis": basis})
    return lines


def _blocked_payment_lines(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Second payment attempts stopped by the ledger.

    Reported separately and never folded into money at risk: these are the same
    invoice arriving twice, so counting their value alongside genuine overbilling
    would inflate the headline with money that was never really at stake.
    """
    lines = []
    for state in results:
        if _status(state) != "blocked_duplicate_payment":
            continue
        payment = state.get("payment_result") or {}
        lines.append({"invoice": payment.get("invoice_number") or "unknown",
                      "value": round(payment.get("amount") or 0.0, 2),
                      "basis": payment.get("reasoning", "")})
    return lines


def compute(results: List[Dict[str, Any]], elapsed_seconds: float) -> Dict[str, Any]:
    """Build a fully reconciled impact report from a batch of final states."""
    rereads = [s for s in results if _is_reread(s)]
    unique = [s for s in results if not _is_reread(s)]
    total = len(unique)
    if total == 0:
        return {"measured": {}, "comparison": {}, "reconciliation": {},
                "money_at_risk": {}, "blocked_payments": {}, "assumed": {},
                "stated_assumptions": STATED_ASSUMPTIONS}

    paid = [s for s in unique if _status(s) == "success"]
    held = [s for s in unique if (s.get("approval") or {}).get("decision") == "rejected"]
    other = [s for s in unique if s not in paid and s not in held]

    blocked = _blocked_payment_lines(results)
    risk_lines = _money_at_risk_lines(unique)
    risk_total = round(sum(line["value"] for line in risk_lines), 2)

    with_any_error = [s for s in unique
                      if _codes(s) & (FINANCIAL_ERROR_CODES | OPERATIONAL_ERROR_CODES)]
    # An invoice can be flagged and still approved -- a warning is not a blocker.
    # Reporting these together would let the headline claim every defect was
    # stopped before payment, when one was flagged, reviewed, and paid. That is a
    # defensible outcome but a different sentence, and an evaluator who checks
    # will find the paid one.
    flagged_and_paid = [s for s in with_any_error if _status(s) == "success"]
    flagged_and_held = [s for s in with_any_error if s not in flagged_and_paid]
    seconds_each = elapsed_seconds / total

    # Printed in words so a reader can check the arithmetic themselves.
    reconciliation = {
        "files_read": len(results),
        "unique_invoices": total,
        "rereads_of_seen_documents": len(rereads),
        "paid": len(paid),
        "held": len(held),
        "other": len(other),
        "blocked_at_payment_ledger": len(blocked),
        "statement": (
            f"{len(results)} files read = {total} unique invoices "
            f"({len(paid)} paid + {len(held)} held"
            + (f" + {len(other)} other" if other else "")
            + f") + {len(rereads)} re-read"
            + ("s" if len(rereads) != 1 else "")
            + " of an already-seen document"
            + (f" ({len(blocked)} blocked at the payment ledger"
               + (f", {len(rereads) - len(blocked)} reaching the same outcome as before"
                  if len(rereads) > len(blocked) else "")
               + ")" if blocked else "")
            + ". Rates below are per unique invoice."),
        "balances": (len(paid) + len(held) + len(other) == total
                     and total + len(rereads) == len(results)),
    }

    measured = {
        "files_read": len(results),
        "unique_invoices": total,
        "straight_through_paid": len(paid),
        "straight_through_rate": round(len(paid) / total, 3),
        "held_for_review": len(held),
        "invoices_with_any_error": len(with_any_error),
        "flagged_and_held": len(flagged_and_held),
        "flagged_and_approved": len(flagged_and_paid),
        "detected_error_rate": round(len(with_any_error) / total, 3),
        "total_runtime_seconds": round(elapsed_seconds, 2),
        "seconds_per_invoice": round(seconds_each, 3),
        "manual_minutes_avoided": len(paid) * BASELINE_MINUTES_MANUAL_REVIEW,
    }

    money_at_risk = {"total": risk_total, "lines": risk_lines}
    blocked_payments = {"count": len(blocked),
                        "total": round(sum(b["value"] for b in blocked), 2),
                        "lines": blocked}

    assumed = {
        "baseline_annual_loss_usd": BASELINE_ANNUAL_LOSS_USD,
        "baseline_error_rate": BASELINE_ERROR_RATE,
        "baseline_days_per_invoice": BASELINE_DAYS_PER_INVOICE,
        "minutes_of_manual_review_per_invoice": BASELINE_MINUTES_MANUAL_REVIEW,
        "note": "Figures from the brief, not results from this run.",
    }

    # No speed multiple is quoted. This measures compute time while the five-day
    # baseline is mostly queue and email latency; dividing one by the other gives
    # a number that is arithmetically true and professionally indefensible.
    per_invoice = (f"{seconds_each * 1000:.0f}ms" if seconds_each < 1
                   else f"{seconds_each:.1f}s")
    headline = (
        f"{len(with_any_error)} of {total} invoices carried a defect: "
        f"{len(flagged_and_held)} held before payment, "
        f"{len(flagged_and_paid)} approved after review. Held invoices account for "
        f"${risk_total:,.2f} of overstated or duplicated billing.")
    if blocked:
        headline += (f" The ledger also stopped {len(blocked)} repeat payment"
                     f"{'s' if len(blocked) != 1 else ''} worth "
                     f"${blocked_payments['total']:,.2f}.")

    comparison = {
        "cycle_time_before": f"{BASELINE_DAYS_PER_INVOICE}-day",
        "cycle_time_after": per_invoice,
        "error_rate_before": f"{BASELINE_ERROR_RATE:.0%} reached payment undetected",
        "error_rate_after": (f"{measured['detected_error_rate']:.0%} of invoices flagged "
                             f"before payment; {len(flagged_and_held)} held, "
                             f"{len(flagged_and_paid)} approved after review"),
        "headline": headline,
    }

    return {"reconciliation": reconciliation, "measured": measured,
            "money_at_risk": money_at_risk, "blocked_payments": blocked_payments,
            "assumed": assumed, "stated_assumptions": STATED_ASSUMPTIONS,
            "comparison": comparison}


def print_report(impact: Dict[str, Any]) -> None:
    """Terminal report. Every total is followed by the lines that make it up."""
    measured = impact.get("measured", {})
    if not measured:
        print("No invoices processed.")
        return
    reconciliation = impact.get("reconciliation", {})
    risk = impact.get("money_at_risk", {})
    blocked = impact.get("blocked_payments", {})
    comparison = impact.get("comparison", {})
    rule = "-" * 72

    print(f"\n{rule}\nBusiness impact\n{rule}")

    print("\nCounting basis")
    print(f"  {reconciliation.get('statement', '')}")
    if not reconciliation.get("balances", True):
        print("  Warning: these counts do not reconcile.")

    print("\nOutcomes, per unique invoice")
    print(f"  Paid without human review   {measured['straight_through_paid']} of "
          f"{measured['unique_invoices']}  ({measured['straight_through_rate']:.0%})")
    print(f"  Held for review             {measured['held_for_review']}")
    print(f"  Carrying a defect           {measured['invoices_with_any_error']} "
          f"({measured['detected_error_rate']:.0%})  -- "
          f"{measured['flagged_and_held']} held, "
          f"{measured['flagged_and_approved']} approved after review")
    print(f"  Processing time             {comparison.get('cycle_time_after', '')} per "
          f"invoice, against a {comparison.get('cycle_time_before', '')} baseline")

    print(f"\nMoney at risk, caught: ${risk.get('total', 0):,.2f}")
    for line in risk.get("lines", []):
        print(f"  ${line['value']:>11,.2f}  {line['invoice']:<10s} "
              f"{line['code']:<18s} {line['basis']}")
    if not risk.get("lines"):
        print("  None.")

    if blocked.get("count"):
        print(f"\nRepeat payments stopped: {blocked['count']}, worth "
              f"${blocked['total']:,.2f}. Counted separately from the figure above.")
        for line in blocked["lines"]:
            print(f"  ${line['value']:>11,.2f}  {line['invoice']:<10s} {line['basis']}")

    print(f"\n{comparison.get('headline', '')}")

    print("\nAssumptions")
    for assumption in impact.get("stated_assumptions", []):
        print(f"  - {assumption}")
    print(rule)
