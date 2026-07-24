"""Single entry point.

    python main.py --invoice_path=data/invoices/invoice_1002.txt

Orchestration and CLI wiring only -- all agent behaviour lives in agents/.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from langgraph.graph import END, StateGraph

import llm_client
from agents import approval, ingestion, payment, validation
from config import INVENTORY_DB_PATH, LOG_DIR
from db import setup_db
from state import InvoiceState

logger = logging.getLogger("invoice-agent")


def build_graph():
    """Wire the six nodes. Compiled once, then invoked per invoice.

    ingest -> validate -+-(critical)-------------------> reject_log -> END
                        `-(clean)-> approve_draft -> approve_critique -+-(revise)-> approve_draft
                                                                      +-(approved)-> pay -> END
                                                                      `-(rejected)-> reject_log -> END
    """
    graph = StateGraph(InvoiceState)

    graph.add_node("ingest", ingestion.ingest)
    graph.add_node("validate", validation.validate)
    graph.add_node("approve_draft", approval.draft_decision)
    graph.add_node("approve_critique", approval.critique_decision)
    graph.add_node("pay", payment.pay)
    graph.add_node("reject_log", payment.reject_log)

    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "validate")
    graph.add_conditional_edges(
        "validate",
        approval.route_after_validation,
        {"reject_log": "reject_log", "approve_draft": "approve_draft"},
    )
    graph.add_edge("approve_draft", "approve_critique")
    graph.add_conditional_edges(
        "approve_critique",
        approval.route_after_critique,
        {"approve_draft": "approve_draft", "pay": "pay", "reject_log": "reject_log"},
    )
    graph.add_edge("pay", END)
    graph.add_edge("reject_log", END)

    return graph.compile()


def run_invoice(app, invoice_path: str) -> Dict[str, Any]:
    """Invoke the compiled graph on one invoice. Never raises."""
    initial: InvoiceState = {"invoice_path": invoice_path, "log": []}
    try:
        return app.invoke(initial, config={"recursion_limit": 25})
    except Exception as exc:            # noqa: BLE001 - a CLI must not show a traceback
        logger.error("Pipeline failed on %s: %s: %s", invoice_path, type(exc).__name__, exc)
        failed = dict(initial)
        failed["error"] = f"{type(exc).__name__}: {exc}"
        return failed


SEVERITY_MARK = {"critical": "[!]", "warning": "[~]", "info": "[ ]"}


def print_summary(final_state: Dict[str, Any]) -> None:
    """Readable terminal report of the final state."""
    extracted = final_state.get("extracted") or {}
    issues = final_state.get("validation_issues") or []
    approval_result = final_state.get("approval") or {}
    payment_result = final_state.get("payment_result") or {}
    rule = "-" * 72

    print(f"\n{rule}\nInvoice processing result\n{rule}")
    print(f"  Source           : {final_state.get('invoice_path')}")
    print(f"  Reasoning engine : {llm_client.active_model()}")

    print("\n  Extraction")
    print(f"    Method    : {extracted.get('extraction_method')}")
    print(f"    Invoice   : {extracted.get('invoice_number')}")
    print(f"    Vendor    : {extracted.get('vendor')}")
    print(f"    Stated    : {extracted.get('amount')} {extracted.get('currency', '')}")
    print(f"    Recomputed: {final_state.get('computed_total')}")
    print(f"    Due date  : {extracted.get('due_date')}")
    for item in extracted.get("items", []):
        price = item.get("unit_price")
        price_text = f" @ {price}" if price is not None else ""
        print(f"      - {item['name']} x{item['quantity']}{price_text}")
    for note in extracted.get("extraction_notes", []):
        print(f"      note: {note}")

    print(f"\n  Validation: {final_state.get('validation_status')}")
    for issue in issues:
        print(f"    {SEVERITY_MARK.get(issue['severity'], '   ')} "
              f"{issue['code']}: {issue['message']}")

    print("\n  Approval")
    print(f"    Decision        : {approval_result.get('decision')}")
    print(f"    Needs scrutiny  : {approval_result.get('requires_scrutiny')}")
    print(f"    Critique rounds : {approval_result.get('critique_rounds')}")
    print(f"    Reasoning       : {approval_result.get('reasoning')}")

    print("\n  Outcome")
    status = payment_result.get("status")
    if status == "success":
        print(f"    Paid {payment_result.get('amount')} to {payment_result.get('vendor')}")
    elif status == "blocked_duplicate_payment":
        print(f"    Blocked. {payment_result.get('reasoning')}")
    else:
        print(f"    Rejected. {payment_result.get('reasoning')}")
    print(rule)


def write_run_log(final_state: Dict[str, Any]) -> Path:
    """Persist the whole final state; it is the audit trail."""
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    path = Path(LOG_DIR) / f"run_{datetime.now():%Y%m%d_%H%M%S_%f}.json"

    serialisable = dict(final_state)
    extracted = serialisable.get("extracted")
    if extracted:                       # raw_text bloats the log without adding signal
        trimmed = dict(extracted)
        trimmed["raw_text"] = trimmed.get("raw_text", "")[:2000]
        serialisable["extracted"] = trimmed

    path.write_text(json.dumps(serialisable, indent=2, default=str), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Process a single invoice end to end.")
    parser.add_argument("--invoice_path", required=True,
                        help="Path to the invoice file (.pdf, .txt, .csv, .json, .xml)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # A fresh clone must run with zero manual setup.
    if setup_db.ensure_db(INVENTORY_DB_PATH):
        logger.info("Created and seeded %s", INVENTORY_DB_PATH)
    if not llm_client.is_available():
        logger.info("No API key found, using the deterministic path")

    final_state = run_invoice(build_graph(), args.invoice_path)

    print_summary(final_state)
    print(f"  Run log: {write_run_log(final_state)}")
    return 1 if final_state.get("error") else 0


if __name__ == "__main__":
    sys.exit(main())
