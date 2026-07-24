"""Batch runner -- processes every invoice in a folder, then writes the
dashboard and the impact report.

main.py remains the required single-invoice entry point. This is the demo and
review tool: one command that produces the artefact a client actually looks at.

    python run_batch.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import dashboard
import impact
import llm_client
from config import INVENTORY_DB_PATH, INVOICE_DIR, LOG_DIR
from db import ledger, setup_db
from main import build_graph, run_invoice

logger = logging.getLogger("invoice-agent.batch")

READABLE = (".txt", ".json", ".csv", ".xml", ".pdf")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Process every invoice in a folder and build the review dashboard.")
    parser.add_argument("--invoice_dir", default=INVOICE_DIR,
                        help="Folder of invoice files (default: data/invoices)")
    parser.add_argument("--keep_ledger", action="store_true",
                        help="Keep prior run history instead of starting clean")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    setup_db.ensure_db(INVENTORY_DB_PATH)
    if not args.keep_ledger:
        # A demo should start from a known state, or duplicate flags from the
        # previous run bleed into this one.
        ledger.reset()

    files = sorted(p for p in Path(args.invoice_dir).iterdir()
                   if p.suffix.lower() in READABLE)
    if not files:
        print(f"No invoice files found in {args.invoice_dir}")
        return 1

    print(f"Processing {len(files)} files from {args.invoice_dir}")
    print(f"Reasoning engine: {llm_client.active_model()}\n")

    app = build_graph()
    started = time.perf_counter()
    results = []
    for path in files:
        state = run_invoice(app, str(path))
        results.append(state)
        extracted = state.get("extracted") or {}
        decision = (state.get("approval") or {}).get("decision", "-")
        status = (state.get("payment_result") or {}).get("status", "-")
        print(f"  {path.name:28s} {str(extracted.get('invoice_number') or '?'):10s} "
              f"{decision:9s} {status}")
    elapsed = time.perf_counter() - started

    report = impact.compute(results, elapsed)
    impact.print_report(report)

    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOG_DIR, "impact.json").write_text(json.dumps(report, indent=2, default=str),
                                            encoding="utf-8")
    page = dashboard.render(results, report)

    print(f"\nReview page: {page}")
    print(f"Impact data: {Path(LOG_DIR, 'impact.json')}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
