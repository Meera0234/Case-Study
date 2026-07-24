"""Approval Agent -- deterministic scrutiny rule plus an LLM reflection loop.

Reads:  extracted, validation_issues
Writes: approval

The draft/critique cycle is the self-correction requirement: one model drafts a
decision, a second reads it as a stricter auditor and can send it back. Hard
capped at MAX_CRITIQUE_ROUNDS -- after the cap the last draft is force-finalised
rather than looping.

Routing helpers live here, next to the state they inspect, so main.py stays
pure wiring.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import llm_client
from config import MAX_CRITIQUE_ROUNDS, SCRUTINY_THRESHOLD
from state import ApprovalDecision, InvoiceState, ValidationIssue

logger = logging.getLogger(__name__)

CRITIQUE_LOG_PREFIX = "[critique]"


def apply_scrutiny_rule(amount: Optional[float]) -> bool:
    """Deterministic: does this invoice need VP-level scrutiny?"""
    return amount is not None and amount > SCRUTINY_THRESHOLD


def build_findings(state: InvoiceState) -> str:
    """The brief handed to both the drafting and critiquing model."""
    extracted = state.get("extracted") or {}
    issues: List[ValidationIssue] = state.get("validation_issues", [])
    lines = [
        f"Invoice: {extracted.get('invoice_number')} from {extracted.get('vendor')}",
        f"Stated total: {extracted.get('amount')} {extracted.get('currency', 'USD')}",
        f"Recomputed from line items: {state.get('computed_total')}",
        f"Due date: {extracted.get('due_date')}",
        "Items: " + (", ".join(f"{i['name']} x{i['quantity']}"
                               for i in extracted.get("items", [])) or "none"),
        f"Validation status: {state.get('validation_status')}",
        f"Requires scrutiny: {apply_scrutiny_rule(extracted.get('amount'))} "
        f"(threshold ${SCRUTINY_THRESHOLD:,})",
        "Findings:",
    ]
    notable = [i for i in issues if i["severity"] != "info"]
    lines += [f"  [{i['severity']}] {i['code']}: {i['message']}" for i in notable] or \
             ["  none beyond routine checks"]
    notes = extracted.get("extraction_notes", [])
    if notes:
        lines.append("Extraction notes: " + "; ".join(notes))
    return "\n".join(lines)


def _deterministic_draft(state: InvoiceState, requires_scrutiny: bool) -> ApprovalDecision:
    """Rule-based decision used when no LLM is available.

    Critical failures never reach this agent -- they route straight to rejection
    -- so only warnings and the scrutiny threshold are in play here.
    """
    issues: List[ValidationIssue] = state.get("validation_issues", [])
    warnings = [i for i in issues if i["severity"] == "warning"]

    if requires_scrutiny and warnings:
        decision = "rejected"
        reasoning = (
            f"Invoice exceeds the ${SCRUTINY_THRESHOLD:,} scrutiny threshold and carries "
            f"{len(warnings)} unresolved warning(s): "
            + "; ".join(i["message"] for i in warnings)
            + ". Withholding approval for human review."
        )
    elif requires_scrutiny:
        decision = "approved"
        reasoning = (
            f"Invoice exceeds the ${SCRUTINY_THRESHOLD:,} scrutiny threshold, so it was "
            "reviewed against inventory and totals. All line items are in catalogue with "
            "sufficient stock and the stated total reconciles to the line items."
        )
    elif warnings:
        decision = "approved"
        reasoning = (
            "Inventory checks passed and the total reconciles. Non-blocking warnings noted: "
            + "; ".join(i["message"] for i in warnings)
            + ". These do not affect whether the invoice is payable."
        )
    else:
        decision = "approved"
        reasoning = ("All line items are in catalogue with sufficient stock, the stated "
                     "total reconciles to the line items, and no warnings were raised.")

    return {"decision": decision, "reasoning": reasoning,
            "critique_rounds": 0, "requires_scrutiny": requires_scrutiny}


def allowed_outcomes(state: InvoiceState) -> set:
    """What decisions are permissible given the findings.

    The LLM may narrow this set, never widen it. An invoice with a critical
    finding cannot be approved no matter how persuasive the model's reasoning
    is -- reasoning is advisory, the rule is not.
    """
    issues: List[ValidationIssue] = state.get("validation_issues", [])
    if any(i["severity"] == "critical" for i in issues):
        return {"rejected"}
    return {"approved", "rejected"}


def draft_decision(state: InvoiceState) -> Dict[str, Any]:
    """LangGraph node `approve_draft`. LLM call 2, with deterministic fallback."""
    extracted = state.get("extracted") or {}
    log = list(state.get("log", []))
    prior: ApprovalDecision = state.get("approval") or {}
    rounds = prior.get("critique_rounds", 0)
    requires_scrutiny = apply_scrutiny_rule(extracted.get("amount"))
    permitted = allowed_outcomes(state)

    # On a re-draft, hand the auditor's objection back so the second attempt has
    # something concrete to answer rather than simply rephrasing.
    last_critique = next((entry.removeprefix(CRITIQUE_LOG_PREFIX).strip()
                          for entry in reversed(log)
                          if entry.startswith(CRITIQUE_LOG_PREFIX)), None)

    response = llm_client.draft_decision(build_findings(state), critique=last_critique)
    if response:
        approval: ApprovalDecision = {
            "decision": response["decision"],
            "reasoning": response["reasoning"].strip() or "No reasoning returned.",
            "critique_rounds": rounds,
            "requires_scrutiny": requires_scrutiny,
        }
        log.append(f"[approve_draft] LLM decision={approval['decision']} (round {rounds})")
    else:
        approval = _deterministic_draft(state, requires_scrutiny)
        approval["critique_rounds"] = rounds
        log.append(f"[approve_draft] rule-based decision={approval['decision']} "
                   f"(round {rounds})")

    if approval["decision"] not in permitted:
        forced = _deterministic_draft(state, requires_scrutiny)
        forced["decision"] = "rejected"
        forced["critique_rounds"] = rounds
        log.append(f"[approve_draft] GATE OVERRIDE: '{approval['decision']}' is not "
                   f"permitted alongside critical findings; forced to 'rejected'")
        approval = forced

    return {"approval": approval, "stage": "approval_drafted", "log": log}


def critique_decision(state: InvoiceState) -> Dict[str, Any]:
    """LangGraph node `approve_critique`. LLM call 3, a stricter second reader.

    Setting decision to 'pending' is how this node requests another draft; the
    router turns that into a loop-back. The round cap is enforced here so the
    graph cannot spin.
    """
    approval: ApprovalDecision = dict(state.get("approval") or {})
    log = list(state.get("log", []))
    rounds = approval.get("critique_rounds", 0)

    if rounds >= MAX_CRITIQUE_ROUNDS:
        log.append(f"[approve_critique] round cap ({MAX_CRITIQUE_ROUNDS}) reached; "
                   f"finalising on decision={approval.get('decision')}")
        return {"approval": approval, "stage": "approval_final", "log": log}

    response = llm_client.critique_decision(
        build_findings(state), approval.get("decision", ""), approval.get("reasoning", ""))

    if not response:
        # No key, or malformed output: accept the draft rather than block.
        log.append("[approve_critique] no usable critique; confirming draft")
        return {"approval": approval, "stage": "approval_final", "log": log}

    critique_text = response["critique"].strip()
    if response["verdict"] == "confirm":
        log.append(f"[approve_critique] confirmed: {critique_text}")
        return {"approval": approval, "stage": "approval_final", "log": log}

    approval["critique_rounds"] = rounds + 1
    approval["decision"] = "pending"
    log.append(f"{CRITIQUE_LOG_PREFIX} {critique_text}")
    log.append(f"[approve_critique] revision requested (round {rounds + 1})")
    return {"approval": approval, "stage": "approval_revising", "log": log}


# --- routing helpers (called by main.py's conditional edges) ----------------

def route_after_validation(state: InvoiceState) -> str:
    """Skip approval entirely on a critical failure.

    There is no point spending reasoning on a phantom item or a negative
    quantity -- the answer is already no, and running the loop anyway would
    invite the model to talk itself into a yes.
    """
    issues: List[ValidationIssue] = state.get("validation_issues", [])
    return "reject_log" if any(i["severity"] == "critical" for i in issues) else "approve_draft"


def route_after_critique(state: InvoiceState) -> str:
    approval: ApprovalDecision = state.get("approval") or {}
    decision = approval.get("decision")
    rounds = approval.get("critique_rounds", 0)
    if decision == "pending" and rounds < MAX_CRITIQUE_ROUNDS:
        return "approve_draft"
    return "pay" if decision == "approved" else "reject_log"
