"""Pydantic models for LLM output.

These exist for one reason: an LLM returns free-form text, and if a malformed
response reaches InvoiceState it corrupts everything downstream. Validating the
shape here means a bad response is caught at the boundary and the caller falls
back to deterministic logic, instead of a null vendor propagating silently into
a payment record.

Pydantic is an optional dependency. If it isn't installed the module falls back
to equivalent hand-rolled checks, so the no-key path needs nothing but stdlib.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field, ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:                     # stdlib-only fallback path
    PYDANTIC_AVAILABLE = False
    BaseModel = object                  # type: ignore[assignment,misc]
    ValidationError = Exception         # type: ignore[assignment,misc]

    def Field(*_args: Any, **_kwargs: Any) -> Any:   # type: ignore[misc]
        return None


if PYDANTIC_AVAILABLE:

    class LineItemModel(BaseModel):
        name: str
        quantity: float
        unit_price: Optional[float] = None

    class ExtractionModel(BaseModel):
        """Shape the extraction prompt asks the model to return."""
        invoice_number: Optional[str] = None
        vendor: Optional[str] = None
        amount: Optional[float] = None
        subtotal: Optional[float] = None
        tax_amount: Optional[float] = None
        other_charges: Optional[float] = None
        currency: str = "USD"
        items: List[LineItemModel] = Field(default_factory=list)
        due_date: Optional[str] = None
        notes: List[str] = Field(default_factory=list)

    class DecisionModel(BaseModel):
        """Shape the approval prompt asks the model to return."""
        decision: str
        reasoning: str = ""

    class CritiqueModel(BaseModel):
        """Shape the critique prompt asks the model to return."""
        verdict: str
        critique: str = ""


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def validate_extraction(payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return a clean extraction dict, or None if the payload is unusable."""
    if not isinstance(payload, dict):
        return None

    if PYDANTIC_AVAILABLE:
        try:
            return ExtractionModel(**payload).model_dump()
        except ValidationError:
            return None

    if not isinstance(payload.get("items"), list):
        return None
    items = []
    for entry in payload["items"]:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or entry.get("item") or "").strip()
        quantity = _as_float(entry.get("quantity"))
        if name and quantity is not None:
            items.append({"name": name, "quantity": quantity,
                          "unit_price": _as_float(entry.get("unit_price"))})
    notes = payload.get("notes")
    return {
        "invoice_number": payload.get("invoice_number"),
        "vendor": payload.get("vendor"),
        "amount": _as_float(payload.get("amount")),
        "subtotal": _as_float(payload.get("subtotal")),
        "tax_amount": _as_float(payload.get("tax_amount")),
        "other_charges": _as_float(payload.get("other_charges")),
        "currency": payload.get("currency") or "USD",
        "items": items,
        "due_date": payload.get("due_date"),
        "notes": [str(n) for n in notes] if isinstance(notes, list) else [],
    }


def validate_decision(payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return a clean decision dict, or None if unusable."""
    if not isinstance(payload, dict):
        return None
    if PYDANTIC_AVAILABLE:
        try:
            model = DecisionModel(**payload)
        except ValidationError:
            return None
        payload = model.model_dump()
    if payload.get("decision") not in ("approved", "rejected"):
        return None
    return {"decision": payload["decision"], "reasoning": str(payload.get("reasoning", ""))}


def validate_critique(payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return a clean critique dict, or None if unusable."""
    if not isinstance(payload, dict):
        return None
    if PYDANTIC_AVAILABLE:
        try:
            model = CritiqueModel(**payload)
        except ValidationError:
            return None
        payload = model.model_dump()
    if payload.get("verdict") not in ("confirm", "revise"):
        return None
    return {"verdict": payload["verdict"], "critique": str(payload.get("critique", ""))}
