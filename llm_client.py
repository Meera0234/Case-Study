"""The only file in the project that talks to an LLM.

Two things are centralised here on purpose:

1. **Prompts.** All three live in this file as LangChain ChatPromptTemplates, so
   they are versioned in one place with named variables rather than being
   f-strings scattered through agent code. When a client says "the approval
   reasoning is too lenient", there is exactly one file to open.

2. **Failure.** Every function returns None rather than raising -- no key, no
   SDK, timeout, malformed JSON, all the same to the caller. That contract is
   what lets agent code stay free of `if api_key:` branching, and it is why the
   pipeline runs identically with no network.

LangChain is used for prompt templating and output parsing only. The provider
call itself is the vendor SDK directly, because wrapping a single
system+user call in a chat model class would add indirection without removing
any code.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, Optional

import schemas
from config import (
    GEMINI_MODEL,
    GROQ_MODEL,
    LLM_MAX_RETRIES,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
    SCRUTINY_THRESHOLD,
)

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:                     # .env support is a convenience, not a requirement
    pass

try:
    from langchain_core.prompts import ChatPromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:                     # deterministic path needs no templating
    LANGCHAIN_AVAILABLE = False
    ChatPromptTemplate = None           # type: ignore[assignment]


# --- prompts ---------------------------------------------------------------

_EXTRACTION_SYSTEM = """You extract structured data from messy supplier invoices.

Return ONLY a JSON object. No prose, no markdown fences. Exact shape:
{{"invoice_number": string|null,
  "vendor": string|null,
  "amount": number|null,
  "subtotal": number|null,
  "tax_amount": number|null,
  "other_charges": number|null,
  "currency": string,
  "items": [{{"name": string, "quantity": number, "unit_price": number|null}}],
  "due_date": string|null,
  "notes": [string]}}

Rules:
- Copy values exactly as they appear. Do NOT correct arithmetic -- downstream
  code recomputes totals and compares. Silently fixing a wrong total hides the
  very error we are looking for.
- "amount" is the stated grand total. "subtotal" is before tax.
- "other_charges" is shipping/freight/handling shown outside the subtotal.
- Keep item names as written, including spaces and parentheses.
- If the due date is not a real date (e.g. "yesterday"), return it verbatim.
- Never include subtotal / tax / shipping rows as items.
- Put anything ambiguous or suspicious into "notes"."""

_EXTRACTION_USER = "{document_text}"

_RETRY_USER = """Your previous response could not be used: {failure_reason}

Return ONLY the JSON object described above, nothing else.

{document_text}"""

_DRAFT_SYSTEM = """You are a VP of Finance deciding whether to pay a supplier invoice.

Return ONLY a JSON object: {{"decision": "approved"|"rejected", "reasoning": "2-3 sentences"}}

Guidance:
- Invoices above ${threshold:,} require heightened scrutiny. State explicitly how
  you satisfied yourself, or reject.
- Address every finding below. Do not approve around one without saying why.
- Cite the specific findings. Generic assurances are not reasoning."""

_DRAFT_USER = """{findings}"""

_REDRAFT_USER = """{findings}

An internal auditor rejected your previous reasoning: {critique}

Produce a revised decision that answers that objection directly."""

_CRITIQUE_SYSTEM = """You are an internal auditor reviewing a VP's invoice decision.

Return ONLY a JSON object: {{"verdict": "confirm"|"revise", "critique": "one or two sentences"}}

Answer "revise" only if the reasoning fails to address a specific finding, ignores
the scrutiny threshold, or does not follow from the facts. Otherwise "confirm"."""

_CRITIQUE_USER = """{findings}

Proposed decision: {decision}
Reasoning: {reasoning}"""


_PLACEHOLDER = re.compile(r"(?<!\{)\{([A-Za-z_][A-Za-z0-9_]*)(:[^}]*)?\}(?!\})")


def _fill(template: str, variables: Dict[str, Any]) -> str:
    """Substitute {name} placeholders, then unescape {{ and }}.

    Mirrors LangChain's f-string template semantics, so the same prompt string
    renders identically whether or not LangChain is installed. Literal JSON
    braces in the prompts are written doubled for exactly this reason.
    """
    def replace(match: re.Match) -> str:
        name, spec = match.group(1), match.group(2) or ""
        if name not in variables:
            return match.group(0)
        return format(variables[name], spec[1:]) if spec else str(variables[name])

    filled = _PLACEHOLDER.sub(replace, template)
    return filled.replace("{{", "{").replace("}}", "}")


def _render(system: str, user: str, **variables: Any) -> tuple[str, str]:
    """Fill a prompt pair.

    Uses LangChain's ChatPromptTemplate when available, so prompts are declared
    once with named variables and can be inspected, versioned, or swapped
    without touching agent code. Falls back to an equivalent local fill when
    LangChain isn't installed, so the deterministic path stays dependency-free.
    """
    if LANGCHAIN_AVAILABLE:
        template = ChatPromptTemplate.from_messages([("system", system), ("human", user)])
        messages = template.format_messages(**variables)
        return messages[0].content, messages[1].content
    return _fill(system, variables), _fill(user, variables)


# --- provider plumbing -----------------------------------------------------

def _provider_config() -> Optional[Dict[str, str]]:
    """Resolve the active provider, or None if no usable key is configured."""
    if LLM_PROVIDER == "gemini":
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        return {"name": "gemini", "key": key, "model": GEMINI_MODEL} if key else None
    key = os.environ.get("GROQ_API_KEY")
    return {"name": "groq", "key": key, "model": GROQ_MODEL} if key else None


def is_available() -> bool:
    """True if an LLM call could plausibly succeed."""
    return _provider_config() is not None


def active_model() -> str:
    config = _provider_config()
    return f"{config['name']}:{config['model']}" if config else "none (deterministic fallback)"


def _call_groq(config: Dict[str, str], system: str, user: str) -> Optional[str]:
    try:
        from groq import Groq
    except ImportError:
        logger.debug("groq SDK not installed; using fallback")
        return None
    try:
        client = Groq(api_key=config["key"], timeout=LLM_TIMEOUT_SECONDS)
        response = client.chat.completions.create(
            model=config["model"],
            temperature=LLM_TEMPERATURE,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        return response.choices[0].message.content
    except Exception as exc:            # noqa: BLE001 - a provider fault must not stop the run
        logger.warning("Groq call failed: %s", exc)
        return None


def _call_gemini(config: Dict[str, str], system: str, user: str) -> Optional[str]:
    try:
        import google.generativeai as genai
    except ImportError:
        logger.debug("google-generativeai SDK not installed; using fallback")
        return None
    try:
        genai.configure(api_key=config["key"])
        model = genai.GenerativeModel(config["model"], system_instruction=system)
        response = model.generate_content(
            user,
            generation_config={"temperature": LLM_TEMPERATURE},
            request_options={"timeout": LLM_TIMEOUT_SECONDS},
        )
        return response.text
    except Exception as exc:            # noqa: BLE001
        logger.warning("Gemini call failed: %s", exc)
        return None


def _complete(system: str, user: str) -> Optional[str]:
    config = _provider_config()
    if config is None:
        return None
    caller = _call_gemini if config["name"] == "gemini" else _call_groq
    for attempt in range(LLM_MAX_RETRIES):
        text = caller(config, system, user)
        if text:
            return text
        if attempt + 1 < LLM_MAX_RETRIES:
            time.sleep(1.5 * (attempt + 1))       # brief backoff on transport failure
    return None


def parse_json(text: Optional[str]) -> Optional[Dict[str, Any]]:
    """Tolerant JSON extraction: strips code fences, takes the outermost object."""
    if not text:
        return None
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


# --- the three calls agents make -------------------------------------------

def extract_invoice(document_text: str) -> tuple[Optional[Dict[str, Any]], list[str]]:
    """LLM call 1. One attempt plus one self-correcting retry.

    The retry carries the actual failure reason, so the model has something
    specific to fix rather than a vague "try again".

    Fallback: returns (None, notes) -- ingestion then uses regex extraction.
    """
    notes: list[str] = []
    system, user = _render(_EXTRACTION_SYSTEM, _EXTRACTION_USER, document_text=document_text)
    result = schemas.validate_extraction(parse_json(_complete(system, user)))
    if result is not None:
        return result, notes

    failure = "the response was not valid JSON matching the required shape"
    notes.append(f"LLM extraction retry triggered: {failure}")
    system, user = _render(_EXTRACTION_SYSTEM, _RETRY_USER,
                           failure_reason=failure, document_text=document_text)
    result = schemas.validate_extraction(parse_json(_complete(system, user)))
    if result is not None:
        notes.append("LLM extraction succeeded on retry")
        return result, notes

    notes.append("LLM extraction failed twice; used deterministic extraction")
    return None, notes


def draft_decision(findings: str, critique: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """LLM call 2. Returns {"decision", "reasoning"} or None.

    Fallback: None -- approval.py then applies its deterministic rule.
    """
    if critique:
        system, user = _render(_DRAFT_SYSTEM, _REDRAFT_USER,
                               threshold=SCRUTINY_THRESHOLD, findings=findings,
                               critique=critique)
    else:
        system, user = _render(_DRAFT_SYSTEM, _DRAFT_USER,
                               threshold=SCRUTINY_THRESHOLD, findings=findings)
    return schemas.validate_decision(parse_json(_complete(system, user)))


def critique_decision(findings: str, decision: str,
                      reasoning: str) -> Optional[Dict[str, Any]]:
    """LLM call 3. Returns {"verdict", "critique"} or None.

    Fallback: None -- approval.py then accepts the draft unchanged.
    """
    system, user = _render(_CRITIQUE_SYSTEM, _CRITIQUE_USER,
                           findings=findings, decision=decision, reasoning=reasoning)
    return schemas.validate_critique(parse_json(_complete(system, user)))


def summarise_findings(findings: str) -> Optional[str]:
    """Cosmetic prose summary for the log. Never affects routing."""
    system = ("You summarise invoice validation findings for an audit log. "
              "One short paragraph, plain prose, no bullet points, no preamble.")
    return _complete(system, findings)
