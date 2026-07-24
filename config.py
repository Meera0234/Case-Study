"""Tunable constants. Nothing in agents/ should hardcode these values inline."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

INVENTORY_DB_PATH = str(PROJECT_ROOT / "inventory.db")
INVOICE_DIR = str(PROJECT_ROOT / "data" / "invoices")
LOG_DIR = str(PROJECT_ROOT / "logs")
REVIEW_PAGE_PATH = str(PROJECT_ROOT / "logs" / "invoice-review.html")

# --- business rules --------------------------------------------------------
SCRUTINY_THRESHOLD = 10000        # invoices above this need VP-level scrutiny
THRESHOLD_HUG_BAND = 0.05         # flag amounts within 5% *below* the threshold
MAX_CRITIQUE_ROUNDS = 2           # hard cap on the draft <-> critique loop
MONEY_TOLERANCE = 0.01            # rounding slack when comparing currency

# --- baseline for the impact report ---------------------------------------
# Straight from the brief. Kept here rather than inside impact.py so a client
# can correct their own numbers in one place without touching logic.
BASELINE_ANNUAL_LOSS_USD = 2_000_000
BASELINE_ERROR_RATE = 0.30
BASELINE_DAYS_PER_INVOICE = 5
BASELINE_MINUTES_MANUAL_REVIEW = 22   # assumption: AP clerk time per invoice

# Seed inventory. setup_db.py is the only writer; validation.py only reads.
SEED_INVENTORY = {
    "WidgetA": 15,
    "WidgetB": 10,
    "GadgetX": 5,
    "FakeItem": 0,
}

# --- LLM -------------------------------------------------------------------
# The brief names Grok; this build targets free-tier providers. llm_client.py is
# the only file that touches a provider SDK, so switching costs one file.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq").lower()   # "groq" | "gemini"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

LLM_TIMEOUT_SECONDS = 45
LLM_MAX_RETRIES = 2               # transport retries, not the critique loop
LLM_TEMPERATURE = 0.0             # deterministic reasoning for a clean audit trail
