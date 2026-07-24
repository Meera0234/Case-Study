# Invoice Processing Agent

Automates supplier invoice handling end to end — **Ingestion → Validation → Approval → Payment** — as a LangGraph state machine over one shared state object.

Run against the 20 sample files, it clears 7 invoices for payment with no human involvement and holds the other 10, holding 10 of the 11 that carry a defect before any money moves, and blocking a further **$12,975** of second payment attempts. The held invoices include **$1,940 of billing that reads as correct on the page**. Every figure in the impact report is itemised by invoice and reconciles to the file count.

---

## Run it

```bash
pip install -r requirements.txt

# one invoice, as specified
python main.py --invoice_path=data/invoices/invoice_1002.txt

# every invoice, then build the review dashboard
python run_batch.py
open logs/invoice-review.html
```

**No API key is needed.** With no key, every LLM step falls back to a deterministic implementation and the whole pipeline still runs, offline, with the same routing decisions. A key adds real reasoning to extraction and approval:

```bash
export GROQ_API_KEY=gsk_...                              # default
# or
export LLM_PROVIDER=gemini && export GEMINI_API_KEY=AIza...
```

`inventory.db` is created and seeded automatically on first run. There is no setup step.

---

## The one decision everything else follows from

**The model reads and explains. Code counts and decides.**

| Job | Owner | Why |
|---|---|---|
| Reading messy text, typos, OCR artifacts | LLM | This is what it is genuinely good at |
| Arithmetic, stock lookups, thresholds | Code | Deterministic, testable, auditable |
| Writing the reasoning for an audit log | LLM | Good at it, and low-risk output |
| Whether money moves | Code | Irreversible — never delegated |

The consequence is worth stating plainly: **an invoice routes identically with or without an API key.** Only the quality of the extraction and the prose of the reasoning change. That is what makes the outcome defensible to a finance team — the AI never decides who gets paid.

---

## Pipeline

| Stage | File | What it does |
|---|---|---|
| **Ingestion** | `agents/ingestion.py` | Dispatches on file type (`.pdf` via pdfplumber, plus `.txt`, `.csv`, `.json`, `.xml`). Free text goes to the LLM with **one self-correcting retry** carrying the actual parse failure; structured formats are parsed exactly by code. Never raises — ambiguity lands in `extraction_notes`. |
| **Validation** | `agents/validation.py` | Pure SQLite and arithmetic. Stock, totals, duplicates, fraud signals. No LLM influences pass/fail. |
| **Approval** | `agents/approval.py` | Deterministic scrutiny rule, then **draft → critique → re-draft**: one model decides, a second audits it and can send it back. Capped at two rounds. Critical failures skip this stage entirely. |
| **Payment** | `agents/payment.py` | Pays the **recomputed** total, never the stated one. Checks a ledger first, so a re-run cannot pay twice. |

```
ingest → validate ─┬─(critical)──────────────────────→ reject_log → END
                   └─(clean)→ approve_draft → approve_critique ─┬─(revise)──→ approve_draft
                                                                ├─(approved)→ pay → END
                                                                └─(rejected)→ reject_log → END
```

---

## What it catches that a person would not

Six of the sample invoices carry defects that survive a careful human read, because the wrong number looks exactly like a right one.

| Invoice | Defect | How it is caught |
|---|---|---|
| **INV-1013** | Total overstated by **$50** | Every line item is individually within stock and the total looks plausible. Recomputing from line items exposes it. |
| **INV-1013** | 22 WidgetA requested, 15 in stock | Listed across three separate lines of 15, 5 and 2. **Each line passes on its own** — quantities are summed per item before comparison. |
| **INV-1004 / _revised** | Same invoice number, $1,890 then $5,940 | A ledger records every invoice seen. Same number with different figures is held for reconciliation; the same document re-read is not. |
| **INV-1008, INV-1012** | $9,900 and $9,975 | Both sit just under the $10,000 approval limit. Shaving an invoice below a threshold is a known fraud pattern and invisible per-invoice. |
| **INV-1012** | `2O26`, `$3,500.O0` | Capital O read as zero. Repaired only inside otherwise-numeric tokens, so words are untouched. |
| **INV-1014** | Denominated in EUR | Held rather than paid at face value. |

---

## The review queue

`run_batch.py` writes `logs/invoice-review.html` — a single file, no external assets, opens from disk.

It is a **work queue, not a chart**. The user is an accounts-payable clerk, and what they need is not throughput statistics but three things: what needs me, why, and what do I do about it. So the page sorts by urgency, states every finding in plain English, and attaches a recommended action to each one — *"Check the receiving log. If the goods arrived, inventory is stale."* The original document sits beside the extracted fields so a number can be verified in two seconds without opening the source file.

---

## Where `inventory.db` comes from

`db/setup_db.py` creates it and seeds `WidgetA(15)`, `WidgetB(10)`, `GadgetX(5)`, `FakeItem(0)`. It is idempotent — the starter snippet in the brief uses a plain `INSERT` and crashes the second time you run it; this uses `INSERT OR REPLACE`. `main.py` calls it automatically when the file is absent.

`db/ledger.py` adds two more tables in the same file: every invoice seen (for duplicate detection) and every invoice paid (so payment cannot happen twice).

---

## One shared state object

A single `InvoiceState` (see `state.py`) flows through every node. Each node is `(state) -> partial_update`; LangGraph merges the result. No agent holds private state, nothing passes between stages via files. By the end that object holds everything extracted, flagged, decided and paid — it **is** the audit trail, printed to the terminal and written to `logs/run_<timestamp>.json`.

Four fields go beyond the minimum, each earning its place by catching a real defect: `unit_price` (without it, no total can be reconciled), `invoice_number` (the duplicate key), `subtotal`/`tax_amount`/`other_charges` (separates "this total is wrong" from "this total includes shipping" — without it, every invoice with a delivery charge raises a false alarm), and `currency`.

---

## Why these libraries

| Library | Why | Why not something else |
|---|---|---|
| **LangGraph** | The approval loop is a genuine cycle — critique can send a decision back for redrafting. Expressing that as a graph with a recursion limit means the loop cannot spin. | A linear function chain cannot express the cycle without hand-rolled loop guards. |
| **LangChain Core** | `ChatPromptTemplate` keeps all three prompts declared once in `llm_client.py` with named variables. When a client says "the approval reasoning is too lenient," there is one file to open. | f-strings scattered through agent files means prompt changes touch four files. |
| **Pydantic** | Validates LLM JSON at the boundary. A malformed response is caught before it can write a null vendor into a payment record. | Trusting `json.loads` means bad data propagates silently. |
| **pdfplumber** | Reads the text layer directly. | PyMuPDF is faster but heavier; neither reads scans, which are out of scope. |

Both LangChain and Pydantic are **optional imports** — if absent, equivalent local code takes over. The no-key path depends on nothing but the standard library and pdfplumber, which is what lets this run on a locked-down machine.

---

## Failsafes and guardrails

Every one of these is tested by `python diagnostic.py`, which attacks them rather than asserting them.

**If an agent fails.** Ingestion never raises — a missing, empty, corrupt, or truncated file returns a well-formed empty result with the reason in `extraction_notes`, which validation turns into a critical finding. A crash anywhere in the graph is caught in `main.run_invoice`, recorded in `state["error"]`, and the invoice is *not* paid. Nothing in the pipeline can fail open.

**If the LLM fails or lies.** No key, no SDK, a timeout, a rate limit, or unparseable output all return `None` from `llm_client`, and the caller takes its deterministic path — a full LLM outage still completes the pipeline end to end. Output that *is* parseable but wrong-shaped is rejected by `schemas.py` before it reaches shared state. And a model that returns `"approved"` on an invoice with critical findings is overridden by `approval.allowed_outcomes` and the override is logged.

**Payment, the one irreversible step, is gated three times over.**

1. The graph routes critical findings to rejection before approval runs.
2. `allowed_outcomes` prevents an approved decision existing at all.
3. `payment.payment_blockers` re-checks independently, immediately before the call.

The third exists because the first two are decisions made elsewhere. If a future edit reorders the graph, that function is what stops the money. It refuses on: any critical finding, a decision that isn't `approved`, no identified payee, a non-positive amount, or a non-USD currency.

**Money cannot move twice.** `db/ledger.py` records every payment, keyed on invoice number, checked immediately before paying. Re-running the same invoice blocks rather than double-paying.

**Loops cannot spin.** The critique cycle is capped at `MAX_CRITIQUE_ROUNDS` inside the node, and the graph is compiled with `recursion_limit=25` as a second bound.

---

## Stated assumptions

Where the system chose a policy rather than measured a fact, the policy is published in the impact report rather than left to be inferred:

- **Revised invoices are held, not auto-superseded.** `invoice_1004_revised.json` says *"Revised invoice - additional items added per PO amendment"* — an ordinary business event, not fraud. But an attacker re-submitting at a higher value looks identical on the page, and the document alone cannot tell you which it is. Both go to a human. The flag is `INVOICE_REVISION`, deliberately not named as a duplicate or a fraud.
- **Non-USD invoices are held, not converted.** No exchange rate offline; inventing one is worse than pausing.
- **Every rate is per unique invoice, not per file.** Re-reads of an already-seen document are reported separately, and the counting basis is printed in words so the arithmetic can be checked.
- **An understated total is valued at zero.** It costs the supplier, not the client, so it is not claimed as a saving.
- **A revision is valued at its overlap, not its face value.** Paying a $1,890 invoice and then a $5,940 revision costs $1,890 — the other $4,050 buys goods genuinely ordered. Claiming face value would overstate the catch by 3×.

---

## Scope cuts

Named deliberately, because what was left out is as much a decision as what went in.

- **Text-layer PDFs only.** Scanned invoices need real OCR — a separate project, and the honest answer is that this system reports them as unreadable rather than guessing.
- **No vendor master data.** The vendor name is taken as extracted, not verified against a supplier list. This is the single biggest remaining fraud gap.
- **No FX rates.** Non-USD invoices are held for a human rather than converted at an invented rate.
- **Stock is checked, not reserved.** Nothing is decremented, so results do not depend on processing order.
- **One approval tier.** Real finance has several, keyed to amount and category.

---

## Verification

`python run_batch.py` processes all 20 files. Expected outcomes, including every scenario in the brief:

| Invoice | Expected | Reason |
|---|---|---|
| INV-1001, 1004, 1006, 1010, 1011, 1015 | paid | clean |
| INV-1002 | held | 20 GadgetX, 5 in stock |
| INV-1003 | held | FakeItem zero stock, urgency wording, unparseable due date |
| INV-1005 | held | 8 GadgetX, 5 in stock |
| INV-1007 | held | total off by $110, stock short |
| INV-1008 | held | unknown items, sits under threshold |
| INV-1009 | held | negative quantity, no vendor |
| INV-1012 | paid, flagged | sits $25 under threshold |
| INV-1013 | held | total off by $50, three items over stock in aggregate |
| INV-1014 | held | EUR |
| INV-1016 | held | WidgetC unknown |
| INV-1004_revised | held | duplicate of a paid invoice at a higher amount |

---

## Known limitations

- The `$10,000` scrutiny branch is exercised by INV-1012 but every invoice *above* the threshold in this dataset also has a critical failure, so it routes to rejection before approval reasoning runs. A clean high-value invoice would be needed to demo that path fully.
- The critique loop only revises when a live LLM is connected. Without a key, the draft is accepted unchanged — the loop is present and capped, but silent.
- Manual-review minutes in the impact report is an estimate, labelled as such. Replace it with the client's own time study before quoting ROI.
