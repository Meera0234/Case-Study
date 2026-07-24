"""Dashboard -- a self-contained HTML review queue.

Design note, because this is a deliberate choice rather than a default:

The user of this system is an accounts-payable clerk, not an engineer. What they
need is not a chart of throughput -- it is a **work queue**: what needs me, why,
and what do I do about it. So the page is sorted by urgency, states every finding
in plain English, and attaches a recommended next action to each one. The raw
document sits beside the extracted fields so a clerk can verify a number in two
seconds without opening the original file.

Output is one HTML file with no external assets, so it opens from disk, survives
being emailed, and can be committed to the repo as evidence of a run.
"""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from config import REVIEW_PAGE_PATH, SCRUTINY_THRESHOLD
from state import ISSUE_CODES

# What a clerk should actually do about each finding. This is the difference
# between a system that reports problems and one that resolves them.
RECOMMENDED_ACTION = {
    "UNKNOWN_ITEM": "Check whether the item exists under a different name. If it is genuinely new, add it to the catalogue before paying.",
    "OUT_OF_STOCK": "Verify this item was really ordered and received. Billing for an item with zero stock is a common phantom-invoice pattern.",
    "INSUFFICIENT_STOCK": "Check the receiving log. If the goods arrived, inventory is stale; if not, query the quantity with the supplier.",
    "INVALID_QUANTITY": "Return to the supplier for a corrected invoice. Do not adjust the quantity yourself.",
    "AMOUNT_MISMATCH": "Pay the recomputed amount, or request a corrected invoice. Do not pay the stated total.",
    "INVOICE_REVISION": "Compare against the earlier copy. If this supersedes it, cancel the original before paying; if it is a re-submission, reject it.",
    "REPROCESSED": "No action. The same document was read twice, most likely in two formats.",
    "THRESHOLD_HUGGING": f"Confirm this was not split from a larger order to stay under the ${SCRUTINY_THRESHOLD:,} approval limit.",
    "NON_USD_CURRENCY": "Apply today's exchange rate manually and confirm the payable amount before releasing payment.",
    "SUSPICIOUS_LANGUAGE": "Verify the supplier using a contact you already hold, never the details printed on the invoice.",
    "MISSING_VENDOR": "Request a complete invoice. Never pay without an identified payee.",
    "MISSING_AMOUNT": "Request a complete invoice showing the total due.",
    "MISSING_DUE_DATE": "Confirm payment terms with the supplier so the payment can be scheduled.",
    "NO_LINE_ITEMS": "Open the original document. It may be a scan, which this system does not read.",
    "ITEM_OK": "",
}

STATUS_STYLE = {
    "success": ("paid", "Paid"),
    "blocked_duplicate_payment": ("blocked", "Blocked"),
    "rejected": ("held", "Needs review"),
}

CSS = """
:root{
  color-scheme: light;
  --bg:#f6f5f1; --panel:#fffefb; --ink:#1a1a17; --muted:#6b6a63; --line:#e0ded5;
  --accent:#0f5c58; --crit:#a02020; --warn:#8a5a00; --ok:#2b6a3f;
  --crit-bg:#fbeaea; --warn-bg:#fbf3e2; --ok-bg:#eaf3ec;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:40px 24px 80px}
h1{font-size:26px;letter-spacing:-.02em;margin:0 0 4px}
.sub{color:var(--muted);font-size:14px;margin:0 0 32px}
.basisnote{font-size:12.5px;color:var(--muted);display:inline-block;margin-top:4px}
.num{font-variant-numeric:tabular-nums;font-feature-settings:"tnum"}

.impact{background:var(--panel);border:1px solid var(--line);border-radius:10px;
  padding:24px 26px;margin-bottom:14px}
.impact h2{font-size:12px;letter-spacing:.09em;text-transform:uppercase;
  color:var(--muted);margin:0 0 18px;font-weight:600}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:22px}
.stat b{display:block;font-size:27px;letter-spacing:-.02em;line-height:1.15}
.stat span{display:block;font-size:12.5px;color:var(--muted);margin-top:3px}
.headline{margin-top:22px;padding-top:18px;border-top:1px solid var(--line);
  font-size:14.5px;color:#333}
.caveat{font-size:12.5px;color:var(--muted);margin:0 0 30px;padding-left:2px}
.basis{margin-top:16px;padding-top:14px;border-top:1px solid var(--line);
  font-size:13px;color:var(--muted)}
.prov{margin-top:14px;font-size:13px;border:1px solid var(--line);border-radius:7px;
  padding:10px 14px;background:#faf9f5}
.prov summary{cursor:pointer;font-weight:600;color:var(--ink);font-size:13px}
.prov table{margin-top:6px}
.prov td.num{width:120px;text-align:right;font-weight:600;vertical-align:top;
  padding-right:16px;white-space:nowrap}
.prov td{padding:7px 0}
.prov .note{font-size:12.5px;color:var(--muted);text-transform:none;
  letter-spacing:0;font-weight:400;margin:-4px 0 8px}
.assum{margin:6px 0 0;padding-left:18px;color:#4a4a44}
.assum li{margin-bottom:5px}

.tabs{display:flex;gap:7px;margin-bottom:18px;flex-wrap:wrap}
.tab{background:var(--panel);border:1px solid var(--line);border-radius:999px;
  padding:7px 15px;font-size:13.5px;cursor:pointer;color:var(--muted)}
.tab:hover{border-color:#c9c6ba}
.tab.on{background:var(--accent);border-color:var(--accent);color:#fff;font-weight:500}

.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;
  margin-bottom:11px;overflow:hidden}
.card.crit{border-left:3px solid var(--crit)}
.card.warnrow{border-left:3px solid var(--warn)}
.card.okrow{border-left:3px solid var(--ok)}
.head{display:flex;align-items:center;gap:16px;padding:15px 20px;cursor:pointer}
.head:hover{background:#faf9f5}
.id{font-weight:600;min-width:96px;letter-spacing:-.01em}
.vend{flex:1;color:var(--muted);font-size:14px;overflow:hidden;
  text-overflow:ellipsis;white-space:nowrap}
.amt{font-variant-numeric:tabular-nums;font-size:15px;min-width:112px;text-align:right}
.amt s{color:var(--muted);font-size:12.5px;display:block;text-decoration:line-through}
.pill{font-size:11.5px;font-weight:600;padding:4px 11px;border-radius:999px;
  min-width:104px;text-align:center;letter-spacing:.01em}
.pill.paid{background:var(--ok-bg);color:var(--ok)}
.pill.held{background:var(--crit-bg);color:var(--crit)}
.pill.blocked{background:var(--warn-bg);color:var(--warn)}
.chev{color:var(--muted);font-size:12px;width:12px}

.body{display:none;padding:4px 20px 22px;border-top:1px solid var(--line)}
.card.open .body{display:block}
.card.open .chev{transform:rotate(90deg)}
.chev{display:inline-block;transition:transform .15s}

.sec{font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);
  font-weight:600;margin:20px 0 9px}
.issue{border-radius:7px;padding:11px 14px;margin-bottom:7px;font-size:14px}
.issue.critical{background:var(--crit-bg)}
.issue.warning{background:var(--warn-bg)}
.issue.info{background:#f1f0ea}
.issue .t{font-weight:600;margin-bottom:2px}
.issue .m{color:#3a3a35}
.issue .a{margin-top:7px;padding-top:7px;border-top:1px solid rgba(0,0,0,.09);
  font-size:13px;color:#4a4a44}
.issue .a b{font-weight:600}

.cols{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media(max-width:760px){.cols{grid-template-columns:1fr}}
pre{background:#f1f0ea;border:1px solid var(--line);border-radius:7px;padding:13px;
  font:12px/1.5 ui-monospace,"SF Mono",Menlo,monospace;overflow:auto;max-height:270px;
  margin:0;white-space:pre-wrap;word-break:break-word}
table{width:100%;border-collapse:collapse;font-size:13.5px}
table.items th{font-size:11px;letter-spacing:.06em;text-transform:uppercase;
  color:var(--muted);font-weight:600;text-align:left;padding-bottom:4px}
table.items th.num{text-align:right}
td.qty{width:52px;color:var(--muted)}
td.total{font-weight:600}
.items td.num{padding-left:14px}
td{padding:5px 0;border-bottom:1px solid var(--line);vertical-align:top}
td:first-child{color:var(--muted);width:38%}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.reason{background:#f1f0ea;border-radius:7px;padding:13px 15px;font-size:14px}
.meta{font-size:12.5px;color:var(--muted);margin-top:7px}
.trail{font:12px/1.65 ui-monospace,"SF Mono",Menlo,monospace;color:var(--muted)}
.empty{text-align:center;color:var(--muted);padding:50px;font-size:14px}
footer{margin-top:36px;font-size:12.5px;color:var(--muted);text-align:center}
"""

JS = """
function tog(el){el.parentElement.classList.toggle('open')}
function filt(btn,key){
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('on'));
  btn.classList.add('on');
  let shown=0;
  document.querySelectorAll('.card').forEach(c=>{
    const ok = key==='all' || c.dataset.bucket===key;
    c.style.display = ok?'':'none';
    if(ok) shown++;
  });
  document.getElementById('empty').style.display = shown?'none':'block';
}
"""


def _provenance_html(impact: Dict[str, Any]) -> str:
    """Every headline figure broken into the lines that make it up.

    A reader who wants to check the arithmetic can do so without opening a JSON
    file. An unverifiable impact number is worth less than none.
    """
    risk = impact.get("money_at_risk", {})
    blocked = impact.get("blocked_payments", {})
    parts = [f"<div class='sec'>Money at risk: {_money(risk.get('total', 0))}</div>",
             "<table>"]
    for line in risk.get("lines", []):
        parts.append(f"<tr><td class='num'>{_money(line['value'])}</td>"
                     f"<td><b>{html.escape(line['invoice'])}</b> "
                     f"{html.escape(line['code'])}<br>"
                     f"<span class='info'>{html.escape(line['basis'])}</span></td></tr>")
    if not risk.get("lines"):
        parts.append("<tr><td colspan='2'>None.</td></tr>")
    parts.append("</table>")

    if blocked.get("count"):
        parts.append(f"<div class='sec'>Repeat payments stopped: "
                     f"{_money(blocked.get('total', 0))}</div>"
                     "<div class='note'>Counted separately from the figure above.</div>")
        parts.append("<table>")
        for line in blocked["lines"]:
            parts.append(f"<tr><td class='num'>{_money(line['value'])}</td>"
                         f"<td><b>{html.escape(line['invoice'])}</b><br>"
                         f"<span class='info'>{html.escape(line['basis'])}</span></td></tr>")
        parts.append("</table>")

    measured = impact.get("measured", {})
    parts.append("<div class='sec'>Defects, by outcome</div><table>"
                 f"<tr><td class='num'>{measured.get('flagged_and_held', 0)}</td>"
                 "<td>held before any money moved</td></tr>"
                 f"<tr><td class='num'>{measured.get('flagged_and_approved', 0)}</td>"
                 "<td>approved after review<br>"
                 "<span class='info'>a warning does not block payment; the flag stays "
                 "on the record either way</span></td></tr></table>")
    parts.append("<div class='sec'>Assumptions</div><ul class='assum'>")
    for assumption in impact.get("stated_assumptions", []):
        parts.append(f"<li>{html.escape(assumption)}</li>")
    parts.append("</ul>")
    return "".join(parts)


def _bucket(state: Dict[str, Any]) -> str:
    """Which queue this invoice belongs in."""
    status = (state.get("payment_result") or {}).get("status")
    if status == "success":
        return "paid"
    if status == "blocked_duplicate_payment":
        return "blocked"
    return "attention"


def _urgency(state: Dict[str, Any]) -> int:
    """Sort key: money-at-risk first, then other blockers, then clean."""
    codes = {i["code"] for i in state.get("validation_issues", [])}
    if codes & {"DUPLICATE_INVOICE", "AMOUNT_MISMATCH", "SUSPICIOUS_LANGUAGE"}:
        return 0
    if any(i["severity"] == "critical" for i in state.get("validation_issues", [])):
        return 1
    if any(i["severity"] == "warning" for i in state.get("validation_issues", [])):
        return 2
    return 3


def _per_invoice(seconds: Any) -> str:
    """Sub-second timings read as '0.0s', which looks like a bug rather than a
    result."""
    if not isinstance(seconds, (int, float)) or seconds <= 0:
        return "&mdash;"
    return f"{seconds * 1000:.0f}ms" if seconds < 1 else f"{seconds:.1f}s"


def _line_total(item: Dict[str, Any]) -> str:
    """Quantity times unit price, so a reader is not left to guess which column
    is which."""
    price, quantity = item.get("unit_price"), item.get("quantity")
    if price is None or quantity is None:
        return "&mdash;"
    return _money(round(quantity * price, 2))


def _money(value: Any) -> str:
    return f"${value:,.2f}" if isinstance(value, (int, float)) else "&mdash;"


def _issue_html(issue: Dict[str, str]) -> str:
    code, severity = issue["code"], issue["severity"]
    plain = ISSUE_CODES.get(code, code)
    action = RECOMMENDED_ACTION.get(code, "")
    action_html = (f"<div class='a'><b>What to do:</b> {html.escape(action)}</div>"
                   if action else "")
    return (f"<div class='issue {severity}'>"
            f"<div class='t'>{html.escape(plain)}</div>"
            f"<div class='m'>{html.escape(issue['message'])}</div>"
            f"{action_html}</div>")


def _card_html(state: Dict[str, Any]) -> str:
    extracted = state.get("extracted") or {}
    approval = state.get("approval") or {}
    payment = state.get("payment_result") or {}
    issues = state.get("validation_issues") or []
    notable = [i for i in issues if i["severity"] != "info"]

    style_key, label = STATUS_STYLE.get(payment.get("status", ""), ("held", "Needs review"))
    bucket = _bucket(state)
    row_class = {0: "crit", 1: "crit", 2: "warnrow", 3: "okrow"}[_urgency(state)]

    stated, computed = extracted.get("amount"), state.get("computed_total")
    disagree = (isinstance(stated, (int, float)) and isinstance(computed, (int, float))
                and abs(stated - computed) > 0.01)
    amount_html = (f"<s>{_money(stated)}</s>{_money(computed)}" if disagree
                   else _money(stated if stated is not None else computed))

    items_html = "".join(
        f"<tr><td>{html.escape(str(i['name']))}</td>"
        f"<td class='num qty'>{i['quantity']:g}</td>"
        f"<td class='num'>{_money(i.get('unit_price'))}</td>"
        f"<td class='num total'>{_line_total(i)}</td></tr>"
        for i in extracted.get("items", [])) or "<tr><td colspan='4'>No items read</td></tr>"

    notes_html = "".join(f"<div class='meta'>{html.escape(n)}</div>"
                         for n in extracted.get("extraction_notes", []))

    issues_html = "".join(_issue_html(i) for i in notable) or \
        "<div class='issue info'><div class='m'>No problems found. " \
        "Every item was in catalogue with stock available, and the total reconciled.</div></div>"

    raw = (extracted.get("raw_text") or "")[:1400]
    trail = "\n".join(state.get("log", []))

    return f"""
<div class="card {row_class}" data-bucket="{bucket}">
  <div class="head" onclick="tog(this)">
    <span class="chev">&#9656;</span>
    <span class="id">{html.escape(str(extracted.get('invoice_number') or '&mdash;'))}</span>
    <span class="vend">{html.escape(str(extracted.get('vendor') or 'Unidentified vendor'))}</span>
    <span class="amt">{amount_html}</span>
    <span class="pill {style_key}">{label}</span>
  </div>
  <div class="body">
    <div class="sec">Findings</div>
    {issues_html}
    <div class="sec">Decision</div>
    <div class="reason">{html.escape(str(approval.get('reasoning') or 'No decision recorded.'))}</div>
    <div class="meta">Critique rounds: {approval.get('critique_rounds', 0)}
      &nbsp;&middot;&nbsp; Extra scrutiny: {'yes' if approval.get('requires_scrutiny') else 'no'}
      &nbsp;&middot;&nbsp; Read by: {html.escape(str(extracted.get('extraction_method', '')))}</div>
    <div class="cols">
      <div>
        <div class="sec">What we read</div>
        <table>
          <tr><td>Invoice</td><td>{html.escape(str(extracted.get('invoice_number') or '&mdash;'))}</td></tr>
          <tr><td>Vendor</td><td>{html.escape(str(extracted.get('vendor') or '&mdash;'))}</td></tr>
          <tr><td>Stated total</td><td class="num">{_money(stated)}</td></tr>
          <tr><td>Recomputed</td><td class="num">{_money(computed)}</td></tr>
          <tr><td>Currency</td><td>{html.escape(str(extracted.get('currency', 'USD')))}</td></tr>
          <tr><td>Due</td><td>{html.escape(str(extracted.get('due_date') or '&mdash;'))}</td></tr>
        </table>
        <div class="sec">Line items</div>
        <table class="items">
          <tr><th>Item</th><th class="num">Qty</th><th class="num">Unit price</th>
              <th class="num">Line total</th></tr>
          {items_html}
        </table>
        {notes_html}
      </div>
      <div>
        <div class="sec">Original document</div>
        <pre>{html.escape(raw)}</pre>
        <div class="sec">Audit trail</div>
        <pre class="trail">{html.escape(trail)}</pre>
      </div>
    </div>
  </div>
</div>"""


def render(results: List[Dict[str, Any]], impact: Dict[str, Any],
           output_path: str = REVIEW_PAGE_PATH) -> Path:
    """Write the dashboard and return its path."""
    measured = impact.get("measured", {})
    comparison = impact.get("comparison", {})

    ordered = sorted(results, key=lambda s: (_urgency(s),
                                             str((s.get("extracted") or {}).get("invoice_number"))))
    counts = {"attention": 0, "paid": 0, "blocked": 0}
    for state in results:
        counts[_bucket(state)] += 1

    cards = "".join(_card_html(s) for s in ordered)

    unique = measured.get("unique_invoices", 0)
    risk = impact.get("money_at_risk", {})
    stats = [
        (str(unique), f"unique invoices, from {measured.get('files_read', 0)} files"),
        (f"{measured.get('straight_through_rate', 0):.0%}",
         f"paid with no human touch ({measured.get('straight_through_paid', 0)} of {unique})"),
        (str(measured.get("invoices_with_any_error", 0)),
         f"defects caught pre-payment (of {unique})"),
        (_money(risk.get("total", 0)), "money at risk, stopped"),
        (_per_invoice(measured.get("seconds_per_invoice", 0)), "per invoice, vs 5 days"),
    ]
    stats_html = "".join(f"<div class='stat'><b class='num'>{v}</b><span>{k}</span></div>"
                         for v, k in stats)

    return _write(output_path, f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Invoice Review - Accounts Payable, {datetime.now():%d %B %Y}</title><style>{CSS}</style></head>
<body><div class="wrap">
  <h1>Invoice review</h1>
  <p class="sub">Accounts payable &nbsp;&middot;&nbsp; {datetime.now():%d %B %Y, %H:%M}
     <br>{sum(counts.values())} files: {counts['attention']} need attention,
     {counts['paid']} paid, {counts['blocked']} blocked
     <br><span class="basisnote">Card counts are per file. The summary below counts
     unique invoices; see the counting basis.</span></p>

  <div class="impact">
    <h2>Run summary</h2>
    <div class="stats">{stats_html}</div>
    <div class="headline">{html.escape(str(comparison.get('headline', '')))}</div>
    <div class="basis"><b>Counting basis.</b> {html.escape(str(impact.get('reconciliation', {}).get('statement', '')))}</div>
    <details class="prov">
      <summary>How these figures are calculated</summary>
      {_provenance_html(impact)}
    </details>
  </div>
  <p class="caveat">The five-day and 30% baselines are the figures given in the brief,
     not results from this run.</p>

  <div class="tabs">
    <button class="tab on" onclick="filt(this,'all')">Everything</button>
    <button class="tab" onclick="filt(this,'attention')">Needs attention ({counts['attention']})</button>
    <button class="tab" onclick="filt(this,'paid')">Paid ({counts['paid']})</button>
    <button class="tab" onclick="filt(this,'blocked')">Blocked ({counts['blocked']})</button>
  </div>

  {cards}
  <div class="empty" id="empty" style="display:none">Nothing in this queue.</div>
  <footer>Select any invoice to see its findings, the decision, and the original document.</footer>
</div><script>{JS}</script></body></html>""")


def _write(output_path: str, markup: str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markup, encoding="utf-8")
    return path
