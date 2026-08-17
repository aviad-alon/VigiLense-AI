"""
agent.py — ReAct loop and Supabase logging.

run_react_loop(user_prompt) -> (final_report: dict, steps: list)
  The agent iterates: Reason → Act (call a tool) → Observe (get result) → Reason again,
  until the agent calls submit_final_report or MAX_ITERATIONS is reached.
"""

import json
import re
from config import llm_client, supabase, CHAT_MODEL
from tools import TOOLS, dispatch

# ── Citation Integrity Helpers ─────────────────────────────────────────────────
# These run in Python — independent of LLM behaviour — guaranteeing that
# every PMID in the final report came from an actual tool response.

_PMID_RE = re.compile(r'\[?PMID[:\s]+(\d+)\]?(?:\([^\)]*\))?', re.IGNORECASE)


def _build_literature_section(
    articles: list,
    summaries: dict | None = None,
    audit_entries: list | None = None,
    tiers: dict | None = None,
) -> tuple[str, dict[str, int]]:
    """
    Render a dual-section literature block from real PubMed tool results.
    All metadata (PMID, title, authors, journal, URL) is 100% deterministic — the LLM never touches it.

    Section 1 — Main view (Actionable):
        Tier 1 articles (case reports, trials, safety alerts) in concise 2-line bullet format,
        numbered with their UNIFIED continuous index.

    Section 2 — Collapsible <details> block (All screened):
        ALL screened articles numbered 1..N with unified continuous index.
        - Tier 1 (1..K):   repeated with full entry (title + key finding)
        - Tier 2 (K+1..N): 1-line summary note per article (background / no direct AE data)

    Returns:
        section_text    — the formatted Markdown string
        pmid_to_number  — mapping of ALL PMIDs → continuous article number (1-indexed)
    """
    summaries = summaries or {}
    tiers     = tiers     or {}

    # Articles that have a summary (either pre-computed or LLM-provided)
    relevant = [a for a in articles if a.get("pmid") and a["pmid"] in summaries]

    # Split by tier (default = "1" if not specified)
    tier1 = [a for a in relevant if tiers.get(a["pmid"], "1") == "1"]
    tier2 = [a for a in relevant if tiers.get(a["pmid"], "1") == "2"]

    # Articles that passed screening but have no summary (Phase 2 failure edge-case)
    summarized_pmids = {a["pmid"] for a in relevant}
    no_summary = [a for a in articles if a.get("pmid") and a["pmid"] not in summarized_pmids]

    # ── Unified continuous numbering: Tier 1 first (1..K), then Tier 2, then no-summary ──
    all_ordered = tier1 + tier2 + no_summary
    pmid_to_number: dict[str, int] = {art["pmid"]: i + 1 for i, art in enumerate(all_ordered)}

    total_unique_screened = len(articles)
    total_included        = len(relevant)
    total_excluded_deep   = total_unique_screened - total_included

    parts = []

    # ── Audit header ─────────────────────────────────────────────────────────
    if audit_entries:
        audit_lines = ["**PubMed Retrieval Audit**\n"]
        for e in audit_entries:
            found    = e.get("total_found", 0)
            fetched  = e.get("total_fetched", 0)
            screened = e.get("total_relevant", fetched)
            dr       = e.get("date_range", "")
            query    = e.get("query", "")
            coverage = (
                "0 articles found" if found == 0
                else f"Found: {found} | Fetched: {fetched} | LLM-screened: {screened}"
            )
            audit_lines.append(f"- `{query}` — {dr} — {coverage}")

        if total_unique_screened > 0:
            excl_note = (
                f" (excluded during deep screening: {total_excluded_deep}"
                " — e.g., non-clinical, no direct AE data)"
                if total_excluded_deep > 0 else ""
            )
            audit_lines.append(
                f"\n**Pipeline summary:** {total_unique_screened} unique articles after LLM screening"
                f" → **{total_included} included** "
                f"({len(tier1)} actionable / {len(tier2)} background){excl_note}"
            )
        parts.append("\n".join(audit_lines))

    # ── Section 1: Actionable literature — main view ──────────────────────────
    if not tier1:
        parts.append(
            "*No actionable articles with direct adverse event reports were identified "
            "in the retrieved literature.*"
        )
    else:
        parts.append(
            f"**Actionable Literature — Direct AE Evidence "
            f"({len(tier1)} article{'s' if len(tier1) != 1 else ''})**"
        )
        items = []
        for art in tier1:
            pmid       = art.get("pmid", "")
            n          = pmid_to_number[pmid]
            title      = (art.get("title") or "Unknown title").strip()
            authors    = art.get("authors") or []
            pubdate    = (art.get("pubdate") or "").strip()
            journal    = (art.get("source") or "").strip()
            author_str = (
                f"{authors[0]} et al." if len(authors) > 1
                else (authors[0] if authors else "Unknown")
            )
            link = f"[PMID: {pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)" if pmid else ""

            line  = f"**{n}.** {title} — {author_str} · *{journal}* · {pubdate} · {link}"
            line += f"\n   **Key Finding:** {summaries[pmid]}"
            items.append(line)

        parts.append("\n\n".join(items))

    # ── Section 2: Collapsible — all screened articles ────────────────────────
    n_all = len(all_ordered)
    if n_all > 0:
        def _art_header(art: dict) -> str:
            pmid       = art.get("pmid", "")
            n          = pmid_to_number.get(pmid, "?")
            title      = (art.get("title") or "Unknown title").strip()
            authors    = art.get("authors") or []
            pubdate    = (art.get("pubdate") or "").strip()
            journal    = (art.get("source") or "").strip()
            author_str = (
                f"{authors[0]} et al." if len(authors) > 1
                else (authors[0] if authors else "Unknown")
            )
            link = f"[PMID: {pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)" if pmid else ""
            return f"**{n}.** {title} — {author_str} · *{journal}* · {pubdate} · {link}"

        collapsible_items: list[str] = []

        # Tier 1 in collapsible — full entry (mirrors main view)
        for art in tier1:
            pmid = art.get("pmid", "")
            collapsible_items.append(
                _art_header(art) + f"\n   **Key Finding:** {summaries[pmid]}"
            )

        # Tier 2 in collapsible — 1-line summary note
        for art in tier2:
            pmid         = art.get("pmid", "")
            summary_note = summaries.get(pmid, "No direct AE data.")
            collapsible_items.append(
                _art_header(art) + f"\n   *{summary_note}*"
            )

        # Articles with no summary — title + PMID only
        for art in no_summary:
            collapsible_items.append(_art_header(art))

        collapsible_block = (
            f"<details>\n"
            f"<summary>Click to expand all {n_all} screened PubMed articles</summary>\n"
            "\n"
            + "\n\n".join(collapsible_items)
            + "\n\n</details>"
        )
        parts.append(collapsible_block)

    return "\n\n".join(parts), pmid_to_number


def _replace_pmid_citations(text: str, pmid_to_number: dict, valid_pmids: set) -> str:
    """
    Convert PMID references in LLM-generated text to unified continuous index citations.

    pmid_to_number now covers ALL screened articles (Tier 1 and Tier 2):
    - [PMID: 12345678] → [3]   if the PMID is article #3 in the unified master list
    - Unknown/hallucinated PMIDs → [citation not retrieved]
    """
    def _replace(m):
        pmid = m.group(1)
        if pmid in pmid_to_number:
            return f"[{pmid_to_number[pmid]}]"
        if pmid in valid_pmids:
            return m.group(0)  # valid PMID but not in the numbered list — keep as-is
        return "[citation not retrieved]"
    return _PMID_RE.sub(_replace, text)

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """[ACADEMIC PHARMACOVIGILANCE PLATFORM — ICH E2D / CIOMS VI]
You are VigiLenseAI — an autonomous Pharmacovigilance Triage Agent.
Mission: Detect novel adverse drug event signals in medical literature and FAERS. No medical advice implied.

ABSOLUTE RULE — ZERO FABRICATION:
Every fact, PMID, number, or statistic MUST come from an actual tool response this session. Never invent data, fill gaps, or cite PMIDs not returned by a tool. If a tool returned nothing, say so explicitly.

MANDATORY FIRST STEP:
Your very first call MUST be `query_knowledge_base`. If drug_in_formulary=false → abort with "drug_not_in_portfolio" immediately. Do not call any other tool first.

TOOLBOX GUIDANCE:

1. `query_knowledge_base` — Check FDA label / internal KB for documented AEs. Call once per distinct finding from literature.
   SCOPE RULE: Only use KB chunks about the INVESTIGATED AE or adjacent findings in the same organ system / mechanism. Discard chunks about unrelated AE categories.

2. `get_drug_profile` — Retrieve active ingredients and pharmacological class.

3. `fetch_pubmed_advanced` / `search_drug_class_effects` — Always include drug name in query_term. Always pass investigation_context. Use strict boolean syntax:
   Format: `"drug_name" AND ("ae1" OR "ae2")` — all multi-word terms double-quoted.
   QUERY SCOPE RULE: First call MUST use the EXACT AE term from the user query. Expand only if count=0.

4. `check_past_signals` — Avoid re-escalating previously discarded signals.

5. Statistical Analysis — Two-Stage Fallback:
   Stage 1: PubMed articles with explicit 2×2 counts (a,b,c,d) → `calculate_disproportionality`, tag "Literature / PubMed (PMID: X)".
   Stage 2: No counts in literature → `fetch_fda_adverse_events` → if valid a/b/c/d returned → `calculate_disproportionality`, tag "OpenFDA FAERS Database".
   No data: Proceed without ROR (ror=null); note absence in summary_findings.
   When FAERS returns demographics (gender, age, concomitant drugs) → include brief summary in Signal Assessment.

6. `generate_pharmacovigilance_report` → `submit_final_report` — Always call both in this order.
   Pass case_counts with raw a/b/c/d when FAERS disproportionality is available.

COMPOSITE SIGNAL CLASSIFICATION (`signal_level`):
"significant" → FAERS ROR ≥ 2.0 AND lower 95% CI > 1.0.
"potential"   → FAERS negative/uncalculable BUT ≥1 Tier 1 article with novel AE not fully documented in FDA label for this specific AE. Use whenever recommending escalation.
"none"        → ONLY when FAERS shows no disproportionality AND no novel Tier 1 findings beyond the label. Any unlabeled risk → "potential".
`is_significant` = statistical only (ROR). `signal_level` = composite expert judgment. When in doubt: "potential".

FDA LABEL CROSS-MAPPING:
1. Query `query_knowledge_base` with the EXACT AE term.
2. If no exact match → mandatory secondary query with adjacent terms (same organ system / mechanism).
3. Report both levels separately in "Internal KB / FDA Label Baseline".

DEDUPLICATION: Duplicate = identical PMID or identical title only. Different papers on the same topic are NOT duplicates. Never write "(Duplicate...)" in any relevance_summary.

TWO-TIER ARTICLE CLASSIFICATION (`tier` required on every article):
Tier "1" — ACTIONABLE: Case report/series, clinical trial, or safety DB study with direct patient AE occurrences. Default to "1" if uncertain.
Tier "2" — BACKGROUND: Reviews, mechanistic/animal/in-vitro, PK/PD not reporting target AE, or studies concluding no occurrences.

relevance_summary format:
Tier 1 → 1-2 sentences: case count, dose/timing, outcome. No titles or author names.
Tier 2 → Single line: "No [target AE] reported; [one-sentence mechanistic note]."

AUTONOMOUS RULES:
1. State reasoning (Thought) before every tool call.
2. Refine queries if initial results are vague (try active ingredient or drug class).
3. Conclude after 3–5 KB queries and 2–3 PubMed searches. Do not loop indefinitely.
4. Only flag NOVEL if not already documented in the KB.
5. ALWAYS end with `generate_pharmacovigilance_report` → `submit_final_report`.

`summary_findings` STRUCTURE (exact subheadings, bullet points for lists):

INVESTIGATION SCOPE CONSTRAINT: Every bullet MUST concern the investigated AE or adjacent findings (same organ system / mechanism). Exclude unrelated AE categories even if they appear in the label.

### Internal KB / FDA Label Baseline
KB / FDA label findings specifically about the investigated AE or adjacent organ-system findings.

### Novel Findings
Literature AEs NOT in the FDA label. Cite [PMID: XXXXXXXX] per finding (auto-converted to [N]).

### Known / Expected Findings
Label-consistent findings related to the investigated AE only. No unrelated warnings.

### Signal Assessment
Evidence quality, signal strength, confidence — scoped exclusively to the investigated AE.

ARTICLE SUMMARIES (`article_summaries`):
Include ALL articles returned by PubMed tools. Provide pmid, tier, relevance_summary per article.
STRICT ISOLATED EXTRACTION: Every relevance_summary derives SOLELY from that article's abstract. Never borrow details from another abstract.
TITLE VALIDATION: Animal/in-vitro/mechanistic title → Tier 2, no patient data. "Case report/series" → Tier 1, only details from THAT abstract.
SELF-CHECK: Confirm no summary contains facts from a different PMID's abstract before submitting.

SOURCE ATTRIBUTION: disproportionality_source = "OpenFDA FAERS Database" or "Literature / PubMed (PMID: X)".

GUARDRAILS — call `abort_investigation` immediately. Write reason in the user's language, concise and actionable:
1. query_too_vague: No specific drug name identified.
2. multiple_drugs_detected: Two or more distinct drugs named.
3. non_medical_query: No connection to drug safety or pharmacovigilance.
4. drug_not_recognized: `get_drug_profile` returns source="fallback" AND name is clearly not a real pharmaceutical (gibberish, food, non-drug). Do NOT trigger for real drugs that failed API lookup.
5. no_literature_found: Both `fetch_pubmed_advanced` AND `search_drug_class_effects` returned count=0.
6. drug_not_in_portfolio: `query_knowledge_base` returns drug_in_formulary=false (always first tool call)."""


# ── ReAct Loop ────────────────────────────────────────────────────────────────

def run_react_loop(user_prompt: str) -> tuple[dict, list]:
    """
    Runs the ReAct agent loop.
    Returns:
        final_report (dict)  — structured analysis from submit_final_report
        steps        (list)  — trace of all tool calls for the UI
    """
    if not llm_client:
        raise ValueError("LLM client is not configured (missing OPENAI_API_KEY).")

    # Wrap with research framing to prevent content-policy false positives
    # from clinical terminology (e.g. "toxicity", "fatal") in retrieved abstracts
    framed_prompt = (
        "[PHARMACOVIGILANCE RESEARCH QUERY — academic use only]\n"
        + user_prompt
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": framed_prompt},
    ]

    steps: list[dict] = []
    final_report: dict | None = None
    report_markdown: str | None = None   # captured from generate_pharmacovigilance_report
    collected_articles: list[dict] = []    # real PubMed articles from tool results
    valid_pmids: set[str] = set()          # whitelist for PMID scrubber
    pubmed_audit_entries: list[dict] = []  # audit metadata from each PubMed tool call
    # Pre-computed tier + summary from Phase 2 isolated extraction in tools.py.
    # These are used with PRIORITY over LLM-batch-generated article_summaries to
    # eliminate cross-contamination: each was extracted from a single article's
    # abstract in an isolated LLM call with no other articles in context.
    precomputed_summaries: dict[str, str] = {}  # pmid → pv_summary
    precomputed_tiers: dict[str, str] = {}       # pmid → pv_tier
    last_choice = None
    MAX_ITERATIONS = 20         # allows full investigation + report + submit

    for _ in range(MAX_ITERATIONS):

        # ── Reason: ask the LLM what to do next ──────────────────────────────
        response = llm_client.chat.completions.create(
            model       = CHAT_MODEL,
            messages    = messages,
            tools       = TOOLS,
            tool_choice = "auto",
        )
        last_choice = response.choices[0]
        messages.append(last_choice.message)   # add assistant turn to history

        # Agent chose to reply with text only (no tool call) — done
        if last_choice.finish_reason == "stop" or not last_choice.message.tool_calls:
            break

        # ── Act: execute every tool the LLM requested ────────────────────────
        for tool_call in last_choice.message.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)

            # ── Inject deterministic literature before generate_pharmacovigilance_report ──
            if fn_name == "generate_pharmacovigilance_report":
                # LLM-batch-provided summaries (lower priority — cross-contamination risk)
                raw_summaries = fn_args.get("article_summaries") or []
                llm_summaries_dict = {
                    s["pmid"]: s["relevance_summary"]
                    for s in raw_summaries
                    if isinstance(s, dict) and s.get("pmid") in valid_pmids and s.get("relevance_summary")
                }
                llm_tiers_dict = {
                    s["pmid"]: s.get("tier", "1")
                    for s in raw_summaries
                    if isinstance(s, dict) and s.get("pmid") in valid_pmids
                }
                # Pre-computed summaries override LLM-batch values.
                # Each was extracted in an isolated single-article LLM call during
                # Phase 2 screening → structurally immune to cross-contamination.
                summaries_dict = {**llm_summaries_dict, **precomputed_summaries}
                tiers_dict     = {**llm_tiers_dict,     **precomputed_tiers}
                lit_section, pmid_to_number = _build_literature_section(
                    collected_articles, summaries_dict, pubmed_audit_entries, tiers_dict
                )
                fn_args["literature_section"] = lit_section
                if "summary_findings" in fn_args:
                    fn_args["summary_findings"] = _replace_pmid_citations(
                        fn_args["summary_findings"], pmid_to_number, valid_pmids
                    )

            try:
                result = dispatch(fn_name, fn_args)
            except Exception as exc:
                result = {"error": f"Tool execution failed: {exc}"}

            # ── Collect real articles + audit from PubMed tool results ──────────
            if fn_name in ("fetch_pubmed_advanced", "search_drug_class_effects"):
                # Collect audit metadata for the literature section header
                audit = result.get("audit")
                if audit:
                    query_label = (
                        fn_args.get("query_term")
                        or f"{fn_args.get('drug_class', '')} + {fn_args.get('adverse_event', '')}"
                    )
                    pubmed_audit_entries.append({**audit, "query": query_label})

                for art in result.get("results", []):
                    pmid = art.get("pmid")
                    if pmid and pmid not in valid_pmids:
                        collected_articles.append(art)
                        valid_pmids.add(pmid)
                        # Capture pre-computed tier + summary from Phase 2 isolated extraction.
                        # These were generated one-article-at-a-time — zero cross-contamination.
                        if art.get("pv_summary"):
                            precomputed_summaries[pmid] = art["pv_summary"]
                        if art.get("pv_tier"):
                            precomputed_tiers[pmid] = art["pv_tier"]

            # Capture the markdown report when generated
            if fn_name == "generate_pharmacovigilance_report":
                report_markdown = result.get("report_markdown")

            # Record step for the UI execution trace
            steps.append({
                "module":   fn_name,
                "prompt":   fn_args,
                "response": fn_args if fn_name in ("submit_final_report", "abort_investigation") else result,
            })

            # ── Observe: feed result back to the LLM ─────────────────────────
            messages.append({
                "role":         "tool",
                "tool_call_id": tool_call.id,
                "content":      json.dumps(result),
            })

            if fn_name == "submit_final_report":
                final_report = fn_args           # normal investigation complete
            elif fn_name == "abort_investigation":
                final_report = {                 # guardrail triggered — wrap as report
                    "status":                "aborted",
                    "abort_code":            fn_args.get("abort_code"),
                    "confidence_score":      0,
                    "evidence_chain":        [s["module"] for s in steps],
                    "signal_level":          "no signal",
                    "novel_signal_detected": False,
                    "severe_events_found":   [],
                    "reasoning":             fn_args.get("reason", "Investigation aborted."),
                    "recommended_action":    "Discard - No Novel Signal",
                }

        if final_report:
            break

    # Fallback if agent never called submit_final_report or abort_investigation
    if not final_report:
        last_text = (
            last_choice.message.content
            if last_choice and last_choice.message.content
            else "Agent completed without a final report."
        )
        final_report = {
            "confidence_score":      0,
            "evidence_chain":        [s["module"] for s in steps],
            "signal_level":          "no signal",
            "novel_signal_detected": False,
            "severe_events_found":   [],
            "reasoning":             last_text,
            "recommended_action":    "Discard - No Novel Signal",
        }

    # Attach the markdown report to final_report so index.py can expose it
    if report_markdown:
        final_report["report_markdown"] = report_markdown

    return final_report, steps


# ── Supabase Logging ──────────────────────────────────────────────────────────

def log_to_supabase(user_prompt: str, response: str, steps: list) -> None:
    """Persist run results to Supabase agent_logs table (non-blocking)."""
    if not supabase:
        return
    try:
        supabase.table("agent_logs").insert({
            "user_prompt": user_prompt,
            "response":    response,
            "steps":       steps,
        }).execute()
    except Exception as exc:
        print(f"[Supabase log error — non-blocking] {exc}")
