"""
agent.py — ReAct loop and Supabase logging.

run_react_loop(user_prompt) -> (final_report: dict, steps: list)
  The agent iterates: Reason → Act (call a tool) → Observe (get result) → Reason again,
  until the agent calls submit_final_report or MAX_ITERATIONS is reached.
"""

import json
import re
import time
import config as _cfg
from config import llm_client, supabase, CHAT_MODEL
from tools import TOOLS, dispatch


# ── Debug Trace Helper ─────────────────────────────────────────────────────────

def _trace(label: str, **kwargs) -> None:
    """Print a structured crash-point trace and accumulate in config.trace_log.
    Gated by DEBUG_TRACE in config.py."""
    if not _cfg.DEBUG_TRACE:
        return
    parts = " | ".join(f"{k}={v!r}" for k, v in kwargs.items()) if kwargs else ""
    line = f"[TRACE] {label}" + (f" — {parts}" if parts else "")
    print(line)
    _cfg.trace_log.append(line)


# ── LLM Call Retry Helper ──────────────────────────────────────────────────────

def _llm_with_retry(client, max_attempts: int = 3, **kwargs):
    """
    Call client.chat.completions.create() with exponential-backoff retry.
    Retries on transient errors: 429 (rate limit), 5xx (server errors),
    connection drops, and timeouts. Raises on the last attempt or on
    non-transient errors (e.g. 400 Bad Request).
    """
    delay = 1.0
    for attempt in range(max_attempts):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            err    = str(exc).lower()
            transient = (
                status in (429, 500, 502, 503, 504)
                or any(t in err for t in (
                    "connection", "timeout", "rate limit",
                    "service unavailable", "bad gateway",
                ))
            )
            if transient and attempt < max_attempts - 1:
                print(f"[LLM transient error — attempt {attempt + 1}/{max_attempts}] {exc}. Retrying in {delay:.0f}s…")
                time.sleep(delay)
                delay *= 2
            else:
                raise

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

    # ── Section 2: Collapsible — background articles only ────────────────────────
    background_articles = tier2 + no_summary
    if background_articles:
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

        # Tier 2 — 1-line contextual note
        for art in tier2:
            pmid         = art.get("pmid", "")
            summary_note = summaries.get(pmid, "No direct AE data.")
            collapsible_items.append(
                _art_header(art) + f"\n   *{summary_note}*"
            )

        # Articles with no summary — title + PMID only
        for art in no_summary:
            collapsible_items.append(_art_header(art))

        n_bg = len(background_articles)
        collapsible_block = (
            f"<details>\n"
            f"<summary>📄 Click to expand {n_bg} background article{'s' if n_bg != 1 else ''} "
            f"(reviews &amp; mechanistic studies — no direct AE reports)</summary>\n"
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

SYSTEM_PROMPT = """[PHARMACOVIGILANCE RESEARCH PLATFORM — ICH E2D / CIOMS VI]
You are VigiLenseAI — an autonomous Pharmacovigilance Triage Agent.
Mission: Identify NOVEL adverse drug event signals not yet documented in the internal safety knowledge base.
Not medical advice. All outputs are for regulatory signal detection only.

ZERO FABRICATION — non-negotiable:
Every fact, PMID, number, or claim MUST come from an actual tool response this session. Never invent or infer from training knowledge. If uncertain — exclude it.

MANDATORY FIRST STEP: Your VERY FIRST call MUST be `query_knowledge_base`. If it returns drug_in_formulary=false → abort immediately with "drug_not_in_portfolio".

WORKFLOW:
1. `query_knowledge_base` — establish baseline (call multiple times, once per finding). Only use chunks discussing the INVESTIGATED AE or clinically adjacent findings (same organ system/mechanism). Discard unrelated AE categories.
2. `get_drug_profile` — retrieve pharmacological class and active ingredients.
3. `fetch_pubmed_advanced` — QUERY SCOPE RULE: first call MUST use the EXACT AE term from the user query (e.g. `"sildenafil" AND "myocardial infarction"`). Only expand to a cluster in a second call if count=0. Always include drug name in query_term and pass investigation_context.
4. `fetch_fda_adverse_events` + `calculate_disproportionality` — statistical analysis.
   - Stage 1: Use explicit 2×2 counts from literature if available (source: "Literature / PubMed (PMID: X)").
   - Stage 2: If no literature counts → use FAERS (source: "OpenFDA FAERS Database"). Pass case_counts (a/b/c/d) to generate_pharmacovigilance_report.
   - No data: proceed with ror=null, note absence in summary_findings.
5. `generate_pharmacovigilance_report` then `submit_final_report` — conclude.

Conclude after 3–5 KB queries and 2–3 PubMed searches. Do not search indefinitely.

FDA LABEL CROSS-MAPPING:
1. First query: EXACT AE term.
2. If no match → mandatory secondary query with adjacent organ-system terms (cardiac → "arrhythmia", "QTc prolongation", "sudden cardiac death"; hepatic → "hepatotoxicity", "liver injury"; spinal → "ischemic stroke", "NAION").
3. Report exact and adjacent matches separately in the baseline section.

SIGNAL CLASSIFICATION — signal_level:
- "significant": FAERS ROR ≥ 2.0 AND lower 95% CI > 1.0.
- "potential": FAERS negative/uncalculable BUT ≥1 novel Tier 1 article. Use whenever recommending escalation.
- "none": ONLY when BOTH FAERS negative AND no novel Tier 1 literature. When in doubt → "potential".
is_significant = statistical threshold only. signal_level = composite expert judgment.

ARTICLE CLASSIFICATION:
Tier "1" ACTIONABLE: case report/series with patient AE occurrences; clinical trial with significant AE incidence; pharmacovigilance DB study with case counts. Default to "1" if uncertain.
Tier "2" BACKGROUND: narrative reviews; mechanistic/animal/in-vitro studies; PK/PD studies; studies concluding no AE occurrences.

relevance_summary format:
- Tier 1: 1-2 sentences — case count/risk metric, dose/timing, outcome. No author names.
- Tier 2: "No [AE] reported; [one-sentence mechanistic note]."
Each summary MUST be derived SOLELY from that article's own abstract — never borrow details from other abstracts. The system pre-computes isolated extractions and uses them with priority; your article_summaries are a fallback.

INVESTIGATION SCOPE — mandatory:
Every finding in every report section MUST concern the specific AE under investigation or a clinically adjacent finding (same organ system/mechanism). Exclude unrelated AE categories even if in the FDA label.

summary_findings structure (exact subheadings, bullet points):
### Internal KB / FDA Label Baseline
KB/label findings directly about the investigated AE or adjacent organ-system findings. No unrelated categories.
### Novel Findings
Literature findings NOT in the label. Cite with [PMID: XXXXXXXX] (auto-converted to numbered citations).
### Known / Expected Findings
Label-consistent in-scope findings. No unrelated boxed warnings.
### Signal Assessment
Evidence quality, signal strength, confidence — scoped to investigated AE only. Include demographics summary if fetch_fda_adverse_events returned demographic data.

DEDUPLICATION: Never call an article a duplicate unless it shares an IDENTICAL PMID or TITLE. Every distinct PMID is a distinct article.

GUARDRAILS — call abort_investigation immediately. Write reason in the user's language, concise and actionable:
1. query_too_vague: No specific drug name in prompt.
2. multiple_drugs_detected: Two or more distinct drugs named.
3. non_medical_query: No connection to drug safety or pharmacovigilance.
4. drug_not_recognized: get_drug_profile returns source="fallback" AND clearly not a real drug (gibberish, food, chemical). Do NOT trigger for real drugs that simply failed API lookup.
5. no_literature_found: Both fetch_pubmed_advanced AND search_drug_class_effects returned count=0.
6. drug_not_in_portfolio: query_knowledge_base returns drug_in_formulary=false (ALWAYS first call).

GENERAL QUERY (no specific AE mentioned): Run discovery scan — identify 2–3 likely AE categories from drug class, run one focused fetch_pubmed_advanced per category, select the AE with strongest novel signal for the full report. Note in Signal Assessment that this was a discovery scan."""


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
    MAX_ITERATIONS    = 15    # allows full investigation + report + submit
    WALL_CLOCK_BUDGET = 240   # seconds — bail out before Vercel's 300 s hard limit

    # Reset trace log for this run
    _cfg.trace_log.clear()
    start_time = time.time()
    _trace("run_react_loop START", prompt_len=len(user_prompt))

    for iteration in range(MAX_ITERATIONS):

        elapsed = time.time() - start_time
        if elapsed > WALL_CLOCK_BUDGET:
            _trace("WALL_CLOCK_BUDGET exceeded — breaking loop early", elapsed_s=round(elapsed, 1))
            break

        _trace(f"ITER {iteration + 1}/{MAX_ITERATIONS}", msg_count=len(messages), elapsed_s=round(elapsed, 1))

        # ── Reason: ask the LLM what to do next ──────────────────────────────
        _trace(f"ITER {iteration + 1} LLM call START")
        try:
            response = _llm_with_retry(
                llm_client,
                model       = CHAT_MODEL,
                messages    = messages,
                tools       = TOOLS,
                tool_choice = "auto",
            )
        except Exception as exc:
            print(f"[LLM call failed after retries — breaking loop] {exc}")
            _trace(f"ITER {iteration + 1} LLM call FAILED", error=str(exc))
            break   # exit gracefully; fallback report is built below
        last_choice = response.choices[0]
        _trace(
            f"ITER {iteration + 1} LLM call END",
            finish_reason=last_choice.finish_reason,
            tool_calls=len(last_choice.message.tool_calls or []),
        )
        messages.append(last_choice.message)   # add assistant turn to history

        # Agent chose to reply with text only (no tool call) — done
        if last_choice.finish_reason == "stop" or not last_choice.message.tool_calls:
            _trace(f"ITER {iteration + 1} no tool_calls — breaking (finish_reason={last_choice.finish_reason})")
            break

        # ── Act: execute every tool the LLM requested ────────────────────────
        for tool_call in last_choice.message.tool_calls:
            fn_name = tool_call.function.name
            _trace(f"ITER {iteration + 1} TOOL SELECTED", fn=fn_name)

            _trace(f"ITER {iteration + 1} parsing args for {fn_name}")
            fn_args = json.loads(tool_call.function.arguments)
            _trace(f"ITER {iteration + 1} args parsed", keys=list(fn_args.keys()))

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

            _trace(f"ITER {iteration + 1} dispatch START", fn=fn_name)
            try:
                result = dispatch(fn_name, fn_args)
                _trace(f"ITER {iteration + 1} dispatch END", fn=fn_name, result_keys=list(result.keys()) if isinstance(result, dict) else type(result).__name__)
            except Exception as exc:
                _trace(f"ITER {iteration + 1} dispatch EXCEPTION", fn=fn_name, error=str(exc))
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
                _trace(f"ITER {iteration + 1} submit_final_report called — loop will end")
                final_report = fn_args           # normal investigation complete
            elif fn_name == "abort_investigation":
                _trace(f"ITER {iteration + 1} abort_investigation called", code=fn_args.get("abort_code"))
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
            _trace(f"ITER {iteration + 1} final_report SET — exiting loop")
            break

    # Fallback: agent exhausted iterations without calling submit_final_report.
    # Force one final LLM call instructing it to generate a proper report immediately.
    # Skip the forced LLM call if the wall-clock budget was already exceeded.
    budget_remaining = WALL_CLOCK_BUDGET - (time.time() - start_time)
    if not final_report:
        _trace("FALLBACK: loop ended without final_report", steps_so_far=len(steps), budget_remaining_s=round(budget_remaining, 1))
        if not report_markdown and llm_client and budget_remaining > 30:
            messages.append({
                "role":    "user",
                "content": (
                    "You have reached the iteration limit. "
                    "You MUST now call `generate_pharmacovigilance_report` followed immediately "
                    "by `submit_final_report` using all the evidence you have gathered so far. "
                    "Do not call any other tools. Generate the report now."
                ),
            })
            _trace("FALLBACK forced LLM call START")
            try:
                forced = _llm_with_retry(
                    llm_client,
                    model       = CHAT_MODEL,
                    messages    = messages,
                    tools       = TOOLS,
                    tool_choice = "auto",
                )
                forced_choice = forced.choices[0]
                _trace("FALLBACK forced LLM call END", finish_reason=forced_choice.finish_reason, tool_calls=len(forced_choice.message.tool_calls or []))
                for tool_call in (forced_choice.message.tool_calls or []):
                    fn_name = tool_call.function.name
                    fn_args = json.loads(tool_call.function.arguments)
                    if fn_name == "generate_pharmacovigilance_report":
                        summaries_dict = {**{
                            s["pmid"]: s["relevance_summary"]
                            for s in (fn_args.get("article_summaries") or [])
                            if isinstance(s, dict) and s.get("pmid") in valid_pmids and s.get("relevance_summary")
                        }, **precomputed_summaries}
                        tiers_dict = {**{
                            s["pmid"]: s.get("tier", "1")
                            for s in (fn_args.get("article_summaries") or [])
                            if isinstance(s, dict) and s.get("pmid") in valid_pmids
                        }, **precomputed_tiers}
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
                        result = {"error": str(exc)}
                    if fn_name == "generate_pharmacovigilance_report":
                        report_markdown = result.get("report_markdown")
                    if fn_name == "submit_final_report":
                        final_report = fn_args
                        steps.append({"module": fn_name, "prompt": fn_args, "response": fn_args})
            except Exception as exc:
                _trace("FALLBACK forced LLM call EXCEPTION", error=str(exc))

        if not final_report:
            _trace("FALLBACK hardcoded report — no submit_final_report was ever called")
            final_report = {
                "confidence_score":      0,
                "evidence_chain":        [s["module"] for s in steps],
                "signal_level":          "no signal",
                "novel_signal_detected": False,
                "severe_events_found":   [],
                "reasoning":             "Agent reached iteration limit.",
                "recommended_action":    "Discard - No Novel Signal",
            }

    # Attach the markdown report to final_report so index.py can expose it
    if report_markdown:
        final_report["report_markdown"] = report_markdown

    _trace("run_react_loop END", total_steps=len(steps), has_report=bool(report_markdown))

    # ── Inject trace log into report markdown (DEBUG only) ───────────────────
    if _cfg.DEBUG_TRACE and _cfg.trace_log and "report_markdown" in final_report:
        trace_lines = "\n".join(_cfg.trace_log)
        debug_block = (
            "\n\n---\n\n"
            "<details>\n"
            "<summary>🔍 Debug Trace Log (click to expand)</summary>\n\n"
            "```\n"
            f"{trace_lines}\n"
            "```\n\n"
            "</details>"
        )
        final_report["report_markdown"] = final_report["report_markdown"].rstrip() + debug_block

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
