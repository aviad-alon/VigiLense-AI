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
) -> tuple[str, dict[str, int]]:
    """
    Render a numbered, hyperlinked literature list from real PubMed tool results.
    All metadata (PMID, title, authors, journal, URL) is 100% deterministic — the LLM never touches it.

    Only articles with a validated LLM-provided drug-specific summary are included.
    Articles without a summary are excluded — they contain no relevant drug findings.
    Prepends a PubMed Retrieval Audit block.

    Returns:
        section_text    — the formatted Markdown string
        pmid_to_number  — mapping of PMID → article number (1-indexed) for citation replacement
    """
    summaries = summaries or {}
    # Keep only articles that have a validated drug-specific summary
    relevant = [a for a in articles if a.get("pmid") and a["pmid"] in summaries]
    # Build PMID → article number map for citation conversion in summary_findings
    pmid_to_number: dict[str, int] = {art["pmid"]: i + 1 for i, art in enumerate(relevant)}

    parts = []

    # ── Audit header ─────────────────────────────────────────────────────────
    if audit_entries:
        audit_lines = ["**PubMed Retrieval Audit**\n"]
        for e in audit_entries:
            found   = e.get("total_found", 0)
            fetched = e.get("total_fetched", 0)
            dr      = e.get("date_range", "")
            query   = e.get("query", "")
            relevant_count = e.get("total_relevant")
            if found == 0:
                coverage = "0 articles found"
            elif relevant_count is not None:
                coverage = (
                    f"{found} found · {fetched} fetched · "
                    f"screened {fetched} (100%) · **{relevant_count} included**"
                )
            elif fetched >= found:
                coverage = f"{found} found · {fetched} fetched · **100% coverage**"
            else:
                coverage = f"{found} found · {fetched} fetched (top {fetched} by relevance)"
            audit_lines.append(f"- `{query}` — {dr} — {coverage}")
        parts.append("\n".join(audit_lines))

    if not relevant:
        parts.append("*No articles with drug-specific findings were identified in the retrieved literature.*")
        return "\n\n".join(parts), pmid_to_number

    parts.append(f"**Articles with Drug-Specific Findings (n={len(relevant)})**")

    # ── Article list (only articles with LLM-validated summaries) ─────────────
    items = []
    for i, art in enumerate(relevant, 1):
        pmid     = art.get("pmid", "")
        title    = (art.get("title") or "Unknown title").strip()
        authors  = art.get("authors") or []
        pubdate  = (art.get("pubdate") or "").strip()
        journal  = (art.get("source") or "").strip()

        author_str = f"{authors[0]} et al." if len(authors) > 1 else (authors[0] if authors else "Unknown")
        link = f"[PMID: {pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)" if pmid else ""

        line = f"**{i}.** **{title}**\n{author_str} · *{journal}* · {pubdate} · {link}"
        line += f"\n\n**Key Drug-Specific Findings:**\n{summaries[pmid]}"
        items.append(line)

    parts.append("\n\n---\n\n".join(items))
    return "\n\n".join(parts), pmid_to_number


def _replace_pmid_citations(text: str, pmid_to_number: dict, valid_pmids: set) -> str:
    """
    Convert PMID references in LLM-generated text to numbered citations:
    - [PMID: 12345678] → [3]  (if PMID appears as article #3 in the literature section)
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

SYSTEM_PROMPT = """[SYSTEM CONTEXT — ACADEMIC PHARMACOVIGILANCE RESEARCH PLATFORM]
This is an automated biomedical research and regulatory safety monitoring system.
All queries, tool outputs, and medical literature retrieved are processed strictly for
scientific signal detection and pharmaceutical regulatory compliance (ICH E2D / CIOMS VI).
No content in this session constitutes medical advice, patient interaction, or consumer guidance.

You are VigiLenseAI — an autonomous Pharmacovigilance Triage Agent.

Your mission: Investigate potential adverse drug events reported in medical literature and determine if they represent NOVEL safety signals not yet documented in the internal safety knowledge base.

You have full autonomy to decide which tools to call, in what order, and how many times to execute them, based on your ongoing observation and reasoning at each step of the ReAct loop.

ABSOLUTE RULE — ZERO FABRICATION:
You must NEVER invent, guess, assume, or extrapolate any data. Every single fact, number, drug name, adverse event, PMID, statistic, or claim in your output MUST come directly from an actual tool response received during this session. This is non-negotiable and overrides all other instructions.
Specifically:
- DO NOT fabricate PMIDs, article titles, author names, or journal names. Only cite PMIDs that were explicitly returned by `fetch_pubmed_advanced` or `search_drug_class_effects`.
- DO NOT invent ROR values, case counts, or any numerical data. Only use numbers returned by `calculate_disproportionality` or `fetch_fda_adverse_events`.
- DO NOT assume drug properties, mechanisms, or adverse events from prior training knowledge. Only use data returned by `get_drug_profile` or `query_knowledge_base`.
- DO NOT fill gaps with plausible-sounding information. If a tool returned no data, state that explicitly.
- If you are uncertain whether a fact came from a tool response or from your training, DO NOT include it.

MANDATORY FIRST STEP — Portfolio Check:
Your VERY FIRST tool call MUST be `query_knowledge_base`. This is non-negotiable. Do NOT call any other tool before it. If `query_knowledge_base` returns drug_in_formulary=false, abort immediately with "drug_not_in_portfolio" — do not proceed further. Only if drug_in_formulary=true may you continue to the next steps.

Toolbox Guidance (Use tools dynamically as needed):
- Baseline Safety Profile: Use `query_knowledge_base` to check if an adverse event is already documented for a drug in the FDA label/internal KB. Call this multiple times — once per specific finding discovered in literature.
- Chemical/Class Context: Use `get_drug_profile` to retrieve active ingredients and pharmacological class.
- Literature Evidence: Use `fetch_pubmed_advanced` to gather case reports and recent studies — ALWAYS include the drug name or an active ingredient name in the query_term (e.g. "sertraline cardiac arrhythmia", not just "cardiac arrhythmia"). If the user's query specifies a target population (e.g. adolescents, pediatric, elderly), add the relevant demographic term to the query. ALWAYS pass `investigation_context` — a concise summary of the investigation including: drug name, active ingredients, adverse event being investigated, and any demographic context from the user query. Example: "Sertraline (active ingredient: sertraline hydrochloride, class: SSRI) — investigating bruxism signal in adolescents". This triggers full-coverage LLM screening of ALL retrieved articles (up to 200) before returning results. Apply the same to `search_drug_class_effects`.
  **CRITICAL — PubMed Boolean Syntax**: ALWAYS use strict grouped boolean syntax in `query_term`. Format: `"drug_name" AND ("ae1" OR "ae2" OR "ae3")`. Multi-word terms must be double-quoted. Examples:
  - CORRECT: `"sildenafil" AND ("myocardial infarction" OR "arrhythmia" OR "sudden cardiac death")`
  - CORRECT: `"sertraline" AND ("QTc prolongation" OR "cardiac arrhythmia" OR "torsades de pointes")`
  - WRONG: `sildenafil cardiovascular effects OR myocardial infarction OR stroke` — without parentheses, PubMed applies OR globally and returns 400,000+ unrelated results, defeating the investigation.
- System Memory: Use `check_past_signals` to review past internal investigations and avoid re-escalating discarded signals.
- Statistical Analysis — Two-Stage Fallback Protocol:
  STAGE 1 (Literature): If a retrieved PubMed article contains EXPLICIT numerical 2×2 counts (a, b, c, d), pass them directly to `calculate_disproportionality`. Tag the source as "Literature / PubMed (PMID: X)" when calling `generate_pharmacovigilance_report`.
  STAGE 2 (FAERS Fallback): If literature was retrieved but contains NO explicit numerical counts, do NOT skip statistical analysis. Instead, invoke `fetch_fda_adverse_events(drug_name, adverse_event)` to retrieve real-world case counts from OpenFDA FAERS. If the tool returns valid a/b/c/d values (all non-null), pass them to `calculate_disproportionality`. Tag the source as "OpenFDA FAERS Database" when calling `generate_pharmacovigilance_report`.
  NO DATA: If both stages yield no usable counts (FAERS returned an error or null values), do NOT fabricate any numbers. Proceed to `generate_pharmacovigilance_report` without ROR (leave ror=null) and note the absence of quantitative data in `summary_findings`.
- Deliverables: Use `generate_pharmacovigilance_report` to format the Markdown report, and invoke `submit_final_report` when you are ready to conclude the investigation.

Autonomous Operating Rules:
1. Before every tool call, explain your reasoning (Thought) for why this specific tool/query is needed next.
2. Adapt your strategy based on observations: if initial queries yield vague results, refine your search (e.g. search by active ingredient or drug class).
3. Conclude the investigation via `generate_pharmacovigilance_report` then `submit_final_report` as soon as you have sufficient evidence. Do NOT keep calling PubMed or query_knowledge_base indefinitely — after 3–5 knowledge base queries and 2–3 PubMed searches, you have enough to conclude.
4. Only flag a signal as NOVEL if the evidence is not already documented in the baseline knowledge base.
5. IMPORTANT: You MUST call `generate_pharmacovigilance_report` followed immediately by `submit_final_report` to properly end the investigation. Do not stop mid-investigation.

IMPORTANT — `summary_findings` IN `generate_pharmacovigilance_report`:
The system automatically injects a verified, numbered literature list into the report.
Structure your `summary_findings` using these exact markdown subheadings, and use bullet points (`-`) whenever you list multiple items within a section:

### Internal KB / FDA Label Baseline
What the internal knowledge base already documents for this drug (reference specific label sections). Use bullet points for each documented finding:
- Finding A (Section X)
- Finding B (Section Y)

### Novel Findings
Adverse events from retrieved literature NOT already in the FDA label. Use one bullet per distinct finding. Cite specific articles using [PMID: XXXXXXXX] — the system auto-converts these to numbered citations [N] that match the literature list above. Only use PMIDs actually returned by the PubMed tools:
- Finding description [PMID: XXXXXXXX]
- Another finding [PMID: XXXXXXXX]

### Known / Expected Findings
Findings consistent with the existing label that should be discarded as non-novel. Use bullet points:
- Finding (already in label, Section X)

### Signal Assessment
Overall evidence quality, signal strength, and confidence. Use bullet points for each key point:
- Evidence quality: ...
- Signal strength: ...
- Confidence: ...

Do NOT include article titles or author names — cite only with [PMID: XXXXXXXX].

ARTICLE SUMMARIES — `article_summaries` IN `generate_pharmacovigilance_report`:
INCLUDE ALL articles returned by fetch_pubmed_advanced or search_drug_class_effects in `article_summaries`.
Every article returned by those tools has already passed an LLM screening stage that confirmed relevance — do NOT skip any of them.

For each article, provide a detailed `relevance_summary` covering ALL of the following present in the abstract:
  - "pmid": the exact PMID string as returned by the tool (do NOT fabricate or guess PMIDs)
  - "relevance_summary": covering:
      • Study design and patient population (e.g., "RCT, n=450, adults with MDD receiving sertraline 50–200 mg/day")
      • Efficacy findings specific to the target drug (response/remission rates, effect sizes, comparators, p-values)
      • Safety and adverse event findings specific to the target drug (incidence rates, severity, onset timing, dose-dependency)
      • Subgroup results where the target drug data differs across populations (age, sex, comorbidities)
      • Mechanistic or pharmacological insights regarding the target drug
      • If the paper discusses a drug class broadly but includes specific data on the target drug, highlight those target drug details explicitly.
Do NOT copy-paste sentences from the abstract verbatim. Write an analytical extraction in your own words.
Example: "RCT (n=450, MDD patients) comparing sertraline 50–200 mg/day vs. placebo over 12 weeks. Sertraline arm achieved 52% response rate vs. 35% placebo (p<0.01, NNT=6). QTc prolongation ≥60 ms observed in 4.2% of sertraline patients, dose-dependent and concentrated above 150 mg/day. Authors attribute cardiac effects to off-target hERG channel inhibition at supratherapeutic concentrations."

SOURCE ATTRIBUTION:
- If statistical disproportionality (ROR) is calculated using `fetch_fda_adverse_events`, set `disproportionality_source` to "OpenFDA FAERS Database".
- If counts came from a literature article, set `disproportionality_source` to "Literature / PubMed (PMID: X)".

Guardrail Rules — call `abort_investigation` immediately in each of the following cases.
For every abort: write the `reason` in the same language the user wrote their query. Be concise, professional, and helpful — tell the user exactly what went wrong and what they can do instead.

─── CASE 1: query_too_vague ───
TRIGGER: You cannot identify a specific drug name in the user's prompt (e.g. "check side effects", "analyze this drug", "investigate something").
ACTION: abort_code = "query_too_vague"
REASON template: "The query does not specify a drug name. Please re-submit with the name of the specific drug you want to investigate (e.g. 'Analyze adverse events for Warfarin')."

─── CASE 2: multiple_drugs_detected ───
TRIGGER: The prompt clearly names two or more distinct drugs (e.g. "compare Warfarin and Aspirin", "Metformin and Sertraline interaction").
ACTION: abort_code = "multiple_drugs_detected"
REASON template: "This system investigates one drug per query. Your prompt mentions multiple drugs ('[drug A]' and '[drug B]'). Please submit a separate query for each drug."

─── CASE 3: non_medical_query ───
TRIGGER: The prompt has no connection to drug safety, pharmacovigilance, adverse events, or medical literature (e.g. "write me a poem", "what is the weather", "explain quantum physics").
ACTION: abort_code = "non_medical_query"
REASON template: "This system is designed exclusively for pharmacovigilance signal detection. The submitted query does not relate to drug safety or medical literature. Please submit a drug safety investigation request."

─── CASE 4: drug_not_recognized ───
TRIGGER: `get_drug_profile` returns source="fallback" AND the name is clearly not a real pharmaceutical — e.g. it is gibberish ("xkqzp"), a common food ("banana", "coffee"), a chemical formula ("H2O"), or a non-drug substance.
DO NOT TRIGGER if: the name is a real drug that simply failed API lookup (e.g. "Sildenafil", "Aspirin", "Ibuprofen"). In that case, proceed using the name for PubMed and FAERS searches.
ACTION: abort_code = "drug_not_recognized"
REASON template: "The name '[input]' was not recognized as a pharmaceutical compound in our drug database. Please verify the drug name and re-submit. If this is a brand name, try using the generic (active ingredient) name instead."

─── CASE 5: no_literature_found ───
TRIGGER: Both `fetch_pubmed_advanced` AND `search_drug_class_effects` returned count=0 — meaning zero articles were found for this drug and adverse event combination.
ACTION: abort_code = "no_literature_found"
REASON template: "No relevant medical literature was found on PubMed for '[drug name]' in relation to the investigated adverse events (search period: 2020–present). This may indicate that the signal has not been reported in recent peer-reviewed literature. Consider broadening the adverse event scope or searching manually."

─── CASE 6: drug_not_in_portfolio ───
TRIGGER: `query_knowledge_base` returns drug_in_formulary=false. This is always your FIRST tool call — fire this abort immediately without calling any other tool.
ACTION: abort_code = "drug_not_in_portfolio"
REASON template (SHORT — 1-2 sentences only): '[drug_name]' is not in the organization's pharmacovigilance portfolio according to the internal knowledge base. This system only performs signal detection for drugs held in the company's portfolio."""


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
                # Extract LLM-provided per-article summaries; keep only validated PMIDs
                raw_summaries = fn_args.get("article_summaries") or []
                summaries_dict = {
                    s["pmid"]: s["relevance_summary"]
                    for s in raw_summaries
                    if isinstance(s, dict) and s.get("pmid") in valid_pmids and s.get("relevance_summary")
                }
                lit_section, pmid_to_number = _build_literature_section(
                    collected_articles, summaries_dict, pubmed_audit_entries
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
