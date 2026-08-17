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
  When calling `generate_pharmacovigilance_report` and you have FAERS disproportionality data, ALWAYS pass `case_counts` with the raw a/b/c/d values from `fetch_fda_adverse_events` so the 2×2 matrix appears in the statistics table.
  When `fetch_fda_adverse_events` returns demographic data (gender, age groups, top concomitant drugs), include a concise demographics summary in the Signal Assessment section of `summary_findings`.

COMPOSITE SIGNAL CLASSIFICATION — `signal_level` in `generate_pharmacovigilance_report`:
This field drives the master report header and Subject table. It MUST reflect BOTH evidence sources:

  "significant" → FAERS ROR ≥ 2.0 AND lower 95% CI > 1.0. Statistical threshold confirmed.

  "potential"   → FAERS is negative OR not calculable, BUT at least one Tier 1 article was found
                  with a novel adverse event NOT already fully documented in the FDA label /
                  internal KB for this specific AE. This includes: unlabeled case reports,
                  case series, or clinical trial findings for the investigated event.
                  Use this whenever your Regulatory Recommendation is to escalate or review.
                  A safety officer reading 🟢 while you recommend escalation is a compliance risk.

  "none"        → Use ONLY when BOTH: (a) FAERS shows no disproportionality or is uncalculable,
                  AND (b) literature contains NO novel Tier 1 findings beyond what the label
                  already documents. If any unlabeled risk exists in literature → use "potential".

RULE: `is_significant` is STATISTICAL ONLY (ROR threshold). `signal_level` is your expert
composite judgment. When in doubt between "potential" and "none", always choose "potential".

FDA LABEL CROSS-MAPPING — NEAREST TERM MATCHING:
When checking the Internal KB / FDA Label Baseline for an adverse event:
1. First call `query_knowledge_base` with the EXACT reported adverse event term (e.g., "spinal cord infarction").
2. If no exact match is documented, MANDATORY secondary check: call `query_knowledge_base` again with overlapping or anatomically adjacent terms from the same organ system or mechanism. Examples:
   - For spinal cord events: also query "transverse myelitis", "spinal ischemia", "ischemic stroke", "NAION"
   - For cardiac events: also query "arrhythmia", "QTc prolongation", "sudden cardiac death"
   - For hepatic events: also query "hepatotoxicity", "liver injury", "elevated transaminases"
3. Report BOTH levels of matching in the Internal KB / FDA Label Baseline section — exact matches and adjacent-term matches separately.

DEDUPLICATION RULE:
Do NOT label any article as a duplicate unless it shares an IDENTICAL PMID or IDENTICAL TITLE with another already-processed article. Different papers on the same topic, same drug, or same adverse event are NOT duplicates. Never write "(Duplicate...)" or "(Same as...)" in any `relevance_summary`. Every distinct PMID is a distinct article.

TWO-TIER ARTICLE CLASSIFICATION in `article_summaries`:
Every article MUST include a `tier` field. Classify BEFORE writing the summary:

TIER "1" — ACTIONABLE (renders full entry in the report). Use when the article meets ANY of:
  - Direct case report or case series documenting actual patient occurrences of the target AE.
  - Clinical trial or cohort study reporting statistically significant incidence, risk ratio, or HR for the target AE.
  - Novel safety alert or pharmacovigilance database study (FAERS, VigiBase, WHO) with direct case counts.
  Default to "1" if uncertain.

TIER "2" — BACKGROUND (omitted from individual entries; counted in grouped note only). Use when:
  - General safety narrative review with no original case data.
  - Mechanistic, animal, or in-vitro study with no patient AE reports.
  - PK/PD study not reporting the target AE.
  - Study explicitly concluding no occurrences of the target AE.

`relevance_summary` format by tier:
  Tier 1 → 1-2 tight sentences: state case count or risk metric, dose/timing if known, clinical outcome. No titles or author names.
    Example: "Case series (n=3, males 45–62 yrs): sildenafil 50–100 mg associated with acute NAION onset within 24h of ingestion; visual recovery partial in 2/3 cases."
  Tier 2 → Single line only: "No [target AE] reported; [one-sentence mechanistic or contextual note]."
    Example: "No NAION reported; supports systemic hypotension as a plausible mechanism via PDE5-mediated vasodilation."

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
INCLUDE ALL articles returned by fetch_pubmed_advanced or search_drug_class_effects — every returned article has already passed LLM screening. Do NOT skip any. Only use PMIDs actually returned by those tools.

For each article provide: `pmid` (exact, do not fabricate), `tier` ("1" or "2" per TWO-TIER ARTICLE CLASSIFICATION above), and `relevance_summary` (format per tier above).

Do NOT copy-paste abstract text verbatim — write analytical extractions. Do NOT include article titles or author names in `relevance_summary`.

STRICT ISOLATED EXTRACTION — MANDATORY for every `relevance_summary`:
Each `relevance_summary` must be derived SOLELY from the abstract of that specific PMID.
NEVER borrow, carry over, or infer ANY clinical detail (case count, patient age/sex, dose, outcome) from another article's abstract.

TITLE-TO-FINDING VALIDATION — run this check before writing each summary:
  - If the title describes an animal, in-vitro, or mechanistic study → assign Tier "2" and do NOT include any patient case details in the summary.
  - If the title says "case report" or "case series" → assign Tier "1" and include ONLY the case details explicitly stated in THAT SPECIFIC abstract.

MANDATORY SELF-CHECK — before submitting `article_summaries`:
For each entry, ask: "Does this PMID's relevance_summary contain ANY fact that came from a DIFFERENT PMID's abstract?" If yes — delete that fact immediately. Every entry must be 100% self-contained.

Note: the system performs its own pre-computed isolated extraction during the screening phase and will use those values with priority. Your `article_summaries` serve as a fallback — apply the same isolation rules regardless.

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
