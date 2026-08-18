"""
agent.py — ReAct loop and Supabase logging.

run_react_loop(user_prompt) -> (final_report: dict, steps: list)
  The agent iterates: Reason → Act (call a tool) → Observe (get result) → Reason again,
  until the agent calls submit_final_report or MAX_ITERATIONS is reached.
"""

import json
import re
import time
import openai
from config import llm_client, supabase, CHAT_MODEL
from tools import TOOLS, dispatch

# Hard wall: exit the ReAct loop this many seconds before Vercel's 300s maxDuration.
# Ensures Python can still serialise and return a valid HTTP response.
_LOOP_TIME_BUDGET = 240  # seconds

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
    Render a literature block from real PubMed tool results.
    All articles have passed the Phase 2 gate — only actionable articles with direct patient evidence.
    All metadata (PMID, title, authors, journal, URL) is 100% deterministic — the LLM never touches it.

    Returns:
        section_text    — the formatted Markdown string
        pmid_to_number  — mapping of PMIDs → article number (1-indexed)
    """
    summaries = summaries or {}

    # Articles with pre-computed summaries (passed Phase 2 gate)
    included   = [a for a in articles if a.get("pmid") and a["pmid"] in summaries]
    no_summary = [a for a in articles if a.get("pmid") and a["pmid"] not in summaries]

    all_ordered = included + no_summary
    pmid_to_number: dict[str, int] = {art["pmid"]: i + 1 for i, art in enumerate(all_ordered)}

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

        if all_ordered:
            from collections import Counter
            type_counts = Counter(a.get("pv_study_type", "") for a in included if a.get("pv_study_type"))
            type_summary = ", ".join(f"{v}× {k}" for k, v in sorted(type_counts.items()))
            audit_lines.append(
                f"\n**Pipeline summary:** {len(articles)} screened"
                f" → **{len(included)} included**"
                + (f" ({type_summary})" if type_summary else " (direct patient-level AE evidence)")
            )
        parts.append("\n".join(audit_lines))

    # ── Actionable articles ───────────────────────────────────────────────────
    if not included:
        parts.append(
            "*No actionable articles with direct adverse event reports were identified "
            "in the retrieved literature.*"
        )
    else:
        parts.append(
            f"**Actionable Literature — Direct AE Evidence "
            f"({len(included)} article{'s' if len(included) != 1 else ''})**"
        )
        items = []
        for art in included:
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

            study_tag = f" · `{art.get('pv_study_type')}`" if art.get("pv_study_type") else ""
            line  = f"**{n}.** {title} — {author_str} · *{journal}* · {pubdate}{study_tag} · {link}"
            line += f"\n   **Key Finding:** {summaries[pmid]}"
            items.append(line)

        parts.append("\n\n".join(items))

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

ZERO FABRICATION — ABSOLUTE RULE:
Every fact, PMID, number, or statistic MUST come from an actual tool response this session. Never invent data, fill gaps, or cite PMIDs not returned by a tool. If a tool returned nothing, say so explicitly.

PARALLEL EXECUTION POLICY:
Whenever two or more tool calls are independent — meaning neither result depends on the other — issue ALL of them in a single turn, not sequentially across multiple turns.
Mandatory parallel opportunities:
  • Drug profile + initial KB safety query → always independent, always issue together.
  • Multiple KB queries for different AEs → independent, batch into one turn.
  • FAERS query for one AE + KB adjacency check for another → independent, issue together.
One turn = one LLM reasoning step. Maximizing parallelism directly reduces cost and latency. Use it aggressively.

NON-REDUNDANCY POLICY:
Before calling any tool, verify the conversation history:
  • Identical tool + identical parameters already called → do NOT repeat. Use the prior result.
  • A tool returned 0 results with a specific query → adapt query parameters or proceed without. Do not retry the exact same call.
  • Do not fetch more literature once you have sufficient patient-level evidence to classify all identified AEs.
Redundant calls increase cost and latency without improving the report.

MANDATORY REPORT TOOL CALL — NON-NEGOTIABLE:
You MUST end EVERY investigation (both Targeted and Broad Mode) by calling the `generate_pharmacovigilance_report` tool followed by the `submit_final_report` tool.
Writing the report as raw text in your final message WITHOUT calling these tools is STRICTLY FORBIDDEN — even if you feel you already have all the information you need.
When ready to report: STOP reasoning → call `generate_pharmacovigilance_report` → then call `submit_final_report`. No exceptions.

INVESTIGATION WORKFLOW:

── STEP 0 — Portfolio Check (always first, always alone):
Call `query_knowledge_base` as your very first action. If drug_in_formulary=false → call `abort_investigation` with "drug_not_in_portfolio" immediately. No other tool may precede this.

── STEP 1 — Parallel Initialization (single turn):
Once portfolio membership is confirmed, determine the investigation mode and issue the following in ONE parallel turn:
  • `get_drug_profile` — resolve active ingredients, pharmacological class, and mechanism.
  • `query_knowledge_base` — initial safety baseline query:
      Targeted Mode: use the specific AE focus term (e.g., "bleeding risk", "hepatotoxicity").
      Broad Mode: use broad safety terms (e.g., "adverse events", "drug interactions", "safety").
  Mode detection from the query:
    TARGETED MODE            — query contains "focus on [event]" → adverse_event = the event, surveillance_mode = false.
    BROAD SURVEILLANCE MODE  — no specific event named → adverse_event = "General Safety Surveillance — Broad Scan", surveillance_mode = true.

── STEP 2 — Evidence-Driven Investigation:

  TARGETED MODE:
  a. Expand label baseline if needed: issue 1–2 additional parallel `query_knowledge_base` calls for adjacent organ-system terms or the drug_interactions section — only if the initial query left coverage gaps.
  b. PubMed search: `fetch_pubmed_advanced` with the EXACT focus term in strict boolean syntax. If count=0, try active ingredient or class-level variant once.
  c. Evidence-triggered KB checks: as literature results arrive, for each AE that is NOT obviously the focus event or a clear class effect, verify label coverage with `query_knowledge_base`. Batch multiple AE checks into one parallel turn.
  d. FAERS fallback: only if literature has no 2×2 counts AND the AE is a Candidate Unlabeled Signal worth quantifying.

  BROAD SURVEILLANCE MODE — run steps (a) and (b) IN THE SAME PARALLEL TURN:
  a. PubMed broad search: `fetch_pubmed_advanced` with:
       query_term = '"[drug_name]" AND ("adverse drug reaction" OR "adverse effect" OR "toxicity" OR "case report" OR "pharmacovigilance")'
       investigation_context = "[drug] — broad safety surveillance (any adverse event)"
       surveillance_mode = true   ← REQUIRED — enables broader gate, date sort, and 2021-present window
       (min_year is automatically enforced to 2021 by the pipeline when surveillance_mode=true)
  b. FAERS AE discovery: `fetch_top_faers_events` with drug_name — call IN PARALLEL with (a).
       Returns top 15 most-reported AEs from spontaneous reports, independent of literature.

  c. AE Consolidation: merge the AE list from PubMed summaries + FAERS top list → deduplicate → build a master AE list.
       If PubMed returned 0 included articles, rely on the FAERS list as the PRIMARY AE source.
       Note for each AE: appeared in PubMed only / FAERS only / both.

  d. KB Classification: for each AE in the master list that is NOT obviously a labeled class effect,
       issue parallel `query_knowledge_base` calls (batch up to 4 per turn). Skip for AEs unambiguously labeled.

  ── ARTICLE-FINDING CROSS-CHECK (mandatory before proceeding to step e) ──
  After KB queries complete, examine every included PubMed article. For each distinct AE
  described in an included article:
    1. State the exact AE term as used in the article (e.g., "sensorineural hearing loss", "acute pancreatitis").
    2. Ask: does that EXACT term (or a direct MedDRA PT synonym) appear in the KB chunks?
       YES → Bucket 1. NO (only broader/adjacent terms found) → Bucket 3.
    3. You CANNOT classify an article-reported AE as Bucket 1 unless you can cite the specific
       term from the KB response. "The label covers myopathy" does NOT cover "hearing loss."
  This cross-check is the PRIMARY driver of your AE classification. KB queries are
  verification tools — the article findings determine what must be verified.

  e. FAERS Disproportionality — MANDATORY for every Candidate Unlabeled Signal (Bucket 3):
       For EACH AE classified as Bucket 3 after step (d), run this sequence:
         1. Call `fetch_fda_adverse_events(drug_name, ae_term)` to retrieve 2×2 counts.
         2. If a/b/c/d counts are returned (not None): immediately call `calculate_disproportionality` with those values.
         3. Record {ror, ci_95, faers_significant} from the result — include in the discovered_events entry for that AE.
       Batch up to 3 Bucket 3 AEs per parallel turn when multiple signals exist.
       ONLY skip if FAERS returns all-None counts (drug-AE pair genuinely absent from FAERS).
       ⚠️ DO NOT proceed to step (f) while any Bucket 3 AE is still awaiting FAERS quantification.

  ── PRE-REPORT MANDATORY CHECK ──
  Before calling generate_pharmacovigilance_report, verify your work:
       • Every AE in the master list has a bucket classification. ✓/✗
       • Every Bucket 3 AE has had fetch_fda_adverse_events → calculate_disproportionality run. ✓/✗
       If ANY Bucket 3 AE is unquantified → go back to step (e) NOW. Do not skip.

  f. Report: generate_pharmacovigilance_report with surveillance_mode=true,
       adverse_event = "General Safety Surveillance — Broad Scan",
       discovered_events = fully populated array of all classified AEs with ror/ci_95/faers_significant for Bucket 3.

  g. signal_level: "significant" if any AE has ROR ≥ 2.0 (lower CI > 1.0) OR any Bucket 2 finding with fatal/life-threatening outcome ·
       "potential" if any Candidate Unlabeled Signal OR any Bucket 2 trigger present (high incidence, vulnerable subpopulation, serious outcome) ·
       "none" ONLY if all AEs are Bucket 1 with no severity triggers present.

── STEP 3 — Stop When Evidence Is Complete:
Stop investigating when ALL of the following are true:
  (a) Every identified AE has a classification (Established Labeled Event, Label Discrepancy, or Candidate Unlabeled Signal).
  (b) Every Candidate Unlabeled Signal (Bucket 3) has had fetch_fda_adverse_events called AND calculate_disproportionality run — or FAERS returned all-None counts for that drug-AE pair.
  (c) No new distinct AE emerged from the most recent PubMed call.
Do NOT continue querying once these criteria are met. Call `generate_pharmacovigilance_report` immediately.
If results are insufficient: adapt query parameters once (different term, active ingredient, or class) — then proceed with available evidence. Never retry identical parameters.

TOOLBOX GUIDANCE:

1. `query_knowledge_base` — FDA label / internal KB. Use to: (a) confirm portfolio membership, (b) build label baseline, (c) verify label coverage for a specific AE.
   If no exact match on primary query → issue one secondary query with adjacent terms (same organ system / mechanism). Do not issue more than two KB queries per AE.
   Multiple KB calls in one turn are encouraged when checking different AEs simultaneously — batch them.

2. `get_drug_profile` — Resolves active ingredients, drug class, and mechanism. Call once per investigation (parallel with initial KB query in Step 1).

3. `fetch_pubmed_advanced` — Drug-specific AE literature from PubMed. Use strict boolean syntax with MeSH terms where possible:
   Preferred pattern: `"DrugName"[MeSH Terms] AND "AdverseEvent"[MeSH Terms]`
   Fallback pattern:  `"drug_name" AND ("ae_term1" OR "ae_term2")` — all multi-word terms double-quoted.
   Example: `"Methotrexate"[MeSH Terms] AND ("hepatotoxicity" OR "liver injury" OR "Drug-Induced Liver Injury"[MeSH Terms])`
   The pipeline automatically appends clinical publication-type filters (Case Reports, Clinical Trials, Observational Studies) — no need to add them manually.
   Always pass investigation_context. In Broad Surveillance Mode: pass surveillance_mode=true.

4. `search_drug_class_effects` — Class-level AE literature. Use to contextualize whether an AE is a class effect or drug-specific signal. Call after `get_drug_profile` to use the correct class name.

5. `check_past_signals` — Historical investigation logs. Prevents re-escalating already-reviewed signals.

6. `fetch_top_faers_events` — BROAD SURVEILLANCE MODE only. Call in Step 2 PARALLEL with `fetch_pubmed_advanced` to enumerate top N spontaneously reported AEs for the drug.
   Use the returned AE list as discovery seeds. High report count ≠ causality — classify every AE via `query_knowledge_base` before drawing conclusions.
   Do NOT call in Targeted Mode (use `fetch_fda_adverse_events` for quantification after AE is confirmed Bucket 3).

7. `fetch_fda_adverse_events` — FAERS real-world case counts for a specific drug-AE pair. Call ONLY when: (a) literature has no explicit 2×2 counts, AND (b) the AE is a Candidate Unlabeled Signal worth quantifying. Do NOT call for Established Labeled Events.
   When FAERS returns demographics (gender, age, concomitant drugs) → include a brief summary in Signal Assessment.

8. `calculate_disproportionality` — ROR + 95% CI. Call only with explicit numerical counts from a tool response. Tag source: "Literature / PubMed (PMID: X)" or "OpenFDA FAERS Database". Never estimate.

⛔ HR ≠ ROR — CRITICAL DISTINCTION:
  - ROR (Reporting Odds Ratio): a FAERS disproportionality metric. Comes ONLY from calculate_disproportionality.
    Place in discovered_events[].ror ONLY after running fetch_fda_adverse_events → calculate_disproportionality.
  - HR / OR / aHR / RR: effect sizes from PubMed cohort/RCT studies. Place in discovered_events[].literature_hr.
  ⚠️ NEVER copy an HR, OR, aHR, or RR from a PubMed article into the `ror` field.
     Doing so mislabels a literature finding as a FAERS pharmacovigilance signal — a methodological error.
     If FAERS was not queried for an AE, leave `ror` empty (omit it).

9. `generate_pharmacovigilance_report` → `submit_final_report` — Always in this order. Pass case_counts with raw a/b/c/d when disproportionality is available.

LITERATURE SIGNAL CLASSIFICATION — 3-BUCKET FRAMEWORK:
Apply to EVERY literature finding during your reasoning. Classify as you observe each tool result — do not defer to the report step.

⚠️ INTERNAL CLASSIFICATION ONLY — bucket names and code labels are for your reasoning only.
They MUST NEVER appear in summary_findings, recommendations, or any user-facing text.
Use these professional terms when writing for the user:
  Bucket 1 (confirmed_labeled)     → write: "Established Labeled Event"
  Bucket 2 (severity_discrepancy)  → write: "Label Discrepancy — Elevated Severity"
  Bucket 3 (potentially_unlabeled) → write: "Candidate Unlabeled Signal"

DEFAULT ASSUMPTION: Every finding starts as confirmed_labeled. Novelty must be proven, not assumed.

⚠️ STRICT LABELING CHECK — Bucket 1 requires that the returned KB chunks contain the EXACT AE term
(or a recognized MedDRA PT-level synonym). Semantic proximity is NOT sufficient:
  ✗ "Hearing Loss" is NOT covered by "Myopathy", "Sensory Disturbances", or "neurotoxicity"
  ✗ "Acute Pancreatitis" is NOT covered by "GI adverse effects" or "abdominal pain"
  ✗ "QTc Prolongation" is NOT covered by "cardiac monitoring" or "cardiovascular effects"
If the KB chunks describe a nearby organ system or mechanism but do NOT name the AE explicitly
→ the AE is NOT confirmed labeled → classify as Bucket 3.
"Mechanism-based" coverage applies ONLY when the KB chunk explicitly states this mechanism causes
this specific type of adverse event — NEVER infer coverage from general drug class action.

Bucket 1 — confirmed_labeled:
  The FDA label / KB covers the AE by name, by drug CLASS, or by mechanism.
  CLASS COVERAGE RULE: If the label mentions a pharmacological class or a mechanism statement,
  then drugs in that class are confirmed_labeled for AEs explicitly named under that class.
  → Report as "Established Labeled Event". Do NOT escalate.

Bucket 2 — severity_discrepancy:
  The AE is labeled BUT literature shows clearly higher frequency, severity, or broader patient population than described.
  Bucket 2 TRIGGERS — assign Bucket 2 (do NOT remain Bucket 1) if ANY of the following are explicitly reported:
    • Fatal or life-threatening outcome from the AE (patient death, ICU admission, organ failure requiring intervention).
    • Reported incidence ≥ 20% in any specific population (e.g., "53% of DILI cases in patients >65").
    • A high-risk subpopulation (elderly, renally/hepatically impaired, pediatric, specific comedication) not adequately warned in the current label.
    • Severity or dose gradient substantially beyond what the current label describes.
  → Report as "Label Discrepancy — Elevated Severity". Recommend "Monitor / Consider Label Update" — NEVER "Discard".

Bucket 3 — potentially_unlabeled:
  Assign ONLY if ALL are true:
  (a) The specific AE is NOT in the label by name.
  (b) The AE is NOT covered by any class-level or mechanism-level statement.
  (c) Absence confirmed via KB primary query AND a secondary query with adjacent terms.
  (d) Evidence comes from direct patient-level data (case report, trial, pharmacovigilance DB).
  → Report as "Candidate Unlabeled Signal". Set signal_level = "potential" or "significant".
  When uncertain between Bucket 2 and Bucket 3: default to Bucket 1. Never escalate class interactions.

COMPOSITE SIGNAL LEVEL (signal_level):
"significant" → FAERS ROR ≥ 2.0 AND lower 95% CI > 1.0, OR any Bucket 2 finding with a fatal or life-threatening outcome explicitly reported in the evidence.
"potential"   → At least one confirmed Bucket 3 finding, OR at least one Bucket 2 trigger present (high incidence ≥ 20%, vulnerable subpopulation not in label, serious non-fatal outcome beyond label description).
"none"        → ALL findings are Bucket 1 AND no Bucket 2 triggers are present. Expected for well-characterized drugs with no severity/frequency discrepancy — not a failure.
is_significant = statistical only (ROR). signal_level = composite expert judgment.
⚠️ A "Label Discrepancy — Elevated Severity" (Bucket 2) finding NEVER produces signal_level = "none". Always "potential" or "significant".

DIRECTIONAL SIGNAL RULE (ICH E2E / GVP Module IX — mandatory):
A Safety Signal is STRICTLY defined as evidence of increased risk, adverse drug reaction, or unexpected harm.
Protective effects (HR < 1, OR < 1, reduced incidence vs. comparator) are NEVER Safety Signals.

Decision tree — apply before setting signal_level:

  Case A — ALL evidence shows protective or neutral effect (HR < 1, OR < 1, reduced risk):
    → signal_level = "none"
    → recommendations = "No safety escalation required — findings show protective association only."
    → In summary_findings: include a "### Efficacy / Protective Observations" subsection summarizing
      the protective findings (HR values, populations studied, effect size). Do NOT write these
      under Novel Signal Assessment. Do NOT escalate.

  Case B — MIXED evidence (protective in overall cohort, elevated risk in a specific subgroup):
    → Isolate the risk subgroup ONLY for signal classification.
    → signal_level and recommendations reflect ONLY the risk element of the subgroup.
    → Protective component goes under "### Efficacy / Protective Observations" — clearly separated.
    → Do NOT average protective and risk effects to conclude "no signal" if a real risk subgroup exists.

  Case C — FAERS ROR > 1 AND literature shows only protective effect:
    → Report both independently. FAERS signal stands on its own (pharmacovigilance data ≠ RCT).
    → Do NOT cancel the FAERS signal because literature is protective (different populations/methods).
    → signal_level is driven by the FAERS ROR (significant or potential as usual).
    → In summary_findings: document the discordance explicitly. Note literature direction vs. FAERS direction.

⚠️ NEVER classify a protective finding (HR < 1, OR < 1) as signal_level = "potential" or "significant".
   Doing so violates GVP Module IX and is a CLASSIFICATION ERROR.

ARTICLE SUMMARIES (article_summaries):
All articles have passed a strict Phase 2 gate — only direct patient-level AE evidence is included.
Provide pmid + relevance_summary per article. 1–3 sentences: case count, dose/timing, clinical outcome. No titles or author names.
STRICT ISOLATED EXTRACTION: Every relevance_summary derives SOLELY from that article's abstract. Never borrow from another.
SELF-CHECK: Confirm no summary contains facts from a different PMID before submitting.
DEDUPLICATION: Duplicate = identical PMID or identical title only. Different papers on the same topic are NOT duplicates. Never write "(Duplicate...)" in any relevance_summary.

`summary_findings` — DETAILED RESEARCH ANALYSIS:

⚠️ ARTICLE-ONLY RULE: Every claim must be grounded in a specific retrieved article. Cite using [N] inline. Do NOT add information from outside the articles.
⛔ DRUG ISOLATION RULE: Every sentence refers only to the drug in `drug_name`. Never mention another drug.
⚠️ COVERAGE RULE: Every included article must be cited at least once by [N]. A summary that skips articles is incomplete.
⚠️ LENGTH RULE: This is the core output of the investigation. Generic sentences without [N] citations and specific effect sizes are not acceptable.

### Evidence Overview
2–3 sentences: how many articles were included, what outcomes they examined, and the dominant direction of evidence. Be specific (e.g., "14 of 16 articles reported protective associations across cancer, cardiovascular, and metabolic outcomes").

### Detailed Findings
Group articles by outcome category. For each category, use a bold header (e.g., **Cancer — Hepatocellular Carcinoma**) and then discuss every article in that group:
- Cite [N], state study design, population size, key effect size with CI, and the clinical conclusion in 1–2 sentences.
- Every [N] from the Literature section must appear here at least once.
- If two articles cover the same outcome, compare their findings directly.

### Conflicting or Risk Findings
Only if any articles showed increased risk or harm: discuss here with [N] citations and effect sizes. Omit entirely if none — do NOT write "none found."

### Bottom Line
2–3 sentences. Based on the weighted evidence:
- State the dominant finding clearly.
- If there is a net safety concern → say what and why.
- If the dominant finding is protective with no overall risk → state that no safety escalation is warranted.
Write as if explaining to a senior clinician. No jargon.

SOURCE ATTRIBUTION: disproportionality_source = "OpenFDA FAERS Database" or "Literature / PubMed (PMID: X)".

GUARDRAILS — call `abort_investigation` immediately. Write reason in the user's language, concise and actionable:
1. query_too_vague: No specific drug name identified.
2. multiple_drugs_detected: Two or more distinct drugs named.
3. non_medical_query: No connection to drug safety or pharmacovigilance.
4. drug_not_recognized: `get_drug_profile` returns source="fallback" AND name is clearly not a real pharmaceutical (gibberish, food, non-drug). Do NOT trigger for real drugs that failed API lookup.
5. no_literature_found: Both `fetch_pubmed_advanced` AND `search_drug_class_effects` returned audit.total_found=0 — meaning PubMed itself found zero matching articles.
   ⚠️ DO NOT trigger if count=0 but audit.total_found > 0: articles were fetched but excluded by LLM screening — adapt the query or widen investigation_context and retry once before aborting.
6. drug_not_in_portfolio: `query_knowledge_base` returns drug_in_formulary=false (always first call)."""


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
    # Pre-computed summaries from Phase 2 gate extraction in tools.py.
    # Each was extracted in an isolated single-article LLM call — zero cross-contamination.
    # Used with PRIORITY over LLM-batch-generated article_summaries.
    precomputed_summaries: dict[str, str] = {}  # pmid → pv_summary
    last_choice = None
    correction_injected = False  # guard: enforce report tools exactly once if agent stops early
    MAX_ITERATIONS = 20         # allows full investigation + report + submit
    # ── FAERS ROR tracking (Fix: bind calculate_disproportionality → discovered_events) ──
    # faers_pending: maps (a, b, c, d) tuple → ae_name for each fetch_fda_adverse_events result.
    # The LLM copies those same counts verbatim into calculate_disproportionality args,
    # so the tuple is a reliable fingerprint linking the two calls.
    faers_pending:  dict[tuple, str]  = {}   # (a,b,c,d) → ae_name
    faers_ror_cache: dict[str, dict]  = {}   # ae_name.lower() → {ror, ci_95, faers_significant}

    _loop_start = time.time()   # wall-clock guard — break before Vercel's 300s maxDuration

    for _ in range(MAX_ITERATIONS):

        # ── Time-budget guard: exit cleanly before Vercel kills the connection ──
        if time.time() - _loop_start > _LOOP_TIME_BUDGET:
            print(f"[ReAct loop] Time budget ({_LOOP_TIME_BUDGET}s) reached at iteration {_}. Breaking.")
            break

        # ── Reason: ask the LLM what to do next ──────────────────────────────
        # After a correction injection, force the LLM to call a tool (cannot write text again)
        _tool_choice = (
            "required"
            if correction_injected and report_markdown is None
            else "auto"
        )
        try:
            response = llm_client.chat.completions.create(
                model       = CHAT_MODEL,
                messages    = messages,
                tools       = TOOLS,
                tool_choice = _tool_choice,
            )
        except openai.RateLimitError as exc:
            print(f"[ReAct loop] OpenAI rate limit — breaking: {exc}")
            break
        except openai.BadRequestError as exc:
            # 400 — most likely context_length_exceeded; re-raise other 400s
            if "context" in str(exc).lower() or "length" in str(exc).lower():
                print(f"[ReAct loop] Context length exceeded — breaking: {exc}")
                break
            raise
        except (openai.APIConnectionError, openai.APITimeoutError) as exc:
            print(f"[ReAct loop] Network error reaching OpenAI — breaking: {exc}")
            break
        except openai.APIStatusError as exc:
            print(f"[ReAct loop] OpenAI API error {exc.status_code} — breaking: {exc}")
            break
        last_choice = response.choices[0]
        messages.append(last_choice.message)   # add assistant turn to history

        # Agent chose to reply with text only (no tool call)
        if last_choice.finish_reason == "stop" or not last_choice.message.tool_calls:
            # If the investigation is underway but report wasn't generated yet,
            # inject ONE correction and force a tool call on the next iteration.
            _has_investigated = any(
                s["module"] not in ("query_knowledge_base", "check_past_signals", "abort_investigation")
                for s in steps
            )
            if (
                not correction_injected
                and report_markdown is None
                and _has_investigated
                and _ < MAX_ITERATIONS - 2
            ):
                correction_injected = True
                messages.append({
                    "role": "user",
                    "content": (
                        "[SYSTEM ENFORCEMENT] You wrote a text response without calling tools. "
                        "This violates the MANDATORY REPORT TOOL CALL rule. "
                        "You MUST now call `generate_pharmacovigilance_report` immediately, "
                        "then call `submit_final_report`. Do NOT write any text — call the tools."
                    )
                })
                continue  # resume the loop with tool_choice="required"
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
                # Pre-computed summaries override LLM-batch values.
                # Each was extracted in an isolated single-article LLM call during
                # Phase 2 gate → structurally immune to cross-contamination.
                summaries_dict = {**llm_summaries_dict, **precomputed_summaries}
                lit_section, pmid_to_number = _build_literature_section(
                    collected_articles, summaries_dict, pubmed_audit_entries
                )
                fn_args["literature_section"] = lit_section
                if "summary_findings" in fn_args:
                    fn_args["summary_findings"] = _replace_pmid_citations(
                        fn_args["summary_findings"], pmid_to_number, valid_pmids
                    )

                # ── Inject tracked FAERS RORs into discovered_events ─────────────
                # The LLM often forgets to carry ror/ci_95/faers_significant from
                # calculate_disproportionality back into the discovered_events entries.
                # Python tracked every (fetch_fda_adverse_events → calculate_disproportionality)
                # chain, so we can fill the gaps deterministically here.
                if faers_ror_cache:
                    for ev in (fn_args.get("discovered_events") or []):
                        ae_key = ev.get("event_name", "").lower().strip()
                        if not ae_key or ev.get("ror") is not None:
                            continue  # already populated — skip
                        match = faers_ror_cache.get(ae_key)
                        if not match:
                            # Fuzzy fallback: substring match handles minor term differences
                            for k, v in faers_ror_cache.items():
                                if k in ae_key or ae_key in k:
                                    match = v
                                    break
                        if match:
                            ev["ror"]               = match["ror"]
                            ev["ci_95"]             = match["ci_95"]
                            ev["faers_significant"] = match["faers_significant"]

                # ── RC-1: Bucket 3 pre-report gate ────────────────────────────────
                # Block the report if any Candidate Unlabeled Signal still has no ROR
                # after the cache injection above. Enforces the mandatory FAERS step
                # without relying on the LLM to self-enforce it.
                _unquantified_b3 = [
                    ev.get("event_name", "unknown")
                    for ev in (fn_args.get("discovered_events") or [])
                    if (
                        ev.get("bucket") == "potentially_unlabeled"
                        and ev.get("ror") is None
                        and ev.get("direction") != "protective"  # protective findings don't need FAERS ROR
                    )
                ]
                if _unquantified_b3:
                    _blocked_result = {
                        "blocked": True,
                        "reason": (
                            "Report blocked — Candidate Unlabeled Signals without FAERS ROR: "
                            + ", ".join(_unquantified_b3)
                            + ". Run fetch_fda_adverse_events → calculate_disproportionality for each first."
                        ),
                    }
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(_blocked_result),
                    })
                    messages.append({
                        "role": "user",
                        "content": (
                            "[SYSTEM ENFORCEMENT — PRE-REPORT GATE] "
                            "generate_pharmacovigilance_report was BLOCKED. "
                            "These Candidate Unlabeled Signals have no FAERS disproportionality data yet: "
                            + ", ".join(_unquantified_b3)
                            + ". You MUST call fetch_fda_adverse_events → calculate_disproportionality "
                            "for EACH of them before retrying the report. "
                            "⚠️ DO NOT reclassify these AEs to confirmed_labeled to bypass this gate — "
                            "that would be a classification error. Their Bucket 3 status was determined "
                            "from article evidence; you must quantify them with FAERS data. Do this NOW."
                        ),
                    })
                    steps.append({
                        "module":   "generate_pharmacovigilance_report [BLOCKED — Bucket 3 FAERS missing]",
                        "prompt":   fn_args,
                        "response": _blocked_result,
                    })
                    continue  # skip dispatch; outer loop re-prompts the LLM

                # ── RC-3: Recommendation scrubber ─────────────────────────────────
                # Catch signal_level / recommendations contradiction before it is
                # rendered into the markdown. Overrides any "dismiss" language when
                # the composite signal is potential or significant.
                _sig_lvl = (fn_args.get("signal_level") or "none").lower()
                _reco    = (fn_args.get("recommendations") or "").lower()
                _dismiss_phrases = (
                    "no further action", "no action required", "no action needed",
                    "no action", "discard", "no novel signal",
                )
                if _sig_lvl in ("potential", "significant") and any(p in _reco for p in _dismiss_phrases):
                    _prefix = (
                        "Escalate to Safety Team — A statistically significant FAERS "
                        "disproportionality signal (ROR ≥ 2.0, lower CI > 1.0)"
                        if _sig_lvl == "significant"
                        else "Escalate to Safety Team — A Candidate Unlabeled Signal"
                    )
                    fn_args["recommendations"] = (
                        _prefix
                        + " was identified in this investigation. Per ICH E2D / CIOMS VI, "
                        "this finding cannot be dismissed without further regulatory review. "
                        "Recommended actions: (1) initiate targeted literature surveillance "
                        "for the identified adverse event; (2) evaluate adequacy of current "
                        "label coverage; (3) consider signal validation study or "
                        "pharmacoepidemiological analysis."
                    )

            try:
                result = dispatch(fn_name, fn_args)
            except Exception as exc:
                result = {"error": f"Tool execution failed: {exc}"}

            # ── Post-dispatch FAERS ROR tracking ─────────────────────────────────
            # Record fetch_fda_adverse_events result: map (a,b,c,d) → ae_name so
            # we can link it to the subsequent calculate_disproportionality call.
            if fn_name == "fetch_fda_adverse_events" and not result.get("error"):
                a = result.get("a")
                b = result.get("b")
                c = result.get("c")
                d = result.get("d")
                ae = fn_args.get("adverse_event", "").lower().strip()
                if all(v is not None for v in (a, b, c, d)) and ae:
                    faers_pending[(a, b, c, d)] = ae

            # Resolve calculate_disproportionality back to ae_name via the fingerprint.
            if fn_name == "calculate_disproportionality" and not result.get("error"):
                a = fn_args.get("cases_drug_event")
                b = fn_args.get("cases_drug_other")
                c = fn_args.get("cases_other_event")
                d = fn_args.get("cases_other_other")
                ae_name = faers_pending.pop((a, b, c, d), None)
                ror_val = result.get("ror")
                if ae_name and ror_val is not None:
                    faers_ror_cache[ae_name] = {
                        "ror":               ror_val,
                        "ci_95":             result.get("ci_95"),
                        "faers_significant": bool(result.get("statistically_significant")),
                    }

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
                        # Capture pre-computed summary from Phase 2 gate extraction.
                        # Generated one-article-at-a-time — zero cross-contamination.
                        if art.get("pv_summary"):
                            precomputed_summaries[pmid] = art["pv_summary"]

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
    # (includes graceful exit from time-budget guard)
    if not final_report:
        elapsed = time.time() - _loop_start
        if elapsed > _LOOP_TIME_BUDGET:
            last_text = (
                f"Investigation did not complete within the {_LOOP_TIME_BUDGET}s time budget "
                f"(elapsed: {elapsed:.0f}s). Try adding a specific focus area to narrow the query."
            )
        else:
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
