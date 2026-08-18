"""
tools.py — Tool definitions (schema for the LLM) and their implementations.

Architecture: ReAct-compatible atomic tools. Each tool is a pure function that
returns a JSON-serializable dict. The agent decides when and how many times to
call each tool based on its reasoning loop.

Tools:
  1.  get_drug_profile                  — resolve a drug name via OpenFDA → RxNorm
  2.  calculate_disproportionality      — ROR + 95% CI from a 2×2 table
  3.  fetch_fda_adverse_events          — OpenFDA FAERS fallback for 2×2 counts
  4.  fetch_pubmed_advanced             — PubMed literature search by clinical term
  5.  search_drug_class_effects         — PubMed search by drug class + adverse event
  6.  check_past_signals                — Supabase historical investigation lookup
  7.  query_knowledge_base              — Pinecone RAG over FDA safety documents
  8.  abort_investigation               — Terminates the ReAct loop on guardrail violation
  9.  generate_pharmacovigilance_report — Markdown report generator (CIOMS/ICH E2D)
  10. submit_final_report               — Terminates the ReAct loop normally
"""

import json
import math
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
import requests
from config import llm_client, pc, supabase, EMBED_MODEL, PINECONE_INDEX, CHAT_MODEL

OPENFDA_LABEL_URL  = "https://api.fda.gov/drug/label.json"
OPENFDA_FAERS_URL  = "https://api.fda.gov/drug/event.json"
RXNORM_BASE_URL    = "https://rxnav.nlm.nih.gov/REST"
HTTP_TIMEOUT       = 10   # seconds — increased for RxNorm reliability
FAERS_TIMEOUT      = 12   # FAERS API can be slower — needs a longer budget


# ── Tool Schema (sent to the LLM) ────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_drug_profile",
            "description": (
                "Retrieve the pharmacological profile of a drug: active ingredients, "
                "drug class, brand names, and mechanism of action. "
                "Call this early in every investigation to enable targeted searches "
                "by ingredient and by drug class. "
                "Data is fetched live from OpenFDA (primary) and RxNorm (fallback)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "drug_name": {
                        "type": "string",
                        "description": (
                            "Brand name or generic name of the drug to profile "
                            "(e.g. 'Warfarin', 'atorvastatin', 'Humira')."
                        )
                    }
                },
                "required": ["drug_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_disproportionality",
            "description": (
                "Calculate the Reporting Odds Ratio (ROR) and 95% Confidence Interval "
                "from a 2×2 contingency table. "
                "IMPORTANT: Only call this tool if you have extracted EXPLICIT numerical "
                "frequency counts directly from a published article, table, or database result. "
                "Do NOT estimate, infer, or fabricate any of the four cell counts. "
                "If the literature does not report exact case counts, skip this tool entirely "
                "and note in your reasoning that quantitative signal analysis was not possible "
                "due to missing frequency data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cases_drug_event": {
                        "type": "integer",
                        "description": "Count of reports with the TARGET drug AND the TARGET adverse event (cell a)."
                    },
                    "cases_drug_other": {
                        "type": "integer",
                        "description": "Count of reports with the TARGET drug AND OTHER adverse events (cell b)."
                    },
                    "cases_other_event": {
                        "type": "integer",
                        "description": "Count of reports with OTHER drugs AND the TARGET adverse event (cell c)."
                    },
                    "cases_other_other": {
                        "type": "integer",
                        "description": "Count of reports with OTHER drugs AND OTHER adverse events (cell d)."
                    }
                },
                "required": [
                    "cases_drug_event",
                    "cases_drug_other",
                    "cases_other_event",
                    "cases_other_other"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_fda_adverse_events",
            "description": (
                "Fallback Disproportionality Tool — query the OpenFDA FAERS (FDA Adverse Event "
                "Reporting System) API to dynamically retrieve real-world case counts for a "
                "drug-event pair and construct a 2×2 contingency table (a, b, c, d). "
                "WHEN TO CALL: Only invoke this tool if PubMed literature was retrieved but "
                "contained NO explicit numerical 2×2 frequency counts. "
                "Do NOT call if you already have explicit counts from literature. "
                "Do NOT fabricate counts — use only what this tool returns. "
                "After a successful call, pass the returned a/b/c/d values directly to "
                "`calculate_disproportionality`, tagging the source as 'OpenFDA FAERS Database'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "drug_name": {
                        "type": "string",
                        "description": (
                            "Generic or brand name of the drug to query in FAERS "
                            "(e.g. 'Atorvastatin', 'Warfarin', 'Humira'). "
                            "Use the same name resolved by get_drug_profile."
                        )
                    },
                    "adverse_event": {
                        "type": "string",
                        "description": (
                            "MedDRA Preferred Term (PT) for the adverse event "
                            "(e.g. 'Bradycardia', 'Rhabdomyolysis', 'Acute pancreatitis'). "
                            "Use clinically precise terminology."
                        )
                    }
                },
                "required": ["drug_name", "adverse_event"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_pubmed_advanced",
            "description": (
                "Fetch medical literature abstracts from PubMed via NCBI E-Utilities API. "
                "Use this to search for clinical trials, case reports, or literature evidence "
                "connecting a drug (or active ingredient) to an adverse event. "
                "ALWAYS include the drug name or active ingredient in the query_term."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query_term": {
                        "type": "string",
                        "description": (
                            "PubMed boolean search query — MUST use strict boolean grouping with parentheses. "
                            "Format: '\"drug_name\" AND (\"adverse_event_1\" OR \"adverse_event_2\" OR \"adverse_event_3\")'. "
                            "Multi-word terms MUST be double-quoted. "
                            "Good examples: "
                            "'\"sildenafil\" AND (\"myocardial infarction\" OR \"arrhythmia\" OR \"sudden cardiac death\")', "
                            "'\"sertraline\" AND (\"QTc prolongation\" OR \"cardiac arrhythmia\" OR \"torsades de pointes\")', "
                            "'\"warfarin\" AND (\"bleeding\" OR \"hemorrhage\" OR \"thrombocytopenia\")'. "
                            "BAD (never do this): 'sildenafil cardiovascular effects OR myocardial infarction OR stroke' — "
                            "without parentheses PubMed treats OR as global and returns 400K+ unrelated results. "
                            "ALWAYS include the drug name AND group adverse events with (... OR ...)."
                        )
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of articles to return (default: 10).",
                        "default": 10
                    },
                    "min_year": {
                        "type": "integer",
                        "description": "Minimum publication year to restrict results to recent literature (default: 2020).",
                        "default": 2020
                    },
                    "investigation_context": {
                        "type": "string",
                        "description": (
                            "Full investigation context passed to the LLM screener: include the drug name, "
                            "active ingredients, adverse event, and any demographic context from the user query "
                            "(e.g. 'Sertraline (sertraline hydrochloride, SSRI) — bruxism signal in adolescents'). "
                            "When provided, ALL matching articles are fetched and LLM-screened for relevance "
                            "before returning results. ALWAYS provide this parameter."
                        )
                    }
                },
                "required": ["query_term"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_drug_class_effects",
            "description": (
                "Search PubMed for literature regarding the entire pharmacological class "
                "to differentiate between a known class effect and a novel drug-specific "
                "safety signal. Call this after get_drug_profile to use the correct class name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "drug_class": {
                        "type": "string",
                        "description": (
                            "Pharmacological class name from the drug profile "
                            "(e.g. 'Vitamin K antagonist', 'HMG-CoA reductase inhibitor', 'SSRI')."
                        )
                    },
                    "adverse_event": {
                        "type": "string",
                        "description": (
                            "The adverse event to search for across the class "
                            "(e.g. 'pancreatitis', 'rhabdomyolysis', 'QT prolongation')."
                        )
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max articles to return (default: 10).",
                        "default": 10
                    },
                    "min_year": {
                        "type": "integer",
                        "description": "Minimum publication year filter (default: 2020).",
                        "default": 2020
                    },
                    "investigation_context": {
                        "type": "string",
                        "description": (
                            "Full investigation context for LLM screening: drug name, active ingredients, "
                            "adverse event, and demographic context. ALWAYS provide this parameter."
                        )
                    }
                },
                "required": ["drug_class", "adverse_event"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_past_signals",
            "description": (
                "Query internal historical investigation logs to check if this "
                "drug-event combination was previously analyzed by the platform. "
                "Use this to retrieve past verdicts and avoid re-escalating already-reviewed signals."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "drug_name": {
                        "type": "string",
                        "description": "Brand or generic name of the drug to look up in historical logs."
                    },
                    "adverse_event": {
                        "type": "string",
                        "description": "Optional: specific adverse event to filter historical records by."
                    }
                },
                "required": ["drug_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_knowledge_base",
            "description": (
                "Search the company's internal vector database (Pinecone) holding official "
                "FDA drug labels and safety documents. "
                "Use this to check whether a SPECIFIC adverse event or finding is already "
                "documented in the known safety profile. "
                "IMPORTANT: Call this tool MULTIPLE TIMES throughout the investigation — "
                "once for each specific adverse event or signal you discover in PubMed literature. "
                "Do NOT call it once with a generic query and assume you have the full picture. "
                "Example workflow: you find 'hepatotoxicity' in a PubMed abstract → call "
                "query_knowledge_base('Warfarin', 'hepatotoxicity') to check if it is already "
                "documented. If no relevant chunks are returned, this may be a novel signal. "
                "If it IS documented, it is not novel — discard it and move on."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "drug_name": {
                        "type": "string",
                        "description": "Name of the drug to query (e.g. 'Warfarin', 'metformin')."
                    },
                    "query": {
                        "type": "string",
                        "description": (
                            "The clinical phenomenon, side effect, or signal to search for "
                            "(e.g. 'acute pancreatitis', 'liver toxicity', 'QT prolongation')."
                        )
                    },
                    "section": {
                        "type": "string",
                        "description": "Optional metadata filter to restrict results to a specific label section.",
                        "enum": [
                            "adverse_reactions",
                            "warnings_and_cautions",
                            "contraindications",
                            "drug_interactions",
                            "boxed_warning"
                        ]
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of matching text chunks to return (default: 3).",
                        "default": 3
                    }
                },
                "required": ["drug_name", "query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "abort_investigation",
            "description": (
                "Immediately terminate the investigation when it CANNOT proceed. "
                "Call this INSTEAD OF submit_final_report when a guardrail condition is met. "
                "Do NOT call any other tools after this — it ends the ReAct loop."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "abort_code": {
                        "type": "string",
                        "enum": [
                            "drug_not_recognized",
                            "multiple_drugs_detected",
                            "non_medical_query",
                            "query_too_vague",
                            "no_literature_found",
                            "drug_not_in_portfolio"
                        ],
                        "description": (
                            "Machine-readable reason code: "
                            "'drug_not_recognized' — drug name does not exist in any medical registry; "
                            "'multiple_drugs_detected' — query mentions more than one drug name; "
                            "'non_medical_query' — query has no relation to pharmacovigilance; "
                            "'query_too_vague' — no drug name can be extracted from the prompt; "
                            "'no_literature_found' — PubMed returned 0 results for both drug and class searches; "
                            "'drug_not_in_portfolio' — query_knowledge_base returned drug_in_formulary=false — drug is not in the company pharmacovigilance portfolio."
                        )
                    },
                    "reason": {
                        "type": "string",
                        "description": "Clear, user-facing explanation of why the investigation was aborted."
                    }
                },
                "required": ["abort_code", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_pharmacovigilance_report",
            "description": (
                "Generate a standardized, executive-ready Pharmacovigilance Evaluation Report "
                "(Markdown format) summarizing all evidence, statistical ROR metrics, "
                "literature findings, and regulatory recommendations. "
                "Call this after completing your full investigation — before submit_final_report."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "drug_name": {
                        "type": "string",
                        "description": "Name of the drug under investigation."
                    },
                    "adverse_event": {
                        "type": "string",
                        "description": "The adverse event or safety signal being evaluated."
                    },
                    "ror": {
                        "type": "number",
                        "description": (
                            "Reporting Odds Ratio from calculate_disproportionality, "
                            "or null if no frequency data was available."
                        )
                    },
                    "ci_95": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": (
                            "95% Confidence Interval as [lower, upper], "
                            "or null if ROR was not calculated."
                        )
                    },
                    "is_significant": {
                        "type": "boolean",
                        "description": (
                            "STATISTICAL ONLY — set to true if and only if ROR ≥ 2.0 AND lower 95% CI > 1.0. "
                            "Set to false if ROR is below threshold or was not calculable. "
                            "Do NOT factor in literature evidence here — use signal_level for composite assessment."
                        )
                    },
                    "signal_level": {
                        "type": "string",
                        "enum": ["none", "potential", "significant"],
                        "description": (
                            "COMPOSITE signal classification based on BOTH FAERS and PubMed evidence. "
                            "This drives the master report header and subject table — set it carefully:\n\n"
                            "'significant' — FAERS ROR ≥ 2.0 with lower CI > 1.0. "
                            "The statistical threshold is met and constitutes a confirmed signal.\n\n"
                            "'potential'   — FAERS is negative or not calculable, BUT literature contains "
                            "novel, actionable case reports or clinical findings NOT already fully documented "
                            "in the FDA label / internal KB for this specific adverse event. "
                            "Use this whenever literature evidence warrants safety team review — "
                            "even a single unlabeled case report qualifies.\n\n"
                            "'none'        — Use ONLY when BOTH: (a) FAERS shows no disproportionality, "
                            "AND (b) no novel unlabeled findings were found in the literature. "
                            "If any novel finding exists, do NOT use 'none'."
                        )
                    },
                    "summary_findings": {
                        "type": "string",
                        "description": (
                            "Analytical interpretation structured in four markdown sections. "
                            "Use bullet points (`-`) to list individual items within each section — do NOT write long paragraphs.\n\n"
                            "SCOPE RULE: Every bullet in every section MUST directly concern the INVESTIGATED AE "
                            "or a clinically adjacent finding in the same organ system/mechanism. "
                            "Do NOT include findings from unrelated AE categories (e.g., suicidality warnings "
                            "when investigating bruxism) even if they appear in the same FDA label or KB chunk.\n\n"
                            "### Internal KB / FDA Label Baseline — what the label already documents about the investigated AE "
                            "or adjacent organ-system findings. Exclude unrelated warnings.\n"
                            "### Novel Findings — new AE findings not in the label. "
                            "Each bullet cites one finding with [PMID: XXXXXXXX] (auto-converted to [N]). "
                            "Only use PMIDs actually returned by PubMed tools.\n"
                            "### Known / Expected Findings — label-consistent findings RELATED TO THE INVESTIGATED AE to discard. "
                            "Exclude unrelated boxed warnings or contraindications.\n"
                            "### Signal Assessment — evidence quality, signal strength, confidence — scoped to investigated AE only.\n"
                            "Do NOT include article titles or author names."
                        )
                    },
                    "recommendations": {
                        "type": "string",
                        "description": (
                            "Regulatory action recommendation with rationale: "
                            "e.g. 'Escalate to Safety Team' with reason, or 'Discard' with reason."
                        )
                    },
                    "disproportionality_source": {
                        "type": "string",
                        "description": (
                            "Data provenance for the ROR calculation. "
                            "Set to 'Literature / PubMed (PMID: X)' when counts came from a published article, "
                            "or 'OpenFDA FAERS Database' when counts were fetched via fetch_fda_adverse_events. "
                            "Omit entirely if no ROR was calculated."
                        )
                    },
                    "article_summaries": {
                        "type": "array",
                        "description": (
                            "INCLUDE ALL articles returned by fetch_pubmed_advanced or search_drug_class_effects. "
                            "Every returned article has passed LLM screening — do NOT skip any. "
                            "ONLY use PMIDs actually returned by those tool calls — do NOT fabricate."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "pmid": {
                                    "type": "string",
                                    "description": "The PMID exactly as returned by the PubMed tool."
                                },
                                "relevance_summary": {
                                    "type": "string",
                                    "description": (
                                        "1-3 sentences: study design, population size (n=X), key finding with effect size and CI if reported, clinical outcome. "
                                        "No titles or author names. Derived SOLELY from that article's abstract — never borrow from another article."
                                    )
                                }
                            },
                            "required": ["pmid", "relevance_summary"]
                        }
                    },
                    "case_counts": {
                        "type": "object",
                        "description": (
                            "Raw 2×2 contingency table cell counts — pass when ROR was calculated so the "
                            "report displays the underlying absolute case numbers alongside the statistical metrics. "
                            "Use the a/b/c/d values returned by fetch_fda_adverse_events or extracted from literature."
                        ),
                        "properties": {
                            "a": {"type": "integer", "description": "Drug + target AE cases"},
                            "b": {"type": "integer", "description": "Drug + other AE cases (drug total minus a)"},
                            "c": {"type": "integer", "description": "Other drugs + target AE cases"},
                            "d": {"type": "integer", "description": "Background: other drugs + other AEs"}
                        }
                    }
                },
                "required": [
                    "drug_name",
                    "adverse_event",
                    "is_significant",
                    "summary_findings",
                    "recommendations"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "submit_final_report",
            "description": (
                "Submit the final structured investigation report and terminate the ReAct loop. "
                "Call this ONLY after generate_pharmacovigilance_report has been called. "
                "This is the last action in every investigation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "confidence_score": {
                        "type": "integer",
                        "description": (
                            "Confidence score 0–100 reflecting evidence quality and coverage. "
                            "≥ 70 required to submit."
                        )
                    },
                    "evidence_chain": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Ordered list of all tools called during the investigation "
                            "(e.g. ['query_knowledge_base', 'get_drug_profile', 'fetch_pubmed_advanced x2'])."
                        )
                    },
                    "signal_level": {
                        "type": "string",
                        "enum": ["drug-specific", "class-level", "no signal"],
                        "description": "Scope of the detected signal."
                    },
                    "novel_signal_detected": {
                        "type": "boolean",
                        "description": (
                            "True only if new literature reveals an adverse event "
                            "NOT already documented in the knowledge base."
                        )
                    },
                    "severe_events_found": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of specific adverse events identified in new literature. "
                            "Empty list if no novel events were found."
                        )
                    },
                    "reasoning": {
                        "type": "string",
                        "description": (
                            "Detailed comparison of new literature findings vs. known safety profile. "
                            "Cite specific abstracts and knowledge-base sections."
                        )
                    },
                    "recommended_action": {
                        "type": "string",
                        "enum": ["Escalate to Safety Team", "Discard - No Novel Signal"],
                        "description": "Final regulatory recommendation."
                    }
                },
                "required": [
                    "confidence_score",
                    "evidence_chain",
                    "signal_level",
                    "novel_signal_detected",
                    "severe_events_found",
                    "reasoning",
                    "recommended_action"
                ]
            }
        }
    },
]


# ── Tool Implementations ──────────────────────────────────────────────────────

def _get_profile_from_openfda(drug_name: str) -> dict | None:
    """
    Query OpenFDA drug label endpoint for pharmacological profile.
    Returns a structured dict on success, None if no results or API error.

    Tries two queries in order:
    1. Exact phrase match — works for simple names (e.g. "Warfarin" → "WARFARIN SODIUM")
    2. Unquoted term match — catches compound generic names (e.g. "Sildenafil" → "SILDENAFIL CITRATE")
    """
    name_lower = drug_name.lower()
    search_queries = [
        f'openfda.brand_name:"{drug_name}"+OR+openfda.generic_name:"{drug_name}"',
        f'openfda.brand_name:{name_lower}+OR+openfda.generic_name:{name_lower}',
    ]

    for search_query in search_queries:
        try:
            resp = requests.get(
                OPENFDA_LABEL_URL,
                params={"search": search_query, "limit": 1},
                timeout=HTTP_TIMEOUT
            )
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            continue

        results = data.get("results", [])
        if not results:
            continue

        label   = results[0]
        openfda = label.get("openfda", {})

        # Active ingredients
        active_ingredients = (
            openfda.get("substance_name")
            or openfda.get("generic_name")
            or [drug_name.lower()]
        )

        # Drug class — prefer EPC (established pharmacologic class), fall back to CS
        drug_class = (
            openfda.get("pharm_class_epc")
            or openfda.get("pharm_class_cs")
            or ["Unknown / Unclassified"]
        )

        # Brand names
        brand_names = openfda.get("brand_name", [drug_name])

        # Mechanism — prefer clinical_pharmacology, fall back to description
        mechanism_raw = (
            label.get("clinical_pharmacology", [])
            or label.get("description", [])
        )
        mechanism = mechanism_raw[0][:800] if mechanism_raw else "Not available in label."

        return {
            "drug_name":          drug_name,
            "active_ingredients": [i.lower() for i in active_ingredients],
            "drug_class":         drug_class if isinstance(drug_class, str) else drug_class[0],
            "brand_names":        brand_names,
            "mechanism":          mechanism,
            "source":             "openfda"
        }

    return None


def _get_profile_from_rxnorm(drug_name: str) -> dict | None:
    """
    Fallback: query RxNorm (NIH) for drug identity and class.
    Returns a structured dict on success, None on failure.
    """
    try:
        # Step 1 — resolve drug name to RxCUI
        r = requests.get(
            f"{RXNORM_BASE_URL}/rxcui.json",
            params={"name": drug_name},
            timeout=HTTP_TIMEOUT
        )
        r.raise_for_status()
        rxcui_list = r.json().get("idGroup", {}).get("rxnormId", [])
        if not rxcui_list:
            return None
        rxcui = rxcui_list[0]

        # Step 2 — get concept properties (canonical name)
        props_resp = requests.get(
            f"{RXNORM_BASE_URL}/rxcui/{rxcui}/properties.json",
            timeout=HTTP_TIMEOUT
        )
        props_resp.raise_for_status()
        concept = props_resp.json().get("properties", {})
        canonical_name = concept.get("name", drug_name)

        # Step 3 — get drug class via FDA Structured Product Label relation
        drug_class = "Unknown / Unclassified"
        try:
            cls_resp = requests.get(
                f"{RXNORM_BASE_URL}/rxclass/class/byRxcui.json",
                params={"rxcui": rxcui, "relaSource": "FDASPL"},
                timeout=HTTP_TIMEOUT
            )
            cls_resp.raise_for_status()
            class_list = (
                cls_resp.json()
                .get("rxclassDrugInfoList", {})
                .get("rxclassDrugInfo", [])
            )
            if class_list:
                drug_class = (
                    class_list[0]
                    .get("rxclassMinConceptItem", {})
                    .get("className", "Unknown / Unclassified")
                )
        except Exception:
            pass

        return {
            "drug_name":          drug_name,
            "active_ingredients": [canonical_name.lower()],
            "drug_class":         drug_class,
            "brand_names":        [drug_name],
            "mechanism":          "Mechanism not available via RxNorm.",
            "source":             "rxnorm",
            "rxcui":              rxcui
        }

    except Exception:
        return None


def calculate_disproportionality(
    cases_drug_event: int,
    cases_drug_other: int,
    cases_other_event: int,
    cases_other_other: int
) -> dict:
    """
    Calculate Reporting Odds Ratio (ROR) and 95% CI from a 2×2 contingency table.

        |               | Target AE | Other AEs |
        |---------------|-----------|-----------|
        | Target drug   |     a     |     b     |
        | Other drugs   |     c     |     d     |

    ROR = (a/b) / (c/d) = (a*d) / (b*c)
    """
    a, b, c, d = (
        cases_drug_event,
        cases_drug_other,
        cases_other_event,
        cases_other_other
    )

    # Validate inputs
    if any(x < 0 for x in [a, b, c, d]):
        return {"error": "All cell counts must be >= 0.", "ror": None, "ci_95": None}

    # Any zero in a denominator cell makes ROR undefined
    if a == 0 or b == 0 or c == 0 or d == 0:
        return {
            "ror":                      None,
            "ci_95":                    None,
            "statistically_significant": False,
            "interpretation": (
                f"ROR is undefined: one or more cells are zero "
                f"(a={a}, b={b}, c={c}, d={d}). "
                "Consider applying Haldane-Anscombe correction (+0.5 to all cells) "
                "if zero counts are expected."
            )
        }

    ror      = (a * d) / (b * c)
    se       = math.sqrt(1/a + 1/b + 1/c + 1/d)
    ln_ror   = math.log(ror)
    ci_lower = math.exp(ln_ror - 1.96 * se)
    ci_upper = math.exp(ln_ror + 1.96 * se)

    significant = bool(ror >= 2.0 and ci_lower > 1.0)

    return {
        "ror":   round(ror, 2),
        "ci_95": [round(ci_lower, 2), round(ci_upper, 2)],
        "statistically_significant": significant,
        "interpretation": (
            "Statistically significant safety signal detected (ROR ≥ 2.0 and lower CI > 1.0)."
            if significant else
            "No statistically significant safety signal detected (within expected background noise)."
        )
    }


PUBMED_ESEARCH_URL   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH_URL    = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PUBMED_FETCH_TIMEOUT  = 20   # seconds — longer budget for efetch XML response
SCREENING_BATCH_SIZE  = 50   # articles per LLM screening call
MAX_PUBMED_SCREEN     = 50   # cap on articles fetched when screening is active


def _screen_articles_llm(
    articles: list[dict],
    investigation_context: str,
) -> list[dict]:
    """
    Two-phase LLM pipeline for article screening and summarization.

    Phase 1 — Batch relevance screening: sends articles in batches of SCREENING_BATCH_SIZE
    to quickly identify relevant PMIDs. Fails open (includes all) on error.

    Phase 2 — Per-article isolated extraction: for each relevant article, one dedicated
    LLM call extracts tier + summary from ONLY that article's abstract.
    One call per article = ZERO cross-contamination between papers.

    Pre-computed summary is stored on each article dict as:
        art["pv_summary"] — concise analytical extraction
    Used by agent.py with priority over LLM-batch-generated summaries.
    """
    if not articles or not llm_client:
        return articles

    # ── Phase 1: Batch relevance screening ───────────────────────────────────
    relevant: list[dict] = []

    for batch_start in range(0, len(articles), SCREENING_BATCH_SIZE):
        batch = articles[batch_start : batch_start + SCREENING_BATCH_SIZE]

        articles_block = ""
        for idx, art in enumerate(batch, 1):
            abstract_snippet = (art.get("abstract") or "")[:1000]
            articles_block += (
                f"\n[{idx}] PMID: {art.get('pmid', 'N/A')}\n"
                f"Title: {art.get('title', '')}\n"
                f"Abstract: {abstract_snippet}\n"
            )

        prompt = (
            "You are a strict Pharmacovigilance Literature Screener.\n\n"
            f"Investigation Context:\n{investigation_context}\n\n"
            f"Review the following {len(batch)} candidate articles. "
            "Apply HIGH selectivity: only include articles with substantive, "
            "direct pharmacovigilance value for the specific drug and adverse event in the investigation context.\n\n"
            "INCLUDE only if ALL of the following are true:\n"
            "- The specific drug (not just its class) is a PRIMARY subject of the study\n"
            "- The article reports an original safety finding, adverse event signal, or case directly relevant to the investigation\n"
            "- The study population involves humans (clinical trial, cohort, case report, pharmacovigilance database)\n"
            "- The finding is substantive and not already a well-known expected effect\n\n"
            "EXCLUDE if ANY of the following applies:\n"
            "- Animal study, in-vitro experiment, or computational/mechanistic study with no direct human data\n"
            "- The investigated drug is only a comparator, secondary agent, or briefly mentioned — not the primary subject\n"
            "- Efficacy study with no safety/adverse event data relevant to the investigation\n"
            "- Narrative review, editorial, commentary, or letter with no original data\n"
            "- The drug is mentioned only in passing with no drug-specific outcome data\n"
            "- The study's adverse events are entirely unrelated to the investigated adverse event\n\n"
            f"Articles:\n{articles_block}\n\n"
            "Respond with a JSON array for ONLY the relevant articles:\n"
            '[{"pmid": "12345678", "reason": "One sentence explaining direct relevance to the drug and adverse event"}, ...]\n'
            "If none are relevant, return: []\n"
            "Return ONLY valid JSON — no explanation, no markdown."
        )

        try:
            resp = llm_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            raw = resp.choices[0].message.content.strip()
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:].strip()
            screened = json.loads(raw)
            relevant_pmids = {
                item["pmid"] for item in screened
                if isinstance(item, dict) and "pmid" in item
            }
            relevant.extend(a for a in batch if a.get("pmid") in relevant_pmids)
        except Exception as exc:
            print(f"[PubMed screening error — batch {batch_start // SCREENING_BATCH_SIZE + 1}] {exc}")
            relevant.extend(batch)  # fail-open

    # ── Phase 2: Per-article isolated summary extraction ─────────────────────
    _extract_article_summaries(relevant, investigation_context)

    # ── Post-Phase-2 filter: drop articles where Phase 2 found no drug-specific data ──
    # Phase 2 writes "NO_DRUG_DATA" when the investigated drug is not a primary subject.
    # These articles passed Phase 1 by mistake and must not appear in the final report.
    relevant = [
        a for a in relevant
        if (a.get("pv_summary") or "") != "NO_DRUG_DATA"
    ]

    return relevant


def _extract_article_summaries(
    articles: list[dict],
    investigation_context: str,
) -> None:
    """
    Extract a relevance_summary for each article INDIVIDUALLY.

    One dedicated LLM call per article — the model sees ONLY that article's
    abstract, title, and PMID. This architectural isolation makes cross-
    contamination between articles structurally impossible.

    Mutates each article dict in-place:
        art["pv_summary"] — concise analytical extraction (1-3 sentences)

    Failures are silent and non-blocking.
    """
    if not articles or not llm_client:
        return

    for art in articles:
        pmid     = art.get("pmid", "")
        title    = art.get("title", "Unknown title")
        abstract = (art.get("abstract") or "")[:2000]

        prompt = (
            "You are a Pharmacovigilance Literature Analyst.\n\n"
            f"Investigation Context:\n{investigation_context}\n\n"
            "Write a concise analytical summary (1-3 sentences) for the article below.\n"
            "Include: study design, population size (n=X), key finding with effect size and CI if reported, clinical outcome.\n"
            "Do NOT include the article title or author names.\n\n"
            "=== CRITICAL ISOLATION RULE ===\n"
            "Your summary must be derived EXCLUSIVELY from the single abstract below.\n"
            "NEVER include information from any other source, memory, or prior article.\n"
            "SELF-CHECK: Does my summary contain ANY fact not present in the abstract below? If yes — remove it.\n"
            "================================\n\n"
            "=== DRUG ISOLATION RULE ===\n"
            "The investigation concerns ONLY the primary drug named in the Investigation Context above.\n"
            "TWO SCENARIOS:\n"
            "A) The investigated drug IS the primary subject of this article → always write a meaningful summary. "
            "Report the study design, population, and main finding for that drug — even if exact statistics are not available. "
            "Do NOT report effect sizes or outcomes for other drugs mentioned in the same abstract.\n"
            "B) The investigated drug is NOT a primary subject (e.g., it only appears as a comparator, a brief mention, or is not named at all) → write exactly: 'NO_DRUG_DATA'\n"
            "===========================\n\n"
            f"=== ARTICLE | PMID: {pmid} ===\n"
            f"Title: {title}\n"
            f"Abstract: {abstract}\n"
            f"=== END ARTICLE | PMID: {pmid} ===\n\n"
            "Respond with JSON only (no explanation, no markdown fences):\n"
            '{"summary": "your extraction here"}'
        )

        try:
            resp = llm_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            raw = resp.choices[0].message.content.strip()
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:].strip()
            extracted = json.loads(raw)
            if isinstance(extracted, dict):
                summary = extracted.get("summary", "").strip()
                if summary and summary != "NO_DRUG_DATA":
                    art["pv_summary"] = summary
                elif summary == "NO_DRUG_DATA":
                    art["pv_summary"] = "NO_DRUG_DATA"   # used by post-Phase-2 filter
        except Exception as exc:
            print(f"[Article summary extraction error — PMID {pmid}] {exc}")


def _pubmed_fetch(term: str, max_results: int, min_year: int = 2020) -> tuple[list[dict], dict]:
    """
    Two-step PubMed retrieval with configurable date range.

    Step 1 — esearch retmax=0  → get total matching count
    Step 2 — esearch retmax=N  → get top-N PMIDs by relevance (N = min(count, max_results))
    Step 3 — efetch            → parse full article records

    Returns:
        articles   — list of structured article dicts
        audit_info — {date_range, total_found, total_fetched}
    """
    dated_term = f"{term} AND {min_year}:3000[dp]"
    base_params = {
        "db":      "pubmed",
        "term":    dated_term,
        "retmode": "json",
        "sort":    "pub+date",  # newest first — pharmacovigilance monitoring prioritizes recency
    }

    # Step 1: count only
    count_resp = requests.get(
        PUBMED_ESEARCH_URL,
        params={**base_params, "retmax": 0},
        timeout=HTTP_TIMEOUT,
    )
    count_resp.raise_for_status()
    total_count = int(count_resp.json().get("esearchresult", {}).get("count", 0))

    audit_info: dict = {
        "date_range":    f"{min_year} – present",
        "total_found":   total_count,
        "total_fetched": 0,
    }

    if total_count == 0:
        return [], audit_info

    # Step 2: fetch top PMIDs by relevance
    fetch_n = min(total_count, max_results)
    id_resp = requests.get(
        PUBMED_ESEARCH_URL,
        params={**base_params, "retmax": fetch_n},
        timeout=HTTP_TIMEOUT,
    )
    id_resp.raise_for_status()
    id_list = id_resp.json().get("esearchresult", {}).get("idlist", [])
    if not id_list:
        return [], audit_info

    # Step 3: efetch full records
    fetch_resp = requests.get(
        PUBMED_EFETCH_URL,
        params={"db": "pubmed", "id": ",".join(id_list), "retmode": "xml", "rettype": "xml"},
        timeout=PUBMED_FETCH_TIMEOUT,
    )
    fetch_resp.raise_for_status()
    root = ET.fromstring(fetch_resp.content)

    articles = []
    for pub in root.findall(".//PubmedArticle"):
        medline = pub.find("MedlineCitation")
        if medline is None:
            continue

        pmid_elem = medline.find("PMID")
        pmid = pmid_elem.text if pmid_elem is not None else ""

        article = medline.find("Article")
        if article is None:
            continue

        title_elem = article.find("ArticleTitle")
        title = (title_elem.text or "") if title_elem is not None else ""

        abstract_parts = [at.text or "" for at in article.findall(".//AbstractText")]
        abstract = " ".join(abstract_parts)  # full abstract — no truncation

        journal_elem = article.find("Journal/Title")
        journal = journal_elem.text if journal_elem is not None else ""

        pubdate_elem = article.find(".//PubDate")
        if pubdate_elem is not None:
            year  = pubdate_elem.findtext("Year", "")
            month = pubdate_elem.findtext("Month", "")
            pubdate = f"{year} {month}".strip() if year else pubdate_elem.findtext("MedlineDate", "")
        else:
            pubdate = ""

        authors = []
        for author in article.findall(".//Author")[:5]:
            last = author.findtext("LastName", "")
            fore = author.findtext("ForeName", "")
            if last:
                authors.append(f"{last} {fore}".strip())

        articles.append({
            "pmid":     pmid,
            "title":    title,
            "pubdate":  pubdate,
            "source":   journal,
            "abstract": abstract,
            "authors":  authors,
        })

    audit_info["total_fetched"] = len(articles)
    return articles, audit_info


def fetch_pubmed_advanced(
    query_term: str,
    max_results: int = 10,
    min_year: int = 2020,
    investigation_context: str | None = None,
) -> dict:
    """
    Search PubMed for recent literature by clinical query term.
    When investigation_context is provided, fetches up to MAX_PUBMED_SCREEN articles
    and runs LLM relevance screening before returning results.
    Returns structured article records with full abstract text and audit metadata.
    """
    try:
        fetch_limit = MAX_PUBMED_SCREEN if investigation_context else max_results
        articles, audit = _pubmed_fetch(query_term, fetch_limit, min_year)

        if investigation_context and articles:
            articles = _screen_articles_llm(articles, investigation_context)
            audit["total_relevant"] = len(articles)

        return {
            "query_term": query_term,
            "count":      len(articles),
            "results":    articles,
            "audit":      audit,
        }
    except Exception:
        return {
            "query_term": query_term,
            "count":      0,
            "results":    [],
            "audit":      {"date_range": f"{min_year} – present", "total_found": 0, "total_fetched": 0},
        }


def search_drug_class_effects(
    drug_class: str,
    adverse_event: str,
    max_results: int = 10,
    min_year: int = 2020,
    investigation_context: str | None = None,
) -> dict:
    """
    Search PubMed for adverse effects across an entire pharmacological class.
    When investigation_context is provided, runs LLM screening on all fetched articles.
    Used to determine whether a signal is a known class effect or drug-specific.
    """
    try:
        term = f'"{drug_class}" AND "{adverse_event}"'
        fetch_limit = MAX_PUBMED_SCREEN if investigation_context else max_results
        articles, audit = _pubmed_fetch(term, fetch_limit, min_year)

        if investigation_context and articles:
            articles = _screen_articles_llm(articles, investigation_context)
            audit["total_relevant"] = len(articles)

        return {
            "drug_class":    drug_class,
            "adverse_event": adverse_event,
            "count":         len(articles),
            "results":       articles,
            "audit":         audit,
        }
    except Exception:
        return {
            "drug_class":    drug_class,
            "adverse_event": adverse_event,
            "count":         0,
            "results":       [],
            "audit":         {"date_range": f"{min_year} – present", "total_found": 0, "total_fetched": 0},
        }


def get_drug_profile(drug_name: str) -> dict:
    """
    Resolve a drug name into its pharmacological identity.
    Pipeline: OpenFDA → RxNorm → safety fallback.
    Always returns a JSON-serializable dict.
    """
    # Primary: OpenFDA
    result = _get_profile_from_openfda(drug_name)
    if result:
        return result

    # Fallback: RxNorm
    result = _get_profile_from_rxnorm(drug_name)
    if result:
        return result

    # Safety fallback — never raises, always returns something usable
    return {
        "drug_name":          drug_name,
        "active_ingredients": [drug_name.lower()],
        "drug_class":         "Unknown / Unclassified",
        "brand_names":        [drug_name],
        "mechanism":          "Information unavailable via automated drug registries.",
        "source":             "fallback"
    }


def fetch_fda_adverse_events(drug_name: str, adverse_event: str) -> dict:
    """
    Query OpenFDA FAERS to build a 2×2 contingency table for a drug-event pair.

    Table structure:
        |               | Target AE | Other AEs |
        |---------------|-----------|-----------|
        | Target drug   |     a     |     b     |
        | Other drugs   |     c     |     d     |

    Derivation:
        a = reports with drug AND event
        b = total_drug - a          (drug WITHOUT this event)
        c = total_event - a         (other drugs WITH this event)
        d = total_all - total_drug - total_event + a   (background)

    Returns None for all counts if the API is unreachable or drug/event not found.
    """
    def _count(search: str | None = None) -> int:
        """Fetch only the total result count from FAERS — does not download records."""
        params: dict = {"limit": 1}
        if search:
            params["search"] = search
        try:
            resp = requests.get(OPENFDA_FAERS_URL, params=params, timeout=FAERS_TIMEOUT)
            if resp.status_code == 404:
                return 0
            resp.raise_for_status()
            return resp.json().get("meta", {}).get("results", {}).get("total", 0)
        except requests.exceptions.Timeout:
            raise TimeoutError("OpenFDA FAERS API timed out.")

    def _count_by(field: str, search: str, limit: int = 10) -> list:
        """Fetch a count breakdown by field from FAERS (non-blocking — returns [] on any failure)."""
        try:
            resp = requests.get(
                OPENFDA_FAERS_URL,
                params={"search": search, "count": field, "limit": limit},
                timeout=FAERS_TIMEOUT
            )
            if resp.status_code == 404:
                return []
            resp.raise_for_status()
            return resp.json().get("results", [])
        except Exception:
            return []

    try:
        drug_q  = f'patient.drug.medicinalproduct:"{drug_name}"'
        event_q = f'patient.reaction.reactionmeddrapt:"{adverse_event}"'

        a           = _count(f"{drug_q} AND {event_q}")
        total_drug  = _count(drug_q)
        total_event = _count(event_q)
        total_all   = _count()           # full FAERS corpus — no filter

        if total_drug == 0:
            return {
                "a": None, "b": None, "c": None, "d": None,
                "source":       "OpenFDA FAERS API",
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "error":        (
                    f"Drug '{drug_name}' was not found in FAERS. "
                    "Try the generic active ingredient name instead."
                )
            }

        b = max(0, total_drug  - a)
        c = max(0, total_event - a)
        d = max(0, total_all   - total_drug - total_event + a)

        # ── Demographics — non-blocking (failures return empty structures) ────────
        sex_map = {"0": "Unknown", "1": "Male", "2": "Female"}
        age_map = {
            "1": "Neonate (0–27 days)", "2": "Infant (28 days–23 months)",
            "3": "Child (2–11 years)",  "4": "Adolescent (12–17 years)",
            "5": "Adult (18–64 years)", "6": "Elderly (65+ years)"
        }

        sex_raw  = _count_by("patient.patientsex",         drug_q)
        age_raw  = _count_by("patient.patientagegroup",    drug_q)
        drug_raw = _count_by("patient.drug.medicinalproduct.exact", drug_q, limit=10)

        demographics_sex = {
            sex_map.get(str(r["term"]), str(r["term"])): r["count"]
            for r in sex_raw
        } if sex_raw else {}

        demographics_age = {
            age_map.get(str(r["term"]), str(r["term"])): r["count"]
            for r in age_raw
        } if age_raw else {}

        drug_name_lower = drug_name.lower()
        concomitant_drugs = [
            {"drug": r["term"], "count": r["count"]}
            for r in drug_raw
            if r.get("term", "").lower() != drug_name_lower
        ][:5]

        return {
            "a":            a,
            "b":            b,
            "c":            c,
            "d":            d,
            "drug_name":    drug_name,
            "adverse_event": adverse_event,
            "source":       "OpenFDA FAERS API",
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "counts_note":  (
                f"a={a} (drug+event), b={b} (drug, other events), "
                f"c={c} (other drugs, this event), d={d} (background)"
            ),
            "demographics": {
                "sex_distribution":    demographics_sex,
                "age_groups":          demographics_age,
                "top_concomitant_drugs": concomitant_drugs
            }
        }

    except TimeoutError as e:
        return {
            "a": None, "b": None, "c": None, "d": None,
            "source":       "OpenFDA FAERS API",
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "error":        str(e)
        }
    except Exception as exc:
        return {
            "a": None, "b": None, "c": None, "d": None,
            "source":       "OpenFDA FAERS API",
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "error":        f"FAERS query failed: {exc}"
        }


def check_past_signals(drug_name: str, adverse_event: str = None) -> dict:
    """
    Query the Supabase `past_signals` table for historical investigations
    of a specific drug-event combination.
    Returns previous verdicts, ROR values, and investigation metadata.
    """
    if not supabase:
        return {
            "status":       "unavailable",
            "message":      "Database client is not configured.",
            "count":        0,
            "past_signals": []
        }

    try:
        query = (
            supabase.table("past_signals")
            .select("investigation_id, created_at, drug_name, adverse_event, verdict, previous_ror")
            .ilike("drug_name", f"%{drug_name}%")
        )

        if adverse_event:
            query = query.ilike("adverse_event", f"%{adverse_event}%")

        result = query.order("created_at", desc=True).execute()
        records = result.data or []

        return {
            "drug_name":     drug_name,
            "adverse_event": adverse_event,
            "count":         len(records),
            "past_signals":  records
        }

    except Exception as exc:
        error_msg = str(exc)
        # Graceful handling if the table doesn't exist yet
        if "past_signals" in error_msg and ("does not exist" in error_msg or "relation" in error_msg):
            return {
                "count":        0,
                "past_signals": [],
                "note":         "past_signals table not yet created. No prior investigations on record."
            }
        return {
            "status":       "error",
            "message":      error_msg,
            "count":        0,
            "past_signals": []
        }


def query_knowledge_base(
    drug_name: str,
    query: str,
    section: str = None,
    top_k: int = 3
) -> dict:
    """
    Semantic search over the Pinecone knowledge base.
    Returns the top_k most relevant chunks from the company's FDA safety documents.
    """
    if not llm_client or not pc:
        return {
            "status":  "unavailable",
            "message": "Vector database or embedding client is not initialized.",
            "chunks":  []
        }

    try:
        # Embed the clinical query
        embedding = llm_client.embeddings.create(
            input=[f"{drug_name} {query}"],
            model=EMBED_MODEL
        ).data[0].embedding

        # Build metadata filter — must match the field name used in seed_db.py
        metadata_filter: dict = {"search_name": drug_name.lower()}
        if section:
            metadata_filter["section"] = section

        index = pc.Index(PINECONE_INDEX)

        # Query with drug-specific filter
        filtered_result = index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True,
            filter=metadata_filter
        )

        # drug_in_formulary = True only if the filtered query found results for this specific drug
        drug_in_formulary = bool(filtered_result.matches)

        if not drug_in_formulary:
            return {
                "drug":              drug_name,
                "query":             query,
                "found":             0,
                "drug_in_formulary": False,
                "chunks":            [],
                "message":           f"'{drug_name}' has no prior documentation in the internal knowledge base. Any retrieved literature findings may represent novel signals."
            }

        chunks = [
            {
                "drug":    m.metadata.get("drug_name", drug_name),
                "section": m.metadata.get("section", "unknown"),
                "text":    m.metadata.get("text", "")[:600],
                "score":   round(m.score, 3)
            }
            for m in filtered_result.matches
        ]

        return {
            "drug":              drug_name,
            "query":             query,
            "found":             len(chunks),
            "drug_in_formulary": True,
            "chunks":            chunks
        }

    except Exception as exc:
        return {
            "status":  "error",
            "message": str(exc),
            "drug":    drug_name,
            "chunks":  []
        }


def abort_investigation(abort_code: str, reason: str) -> dict:
    """
    Terminates the investigation when a guardrail condition is met.
    Called instead of submit_final_report for unrecoverable input problems.
    """
    return {
        "status":     "aborted",
        "abort_code": abort_code,
        "reason":     reason
    }


def generate_pharmacovigilance_report(
    drug_name: str,
    adverse_event: str,
    is_significant: bool,
    summary_findings: str,
    recommendations: str,
    ror: float | None = None,
    ci_95: list[float] | None = None,
    disproportionality_source: str | None = None,
    literature_section: str | None = None,    # Python-injected by agent.py — NOT from LLM
    case_counts: dict | None = None,          # raw 2×2 a/b/c/d values for transparent reporting
    signal_level: str | None = None,          # composite: "none" | "potential" | "significant"
) -> dict:
    """
    Generate a standardized Pharmacovigilance Evaluation Report in Markdown.
    Follows CIOMS/ICH E2D executive report structure.

    signal_level drives the master header and Subject table:
      "significant" → 🔴  (FAERS ROR threshold met)
      "potential"   → 🟡  (FAERS negative but novel Tier 1 literature found)
      "none" / None → 🟢  (both sources show no signal)

    is_significant is used exclusively for the FAERS statistics sub-section.
    """
    generated_at = datetime.now(timezone.utc).isoformat()

    # ── Composite badge (header + Subject table) ──────────────────────────────
    _level = (signal_level or "none").strip().lower()
    if _level == "significant":
        composite_badge = "🔴 SIGNIFICANT SAFETY SIGNAL"
    elif _level == "potential":
        composite_badge = "🟡 POTENTIAL SIGNAL — REQUIRES SAFETY REVIEW"
    else:
        composite_badge = "🟢 NO SIGNIFICANT SIGNAL"

    # ── FAERS-specific status row (statistics section only) ───────────────────
    if is_significant:
        faers_row_badge = "🔴 Significant — ROR ≥ 2.0 and lower CI > 1.0"
    elif ror is not None:
        faers_row_badge = "🟢 Not Significant"
    else:
        faers_row_badge = "⬜ Not Calculable — No usable frequency data available"

    # ── Literature signal row (Subject table) ─────────────────────────────────
    if _level == "potential":
        lit_badge = "🟡 Actionable case reports identified — potential unlabeled risk"
    elif _level == "significant":
        lit_badge = "🔴 Novel findings consistent with or confirming statistical signal"
    else:
        lit_badge = "🟢 No novel actionable findings"

    if ror is not None:
        ci_str     = f"(95% CI: {ci_95[0]}, {ci_95[1]})" if ci_95 else "(95% CI: N/A)"

        table_rows = (
            f"| **Calculated ROR** | {ror} {ci_str} |\n"
            f"| **Statistical Signal (FAERS)** | {faers_row_badge} |"
        )

        if case_counts:
            a = case_counts.get("a", "N/A")
            b = case_counts.get("b", "N/A")
            c = case_counts.get("c", "N/A")
            d = case_counts.get("d", "N/A")
            table_rows += (
                f"\n| **Cases: Drug + AE (a)** | {a} |"
                f"\n| **Cases: Drug + Other AEs (b)** | {b} |"
                f"\n| **Cases: Other Drugs + AE (c)** | {c} |"
                f"\n| **Background Cases (d)** | {d} |"
            )

        if disproportionality_source:
            table_rows += f"\n| **Data Source** | {disproportionality_source} |"

        faers_note = ""
        if disproportionality_source and "FAERS" in disproportionality_source:
            faers_note = (
                "\n> **Note:** Literature lacked quantitative 2×2 contingency figures. "
                "Real-world pharmacovigilance data was retrieved directly from OpenFDA FAERS "
                "to enable disproportionality analysis.\n"
            )

        stats_section = (
            "## Statistical Disproportionality Analysis\n\n"
            "| Metric / Parameter | Value / Data |\n"
            "| :--- | :--- |\n"
            f"{table_rows}\n"
            "\n> Significance criterion: ROR ≥ 2.0 AND lower 95% CI > 1.0\n"
            f"{faers_note}"
        )
    else:
        stats_section = (
            "## Statistical Disproportionality Analysis\n\n"
            f"| Metric / Parameter | Value / Data |\n"
            f"| :--- | :--- |\n"
            f"| **Statistical Signal (FAERS)** | {faers_row_badge} |\n"
            "\n> ROR not calculated — no explicit 2×2 frequency counts were found in the retrieved "
            "literature, and no usable data was available from OpenFDA FAERS for this drug-event pair.\n"
        )

    report_markdown = f"""# VigiLenseAI — Pharmacovigilance Evaluation Report

**Report Date:** {generated_at}
**Status:** {composite_badge}

---

## Subject

| Field | Value |
|-------|-------|
| **Drug Under Investigation** | {drug_name} |
| **Adverse Event** | {adverse_event} |
| **Overall Signal Classification** | {composite_badge} |
| **Statistical Signal (FAERS)** | {faers_row_badge} |
| **Literature Signal (PubMed)** | {lit_badge} |

---

{stats_section}
---

## Literature Retrieved from PubMed

{literature_section or "*No relevant published literature was retrieved from PubMed during this investigation.*"}

---

## Agent Analysis

{summary_findings}

---

## Regulatory Recommendation

{recommendations}

---

*This report was automatically generated by VigiLenseAI. The literature section above is rendered directly from live PubMed API results — all PMIDs are verified and link to real publications. The agent analysis section reflects AI reasoning over retrieved data. All findings must be reviewed by a qualified pharmacovigilance professional before regulatory submission (ICH E2D / CIOMS VI).*
"""

    return {
        "status":          "generated",
        "report_markdown": report_markdown,
        "generated_at":    generated_at
    }


def submit_final_report(
    confidence_score: int,
    evidence_chain: list,
    signal_level: str,
    novel_signal_detected: bool,
    severe_events_found: list,
    reasoning: str,
    recommended_action: str
) -> dict:
    """
    Terminates the ReAct loop. Called by the agent as its final action.
    Returns a confirmation dict that the agent loop detects to stop iteration.
    """
    return {
        "status":               "submitted",
        "confidence_score":     confidence_score,
        "signal_level":         signal_level,
        "novel_signal_detected": novel_signal_detected,
        "recommended_action":   recommended_action
    }


# ── Tool Dispatcher ───────────────────────────────────────────────────────────

def _filter(fn_args: dict, allowed: set) -> dict:
    """Strip any unexpected keys the model may inject into function arguments."""
    return {k: v for k, v in fn_args.items() if k in allowed}


def dispatch(fn_name: str, fn_args: dict) -> dict:
    """Route a tool call from the agent to its implementation."""
    if fn_name == "get_drug_profile":
        return get_drug_profile(**_filter(fn_args, {"drug_name"}))
    if fn_name == "calculate_disproportionality":
        return calculate_disproportionality(**_filter(fn_args, {
            "cases_drug_event", "cases_drug_other", "cases_other_event", "cases_other_other"
        }))
    if fn_name == "fetch_fda_adverse_events":
        return fetch_fda_adverse_events(**_filter(fn_args, {"drug_name", "adverse_event"}))
    if fn_name == "fetch_pubmed_advanced":
        return fetch_pubmed_advanced(**_filter(fn_args, {"query_term", "max_results", "min_year", "investigation_context"}))
    if fn_name == "search_drug_class_effects":
        return search_drug_class_effects(**_filter(fn_args, {"drug_class", "adverse_event", "max_results", "min_year", "investigation_context"}))
    if fn_name == "check_past_signals":
        return check_past_signals(**_filter(fn_args, {"drug_name", "adverse_event"}))
    if fn_name == "query_knowledge_base":
        return query_knowledge_base(**_filter(fn_args, {"drug_name", "query", "section", "top_k"}))
    if fn_name == "abort_investigation":
        return abort_investigation(**_filter(fn_args, {"abort_code", "reason"}))
    if fn_name == "generate_pharmacovigilance_report":
        return generate_pharmacovigilance_report(**_filter(fn_args, {
            "drug_name", "adverse_event", "is_significant", "signal_level",
            "summary_findings", "recommendations", "ror", "ci_95",
            "disproportionality_source", "literature_section", "case_counts"
        }))
    if fn_name == "submit_final_report":
        return submit_final_report(**_filter(fn_args, {
            "confidence_score", "evidence_chain", "signal_level",
            "novel_signal_detected", "severe_events_found", "reasoning", "recommended_action"
        }))
    return {"error": f"Unknown tool: {fn_name}"}
