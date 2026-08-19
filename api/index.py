"""
index.py — FastAPI application entry point.

Responsibilities:
  - Define data models (request / response)
  - Register all HTTP routes
  - Delegate business logic to agent.py and tools.py

Routes:
  GET  /                        serve the frontend GUI
  GET  /api/team_info           student details
  GET  /api/agent_info          agent description + example
  GET  /api/model_architecture  architecture diagram image
  POST /api/execute             run the ReAct agent
"""

import os
import json
import sys

# Ensure sibling modules (config, tools, agent) are importable
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from agent import run_react_loop, log_to_supabase
from config import supabase

app = FastAPI()


# ── Data Models ───────────────────────────────────────────────────────────────

class StepRecord(BaseModel):
    module:   str
    prompt:   Dict[str, Any]
    response: Dict[str, Any]

class ExecuteRequest(BaseModel):
    prompt: str

class ExecuteResponse(BaseModel):
    status:   str
    error:    Optional[str]    = None
    response: Optional[str]    = None
    steps:    List[StepRecord] = []


# ── Static Routes ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def serve_gui():
    """Serve the frontend HTML (located one level up at project root)."""
    html_path = os.path.join(os.path.dirname(__file__), "..", "index.html")
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>index.html not found.</h1>"


@app.get("/api/model_architecture")
def get_model_architecture():
    """Return the architecture diagram image (add architecture.png to api/)."""
    image_path = os.path.join(os.path.dirname(__file__), "architecture.png")
    if os.path.exists(image_path):
        return FileResponse(image_path, media_type="image/png")
    return {"error": "architecture.png not found. Add it to the api/ folder."}


# ── Info Routes ───────────────────────────────────────────────────────────────

@app.get("/api/team_info")
def get_team_info():
    return {
        "group_batch_order_number": "1_6",
        "team_name": "VigiLenseAI",
        "students": [
            {"name": "Ofek Shpinner", "email": "ofek.shpinner@gmail.com"},
            {"name": "Aviad Alon",    "email": "aviadalon96@gmail.com"},
        ]
    }


@app.get("/api/agent_info")
def get_agent_info():
    return {
        "description": (
            "VigiLenseAI is a pharmacovigilance AI agent built on the ReAct "
            "(Reason \u2192 Act \u2192 Observe) architecture. "
            "Given a query about a drug in the company's portfolio, the agent conducts a "
            "full investigation aimed at discovering new and previously unknown adverse events "
            "for the drug under review. It has access to 9 specialized tools and independently "
            "decides which tool to call at each step in service of the final goal. "
            "Once it determines the investigation is complete, or upon reaching its iteration "
            "limit, it produces a structured report summarizing all findings."
        ),
        "purpose": (
            "VigiLenseAI is built to support pharmaceutical safety teams in their post-market "
            "surveillance work. It automates the most time-consuming parts of the process - "
            "screening literature, querying adverse event databases, and generating reports - "
            "so safety teams can focus on decision-making rather than data gathering. Faster signal "
            "detection also reduces the risk of missing reporting deadlines and the regulatory "
            "consequences that follow."
        ),
        "prompt_template": {
            "template": (
                "Analyze recent medical literature for {drug_name}. "
                "Identify any severe adverse events NOT present in the known safety profile."
            )
        },
        "prompt_examples": [
            {
                "prompt": (
                    "Review the latest PubMed findings on Metformin side effects, "
                    "especially in elderly patients, and flag any emerging risks."
                ),
                "full_response": '# VigiLenseAI — Pharmacovigilance Evaluation Report\n\n**Report Date:** 2026-08-15T20:42:57.542762+00:00\n**Status:** 🔴 SIGNIFICANT SAFETY SIGNAL\n\n---\n\n## Subject\n\n| Field | Value |\n|-------|-------|\n| **Drug Under Investigation** | metformin |\n| **Adverse Event** | Vitamin B12 deficiency and related neurological sequelae in elderly patients |\n| **Signal Classification** | 🔴 SIGNIFICANT SAFETY SIGNAL |\n\n---\n\n## Statistical Disproportionality Analysis\n\n| Metric | Value |\n|--------|-------|\n| **Calculated ROR** | 11.06 |\n| **95% CI** | [10.26, 11.92] |\n| **Significant?** | Yes — ROR ≥ 2.0 and lower CI > 1.0 |\n| **Data Source** | OpenFDA FAERS Database |\n\n> Significance criterion: ROR ≥ 2.0 AND lower 95% CI > 1.0\n\n> **Note:** Literature lacked quantitative 2×2 contingency figures. Real-world pharmacovigilance data was retrieved directly from OpenFDA FAERS to enable disproportionality analysis.\n\n---\n\n## Literature Retrieved from PubMed\n\n**PubMed Retrieval Audit**\n\n- `"metformin" AND ("adverse effects" OR "side effects" OR "lactic acidosis" OR "hypoglycemia" OR "vitamin B12 deficiency" OR "falls" OR "cognitive impairment" OR "renal impairment" OR "gastrointestinal")` — 2020 – present — 4162 found · 196 fetched · screened 196 (100%) · **196 included**\n\n**Articles with Drug-Specific Findings (n=8)**\n\n**1.** **Metformin-Induced Vitamin B12 Deficiency in Patients With Type-2 Diabetes Mellitus.**\nTiwari Aakriti et al. · *Cureus* · 2023 Oct · [PMID: 38034222](https://pubmed.ncbi.nlm.nih.gov/38034222/)\n\n**Key Drug-Specific Findings:**\nReview article (2023, Cureus) focused on metformin-induced vitamin B12 deficiency in T2DM patients. Summarizes observational data linking metformin to reduced B12 absorption, clinical manifestations (macrocytic anemia, neuropathy, demyelination), and risk modifiers (older age, PPI use, bariatric surgery). Recommends early detection using methylmalonic acid/homocysteine when B12 borderline and consideration of supplementation.\n\n---\n\n**2.** **Association between metformin and vitamin B12 deficiency in patients with type 2 diabetes.**\nSayedali Ehsan et al. · *World journal of diabetes* · 2023 May · [PMID: 37273250](https://pubmed.ncbi.nlm.nih.gov/37273250/)\n\n**Key Drug-Specific Findings:**\nReview article (2023) addressing the association between metformin and vitamin B12 deficiency across clinical studies. Study types included: observational cohorts and cross-sectional studies. Findings: consistent signal of B12 deficiency among metformin-treated T2DM patients; suggested mechanisms and clinical consequences (neuropathy, anemia). The authors highlight factors accelerating B12 depletion (PPI use, bariatric surgery, older age) and recommend measurement of methylmalonic acid/homocysteine for borderline B12 levels.\n\n---\n\n**3.** **Metformin-induced vitamin B12 deficiency can cause or worsen distal symmetrical, autonomic and cardiac neuropathy in the patient with diabetes.**\nBell David S H · *Diabetes, obesity & metabolism* · 2022 Aug · [PMID: 35491956](https://pubmed.ncbi.nlm.nih.gov/35491956/)\n\n**Key Drug-Specific Findings:**\nNarrative review and synthesis (2022) summarizing observational studies and meta-analyses reporting an association between metformin use and reduced vitamin B12 concentrations. Patient population: adults with type 2 diabetes on metformin in multiple cohorts. Safety findings: metformin users exhibited lower serum B12 and higher frequencies of distal symmetrical polyneuropathy and autonomic neuropathy (including cardiac denervation). The review emphasizes dose- and duration-dependence (higher prevalence with long-term and high-dose metformin) and recommends periodic B12 monitoring, especially after ~5 years of therapy; mechanistic discussion includes interference with calcium-dependent intrinsic factor–B12 complex uptake in the terminal ileum.\n\n---\n\n**4.** **Metformin, Cognitive Function, and Changes in the Gut Microbiome.**\nRosell-Díaz Marisel et al. · *Endocrine reviews* · 2024 Mar · [PMID: 37603460](https://pubmed.ncbi.nlm.nih.gov/37603460/)\n\n**Key Drug-Specific Findings:**\nReview (2024, Endocrine Reviews) examining metformin’s effects on cognitive function and gut microbiome in humans. Study designs summarized include observational cohorts and interventional studies in adults with T2DM. Findings: mixed results — some studies report cognitive benefit or lower dementia risk, others are neutral; metformin alters gut microbiota taxonomy and metabolomics. Relevance to signal: raises mechanistic plausibility that metformin-modulated microbiome and B12 malabsorption could influence cognition in older patients, and underscores heterogeneity across studies.\n\n---\n\n**5.** **A Case of Metformin-Associated Lactic Acidosis.**\nMahmood Rabia et al. · *Cureus* · 2023 Apr · [PMID: 37252492](https://pubmed.ncbi.nlm.nih.gov/37252492/)\n\n**Key Drug-Specific Findings:**\nCase report (2023, Cureus) of an elderly woman with multiple comorbidities on metformin presenting with acute renal failure, severe metabolic acidosis and markedly elevated lactate consistent with MALA; metformin was discontinued. Clinical course: required intubation and continuous renal replacement therapy and recovered. Relevance: underscores MALA risk in decompensated elderly patients and the need to recognize overlapping risks (AKI precipitating both MALA and B12-related neuropathy can coexist).\n\n---\n\n**6.** **Metformin-Associated Vitamin B12 Deficiency: An Underrecognized Complication.**\nThewjitcharoen Yotsapon et al. · *JCEM case reports* · 2024 Apr · [PMID: 38623534](https://pubmed.ncbi.nlm.nih.gov/38623534/)\n\n**Key Drug-Specific Findings:**\nCase report (2024, JCEM case reports) of an 84-year-old man on long-term metformin (1700 mg/day for >25 years) presenting with gait instability, frequent falls and an acute-on-chronic subdural hematoma. Laboratory work-up showed undetectable plasma vitamin B12 and elevated homocysteine. Clinical course: neurological improvement occurred within 1 week of vitamin B12 replacement and surgical hematoma evacuation. Relevance: illustrates severe clinical sequelae (falls, intracranial hemorrhage) that can follow metformin-associated B12 deficiency in elderly patients.\n\n---\n\n**7.** **Metformin-Induced Vitamin B12 Deficiency among Type 2 Diabetes Mellitus\' Patients: A Systematic Review.**\nKhattab Rabie et al. · *Current diabetes reviews* · 2023 · [PMID: 35440313](https://pubmed.ncbi.nlm.nih.gov/35440313/)\n\n**Key Drug-Specific Findings:**\nSystematic review (2023, Current Diabetes Reviews) pooling 19 studies (observational and RCTs) on metformin-induced B12 deficiency. Findings: positive correlation between metformin exposure and B12 deficiency, increased homocysteine, and borderline evidence of clinical neuropathy. Recommends routine monitoring for long-term metformin therapy and prophylactic supplementation strategies.\n\n---\n\n**8.** **Vitamin B12 deficiency in diabetic patients treated with metformin: A cross-sectional study.**\nHuynh Dat Tan et al. · *PloS one* · 2024 · [PMID: 38662670](https://pubmed.ncbi.nlm.nih.gov/38662670/)\n\n**Key Drug-Specific Findings:**\nCross-sectional study (2024, PLoS One) in Vietnamese T2DM patients treated with metformin assessing prevalence and risk factors for B12 deficiency. Findings: B12 deficiency prevalence 18.6%; higher metformin dose and longer duration associated with deficiency; dose > median and long-term use combined increased odds. Relevance: provides population-specific quantitative support for dose-duration association in adults, including older patients.\n\n---\n\n## Agent Analysis\n\n### Internal KB / FDA Label Baseline\n- Lactic acidosis — Boxed Warning (Section 5.1) (Warnings and boxed_warning chunks returned by internal KB)\n- Contraindicated in severe renal impairment (eGFR <30 mL/min/1.73 m2) — Contraindications (Section 4)\n- Warnings and Precautions list acute renal failure, pancreatitis, and other comorbidities increasing risk of lactic acidosis (Warnings And Precautions chunk returned by internal KB)\n\n### Novel Findings\n- Association between long-term metformin use and vitamin B12 deficiency with clinical consequences (peripheral neuropathy, autonomic and cardiac neuropathy) [3]\n- Case reports and series describing severe B12-deficiency–related outcomes in older patients on metformin, including gait instability, frequent falls and an acute-on-chronic subdural hematoma requiring neurosurgical intervention [6]\n- Systematic reviews and observational studies consistently reporting higher prevalence of biochemical vitamin B12 deficiency among metformin-treated patients, with dose- and duration-dependence highlighted [2]\n- Emerging evidence linking metformin exposure to changes in cognition and gut microbiome with mixed signals on benefit vs. risk for cognitive outcomes in older adults (raises concern about B12-mediated cognitive impairment where deficiency is present) [4]\n\n### Known / Expected Findings\n- Gastrointestinal adverse effects (nausea, diarrhea) — well described in label and multiple reviews (Warnings/Adverse Reactions sections)\n- Metformin-associated lactic acidosis (MALA) — recognized, boxed warning (Section 5.1)\n\n### Signal Assessment\n- Evidence quality: Multiple recent systematic reviews, meta-analyses, observational cohorts, and case reports in the 2020–2025 literature consistently identify an association between metformin exposure and lower vitamin B12 levels and clinical sequelae in some patients (see included PMIDs).\n- Signal strength: Strong disproportionality in FAERS (ROR = 11.06; 95% CI 10.26–11.92) for the PT "Vitamin B12 deficiency" with metformin (OpenFDA FAERS Database counts a=852, b=432,050, c=3,612, d=20,256,176).\n- Confidence: High that an association exists between metformin use and reports of vitamin B12 deficiency; moderate confidence that clinically consequential outcomes (neuropathy, falls, cognitive impairment) occur in a subset of elderly patients, given case-series and observational data.\n\n\n\n---\n\n## Regulatory Recommendation\n\nEscalate to Safety Team — Rationale: Recent literature (systematic reviews, observational cohorts, and multiple case reports) plus a statistically significant FAERS disproportionality signal (ROR 11.06, 95% CI 10.26–11.92) indicate that vitamin B12 deficiency and related neurological sequelae in elderly metformin users warrant regulatory review. Suggested actions: 1) Consider targeted safety communication to clinicians about screening B12 in older patients, those on high-dose or long-duration metformin, and patients with neuropathy or cognitive decline; 2) Review internal labeling and monitoring guidance for inclusion of B12 deficiency screening recommendations; 3) Convene a signal review with the clinical safety, epidemiology, and regulatory teams to evaluate need for label update or targeted pharmacoepidemiology studies.\n\n---\n\n*This report was automatically generated by VigiLenseAI. The literature section above is rendered directly from live PubMed API results — all PMIDs are verified and link to real publications. The agent analysis section reflects AI reasoning over retrieved data. All findings must be reviewed by a qualified pharmacovigilance professional before regulatory submission (ICH E2D / CIOMS VI).*\n',
                "steps": [   {   'module': 'query_knowledge_base',
        'prompt': {'drug_name': 'metformin', 'query': 'side effects in elderly', 'top_k': 3},
        'response': {'drug': 'metformin', 'drug_in_formulary': True, 'found': 3}},
    {   'module': 'get_drug_profile',
        'prompt': {'drug_name': 'metformin'},
        'response': {   'active_ingredients': ['metformin'],
                        'brand_names': ['metformin'],
                        'drug_class': 'Biguanides',
                        'drug_name': 'metformin',
                        'mechanism': 'Mechanism not available via RxNorm.',
                        'rxcui': '6809',
                        'source': 'rxnorm'}},
    {   'module': 'fetch_pubmed_advanced',
        'prompt': {   'max_results': 20,
                      'min_year': 2020,
                      'query_term': '"metformin" AND ("adverse effects" OR "side effects" OR "lactic acidosis" OR '
                                    '"hypoglycemia" OR "vitamin B12 deficiency" OR "falls" OR "cognitive impairment" '
                                    'OR "renal impairment" OR "gastrointestinal")'},
        'response': {   'count': 196,
                        'query_term': '"metformin" AND ("adverse effects" OR "side effects" OR "lactic acidosis" OR '
                                      '"hypoglycemia" OR "vitamin B12 deficiency" OR "falls" OR "cognitive impairment" '
                                      'OR "renal impairment" OR "gastrointestinal")'}},
    {   'module': 'query_knowledge_base',
        'prompt': {'drug_name': 'metformin', 'query': 'vitamin B12 deficiency', 'top_k': 3},
        'response': {'drug': 'metformin', 'drug_in_formulary': True, 'found': 3}},
    {   'module': 'fetch_fda_adverse_events',
        'prompt': {'adverse_event': 'Vitamin B12 deficiency', 'drug_name': 'metformin'},
        'response': {   'a': 852,
                        'adverse_event': 'Vitamin B12 deficiency',
                        'b': 432050,
                        'c': 3612,
                        'd': 20256176,
                        'drug_name': 'metformin',
                        'source': 'OpenFDA FAERS API'}},
    {   'module': 'calculate_disproportionality',
        'prompt': {   'cases_drug_event': 852,
                      'cases_drug_other': 432050,
                      'cases_other_event': 3612,
                      'cases_other_other': 20256176},
        'response': {   'ci_95': [10.26, 11.92],
                        'interpretation': 'Statistically significant safety signal detected (ROR ≥ 2.0 and lower CI > '
                                          '1.0).',
                        'ror': 11.06,
                        'statistically_significant': True}},
    {   'module': 'generate_pharmacovigilance_report',
        'prompt': {   'adverse_event': 'Vitamin B12 deficiency and related neurological sequelae in elderly patients',
                      'ci_95': [10.26, 11.92],
                      'disproportionality_source': 'OpenFDA FAERS Database',
                      'drug_name': 'metformin',
                      'is_significant': True,
                      'recommendations': 'Escalate to Safety Team — Rationale: Recent literature (systematic reviews, '
                                         'observational cohorts, and multiple case reports) plus a statistically '
                                         'significant FAERS disproportionality signal (ROR 11.06, 95% CI 10.26–11.92) '
                                         'indicate that vitamin B12 deficiency and related neurological sequelae in '
                                         'elderly metformin users warrant regulatory review. Suggested actions: 1) '
                                         'Consider targeted safety communication to clinicians about screening B12 in '
                                         'older patients, those on high-dose or long-duration metformin, and patients '
                                         'with neuropathy or cognitive decline; 2) Review internal labeling and '
                                         'monitoring guidance for inclusion of B12 deficiency screening '
                                         'recommendations; 3) Convene a signal review with the clinical safety, '
                                         'epidemiology, and regulatory teams to evaluate need for label update or '
                                         'targeted pharmacoepidemiology studies.',
                      'ror': 11.06,
                      'summary_findings': '### Internal KB / FDA Label Baseline\n'
                                          '- Lactic acidosis — Boxed Warning (Section 5.1) (Warnings and boxed_warning '
                                          'chunks returned by internal KB)\n'
                                          '- Contraindicated in severe renal impairment (eGFR <30 mL/min/1.73 m2) — '
                                          'Contraindications (Section 4)\n'
                                          '- Warnings and Precautions list acute renal failure, pancreatitis, and '
                                          'other comorbidities increasing risk of lactic acidosis (Warnings And '
                                          'Precautions chunk returned by internal KB)\n'
                                          '\n'
                                          '### Novel Findings\n'
                                          '- Association between long-term metformin use and vitamin B12 deficiency '
                                          'with clinical consequences (peripheral neuropathy, autonomic and cardiac '
                                          'neuropathy) [3]\n'
                                          '- Case reports and series describing severe B12-deficiency–related outcomes '
                                          'in older patients on metformin, including gait instability, frequent falls '
                                          'and an acute-on-chronic subdural hematoma requiring neurosurgical '
                                          'intervention [6]\n'
                                          '- Systematic reviews and observational studies consistently reporting '
                                          'higher prevalence of biochemical vitamin B12 deficiency among '
                                          'metformin-treated patients, with dose- and duration-dependence highlighted '
                                          '[2]\n'
                                          '- Emerging evidence linking metformin exposure to changes in cognition and '
                                          'gut microbiome with mixed signals on benefit vs. risk for cognitive '
                                          'outcomes in older adults (raises concern about B12-mediated cognitive '
                                          'impairment where deficiency is present) [4]\n'
                                          '\n'
                                          '### Known / Expected Findings\n'
                                          '- Gastrointestinal adverse effects (nausea, diarrhea) — well described in '
                                          'label and multiple reviews (Warnings/Adverse Reactions sections)\n'
                                          '- Metformin-associated lactic acidosis (MALA) — recognized, boxed warning '
                                          '(Section 5.1)\n'
                                          '\n'
                                          '### Signal Assessment\n'
                                          '- Evidence quality: Multiple recent systematic reviews, meta-analyses, '
                                          'observational cohorts, and case reports in the 2020–2025 literature '
                                          'consistently identify an association between metformin exposure and lower '
                                          'vitamin B12 levels and clinical sequelae in some patients (see included '
                                          'PMIDs).\n'
                                          '- Signal strength: Strong disproportionality in FAERS (ROR = 11.06; 95% CI '
                                          '10.26–11.92) for the PT "Vitamin B12 deficiency" with metformin (OpenFDA '
                                          'FAERS Database counts a=852, b=432,050, c=3,612, d=20,256,176).\n'
                                          '- Confidence: High that an association exists between metformin use and '
                                          'reports of vitamin B12 deficiency; moderate confidence that clinically '
                                          'consequential outcomes (neuropathy, falls, cognitive impairment) occur in a '
                                          'subset of elderly patients, given case-series and observational data.\n'
                                          '\n'},
        'response': {'generated_at': '2026-08-15T20:42:57.542762+00:00', 'status': 'generated'}},
    {   'module': 'submit_final_report',
        'prompt': {   'confidence_score': 78,
                      'evidence_chain': [   'query_knowledge_base',
                                            'get_drug_profile',
                                            'fetch_pubmed_advanced',
                                            'query_knowledge_base',
                                            'fetch_fda_adverse_events',
                                            'calculate_disproportionality',
                                            'generate_pharmacovigilance_report'],
                      'novel_signal_detected': True,
                      'reasoning': 'Recent systematic reviews, observational studies, and case reports identify '
                                   'metformin-associated vitamin B12 deficiency; internal KB documents lactic acidosis '
                                   'but does not enumerate vitamin B12 deficiency as a labelled adverse reaction in '
                                   'the returned chunks. FAERS disproportionality analysis (OpenFDA counts: a=852, '
                                   'b=432,050, c=3,612, d=20,256,176) yielded ROR 11.06 (95% CI 10.26-11.92), a '
                                   'statistically significant signal. Multiple PMIDs (e.g., 35491956; 37273250; '
                                   '38623534; 35440313; 38662670) provide supporting clinical and mechanistic data — '
                                   'including reports in older patients describing falls and subdural hematoma that '
                                   'improved with B12 replacement. Given the combination of consistent literature and '
                                   'strong disproportionality, escalation is recommended for targeted review of '
                                   'labeling and clinician guidance on B12 monitoring in elderly or '
                                   'long-term/high-dose metformin users.',
                      'recommended_action': 'Escalate to Safety Team',
                      'severe_events_found': ['Vitamin B12 deficiency leading to neuropathy/falls/subdural hematoma'],
                      'signal_level': 'drug-specific'},
        'response': {   'confidence_score': 78,
                        'evidence_chain': [   'query_knowledge_base',
                                              'get_drug_profile',
                                              'fetch_pubmed_advanced',
                                              'query_knowledge_base',
                                              'fetch_fda_adverse_events',
                                              'calculate_disproportionality',
                                              'generate_pharmacovigilance_report'],
                        'novel_signal_detected': True,
                        'reasoning': 'Recent systematic reviews, observational studies, and case reports identify '
                                     'metformin-associated vitamin B12 deficiency; internal KB documents lactic '
                                     'acidosis but does not enumerate vitamin B12 deficiency as a labelled adverse '
                                     'reaction in the returned chunks. FAERS disproportionality analysis (OpenFDA '
                                     'counts: a=852, b=432,050, c=3,612, d=20,256,176) yielded ROR 11.06 (95% CI '
                                     '10.26-11.92), a statistically significant signal. Multiple PMIDs (e.g., '
                                     '35491956; 37273250; 38623534; 35440313; 38662670) provide supporting clinical '
                                     'and mechanistic data — including reports in older patients describing falls and '
                                     'subdural hematoma that improved with B12 replacement. Given the combination of '
                                     'consistent literature and strong disproportionality, escalation is recommended '
                                     'for targeted review of labeling and clinician guidance on B12 monitoring in '
                                     'elderly or long-term/high-dose metformin users.',
                        'recommended_action': 'Escalate to Safety Team',
                        'severe_events_found': ['Vitamin B12 deficiency leading to neuropathy/falls/subdural hematoma'],
                        'signal_level': 'drug-specific'}}],
            },
            {
                "prompt": (
                    "Analyze recent medical literature for Acamol (paracetamol/acetaminophen). "
                    "Identify any severe adverse events not present in the known safety profile."
                ),
                "full_response": "'Acamol' is not in the organization's pharmacovigilance portfolio according to the internal knowledge base. This system only performs signal detection for drugs held in the company's portfolio.",
                "steps": [   {   'module': 'query_knowledge_base',
        'prompt': {'drug_name': 'Acamol', 'query': 'severe adverse events', 'top_k': 3},
        'response': {   'chunks': [],
                        'drug': 'Acamol',
                        'drug_in_formulary': False,
                        'found': 0,
                        'message': "'Acamol' has no prior documentation in the internal knowledge base. Any retrieved "
                                   'literature findings may represent novel signals.',
                        'query': 'severe adverse events'}},
    {   'module': 'abort_investigation',
        'prompt': {   'abort_code': 'drug_not_in_portfolio',
                      'reason': "'Acamol' is not in the organization's pharmacovigilance portfolio according to the "
                                'internal knowledge base. This system only performs signal detection for drugs held in '
                                "the company's portfolio."},
        'response': {   'abort_code': 'drug_not_in_portfolio',
                        'reason': "'Acamol' is not in the organization's pharmacovigilance portfolio according to the "
                                  'internal knowledge base. This system only performs signal detection for drugs held '
                                  "in the company's portfolio."}}],
            },
            {
                "prompt": (
                    "Review recent PubMed findings on Methotrexate toxicity, "
                    "including hepatotoxicity and pulmonary risks, for novel signals."
                ),
                "full_response": '# VigiLenseAI — Pharmacovigilance Evaluation Report\n\n**Report Date:** 2026-08-15T21:00:46.409538+00:00\n**Status:** 🔴 SIGNIFICANT SAFETY SIGNAL\n\n---\n\n## Subject\n\n| Field | Value |\n|-------|-------|\n| **Drug Under Investigation** | Methotrexate |\n| **Adverse Event** | Hepatotoxicity and Pulmonary toxicity (pneumonitis/ILD) |\n| **Signal Classification** | 🔴 SIGNIFICANT SAFETY SIGNAL |\n\n---\n\n## Statistical Disproportionality Analysis\n\n| Metric | Value |\n|--------|-------|\n| **Calculated ROR** | 5.27 |\n| **95% CI** | [5.05, 5.49] |\n| **Significant?** | Yes — ROR ≥ 2.0 and lower CI > 1.0 |\n| **Data Source** | OpenFDA FAERS Database |\n\n> Significance criterion: ROR ≥ 2.0 AND lower 95% CI > 1.0\n\n> **Note:** Literature lacked quantitative 2×2 contingency figures. Real-world pharmacovigilance data was retrieved directly from OpenFDA FAERS to enable disproportionality analysis.\n\n---\n\n## Literature Retrieved from PubMed\n\n**PubMed Retrieval Audit**\n\n- `"methotrexate" AND ("hepatotoxicity" OR "drug-induced liver injury" OR "liver injury" OR "hepatitis" OR "transaminase elevation")` — 2020 – present — 511 found · 196 fetched · screened 196 (100%) · **196 included**\n- `"methotrexate" AND ("pulmonary toxicity" OR "pneumonitis" OR "interstitial lung disease" OR "pulmonary fibrosis" OR "pulmonary" )` — 2020 – present — 715 found · 200 fetched · screened 200 (100%) · **200 included**\n\n**Articles with Drug-Specific Findings (n=9)**\n\n**1.** **Glucarpidase for treatment of high-dose methotrexate toxicity.**\nGupta Shruti et al. · *Blood* · 2025 Apr · [PMID: 39760780](https://pubmed.ncbi.nlm.nih.gov/39760780/)\n\n**Key Drug-Specific Findings:**\nMulticenter observational cohort emulating target trial assessing glucarpidase use after high-dose MTX-associated acute kidney injury. N=708 adults with MTX-AKI across 28 centers; 209 (29.5%) received glucarpidase. Primary endpoint: kidney recovery at discharge. Glucarpidase receipt associated with improved renal recovery (adjusted OR 2.70; 95% CI 1.69–4.31) and reduced grade ≥2 transaminitis at day 7 (aOR 0.50; 95% CI 0.28–0.91). Study design: retrospective multi-center comparative effectiveness with adjustment. Safety/adverse events: no increase in mortality; suggests rapid MTX cleavage may reduce hepatic injury in HD-MTX toxicity settings.\n\n---\n\n**2.** **Busting the myth of methotrexate chronic hepatotoxicity.**\nDi Martino Vincent et al. · *Nature reviews. Rheumatology* · 2023 Feb · [PMID: 36564450](https://pubmed.ncbi.nlm.nih.gov/36564450/)\n\n**Key Drug-Specific Findings:**\nNarrative review (Nature Reviews Rheumatology) reappraising chronic MTX hepatotoxicity: synthesizes recent evidence suggesting that apparent MTX-related liver fibrosis often reflects metabolic liver disease (NAFLD/MASLD) and comorbidities rather than direct MTX accumulation. Discusses updated understanding of mechanisms of acute vs chronic hepatotoxicity, role of non-invasive fibrosis tests (elastography), and proposes risk-stratified monitoring. Population: RA patients on long-term low-dose MTX; outcomes: re-evaluation of historical concept of cumulative-dose–driven fibrosis; implications: use of non-invasive tools and focus on metabolic risk factors in monitoring.\n\n---\n\n**3.** **Association between SLCO1B1 polymorphism and methotrexate-induced hepatotoxicity: a systematic review and meta-analysis.**\nHan Ji Min et al. · *Anti-cancer drugs* · 2022 Jan · [PMID: 34726639](https://pubmed.ncbi.nlm.nih.gov/34726639/)\n\n**Key Drug-Specific Findings:**\nSystematic review and meta-analysis (patients receiving high-dose MTX in oncology) assessing association of SLCO1B1 T521C (rs4149056) polymorphism with MTX-induced hepatotoxicity. Pooled analysis of five studies (n=465) receiving high-dose MTX (1–5 g/m2) showed carriers of the C allele had ~1.9-fold higher odds of hepatotoxicity vs TT homozygotes (OR 1.94; 95% CI 1.14–3.31). Study designs: observational pharmacogenetic studies in cancer cohorts on HD-MTX. Safety finding: SLCO1B1 variant associated with increased hepatotoxicity risk — potential for pharmacogenetic risk stratification in oncology HD-MTX, uncertain generalizability to low-dose rheumatologic MTX.\n\n---\n\n**4.** **Is methotrexate hepatotoxicity associated with cumulative dose? A systematic review and meta-analysis.**\nAzzam Ali et al. · *The Australasian journal of dermatology* · 2021 May · [PMID: 33769558](https://pubmed.ncbi.nlm.nih.gov/33769558/)\n\n**Key Drug-Specific Findings:**\nSystematic review and meta-analysis evaluating association of cumulative MTX dose with hepatotoxicity. 35 studies included; pooled analyses did not show consistent association between cumulative MTX dose and liver fibrosis. Study populations varied (dermatology, rheumatology). Outcome measures heterogeneous (biopsy, elastography, FibroTest, liver enzymes). Conclusion: challenges the assertion that cumulative low-dose MTX inevitably causes progressive fibrosis; highlights confounding by metabolic factors.\n\n---\n\n**5.** **Ferritinophagy-mediated ferroptosis facilitates methotrexate-induced hepatotoxicity by high-mobility group box\xa01 (HMGB1).**\nWang Chengbo et al. · *Liver international : official journal of the International Association for the Study of the Liver* · 2024 Mar · [PMID: 38082504](https://pubmed.ncbi.nlm.nih.gov/38082504/)\n\n**Key Drug-Specific Findings:**\nPreclinical mouse and cellular study exploring MTX-induced hepatotoxicity mechanisms. Demonstrated MTX triggers autophagy-dependent ferritinophagy (NCOA4-mediated) leading to iron-dependent lipid peroxidation and ferroptosis in hepatocytes. HMGB1 expression was elevated with MTX; genetic depletion or pharmacologic inhibition of HMGB1 reduced autophagy, ferroptosis markers, and liver injury. Glycyrrhizic acid (HMGB1 inhibitor) suppressed autophagy/ferroptosis and mitigated hepatic damage. Study population: in vitro hepatocytes and MTX-treated mice. Safety/clinical outcomes: mechanistic rescue with HMGB1 inhibitors reduced biochemical and histopathologic liver injury; translational implication: identifies ferritinophagy/HMGB1 as potential therapeutic targets to prevent MTX hepatotoxicity.\n\n---\n\n**6.** **Contribution of aldehyde oxidase to methotrexate-induced hepatotoxicity: in vitro and pharmacoepidemiological approaches.**\nMoriyama Ayako et al. · *Expert opinion on drug metabolism & toxicology* · 2024 May · [PMID: 38706380](https://pubmed.ncbi.nlm.nih.gov/38706380/)\n\n**Key Drug-Specific Findings:**\nCombined in vitro experiments and FAERS pharmacoepidemiologic analysis assessing role of hepatic aldehyde oxidase (AOX) in MTX metabolism and hepatotoxicity. In HepG2 cells, AOX inhibition (by raloxifene or siRNA) increased intracellular MTX accumulation and cytotoxicity. FAERS analysis reported increased ROR for MTX-related hepatotoxicity when co-reported with AOX inhibitors (authors\' analysis). Study type: mechanistic cell studies plus spontaneous-report database mining. Clinical implication: co-medications that inhibit AOX may enhance hepatic MTX exposure and hepatotoxicity; suggests need for DDI awareness and prospective study.\n\n---\n\n**7.** **Overview of Methotrexate Toxicity: A Comprehensive Literature Review.**\nHamed Khalid M et al. · *Cureus* · 2022 Sep · [PMID: 36312688](https://pubmed.ncbi.nlm.nih.gov/36312688/)\n\n**Key Drug-Specific Findings:**\nComprehensive literature review of methotrexate toxicity, summarizing multi-organ adverse effects including hepatotoxicity and pulmonary toxicity. Integrates mechanistic studies highlighting oxidative stress, mitochondrial dysfunction, inflammation, and fibrosis and suggests monitoring strategies; notes knowledge gaps around genetic susceptibility and mechanisms. Provides a backdrop for novel mechanistic studies (e.g., ferroptosis) and supports continued research into mitigation strategies.\n\n---\n\n**8.** **Methotrexate and rheumatoid arthritis associated interstitial lung disease.**\nJuge Pierre-Antoine et al. · *The European respiratory journal* · 2021 Feb · [PMID: 32646919](https://pubmed.ncbi.nlm.nih.gov/32646919/)\n\n**Key Drug-Specific Findings:**\nCase-control multicentre genetic/epidemiologic study investigating association of MTX exposure with RA-associated interstitial lung disease (RA-ILD). Across discovery and replication cohorts (total N>1000), MTX exposure was inversely associated with RA-ILD (adjusted OR ~0.43), and MTX-ever users were less frequent among RA-ILD cases versus RA controls. Data suggest MTX does not increase RA-ILD risk and may be associated with later detection of ILD. Study design: observational case-control across international sites. Clinical implication: evidence counters historical belief that MTX triggers RA-ILD; prompts re-evaluation of MTX contraindication in RA patients at risk for ILD.\n\n---\n\n**9.** **Methotrexate-associated pneumonitis.**\nHanai S et al. · *QJM : monthly journal of the Association of Physicians* · 2022 May · [PMID: 35253857](https://pubmed.ncbi.nlm.nih.gov/35253857/)\n\n**Key Drug-Specific Findings:**\n(Duplicate PMID included above for emphasis) Review focused on methotrexate-associated pneumonitis (MAP) summarizing clinical features, risk factors, and management; supports label-consistent pulmonary toxicity risk but underscores idiosyncratic nature and diagnostic challenges.\n\n---\n\n## Agent Analysis\n\n### Internal KB / FDA Label Baseline\n- Hepatotoxicity [Section: Adverse Reactions / Warnings and Precautions (5.5)]\n- Pulmonary toxicity (including pneumonitis) [Section: Adverse Reactions / Warnings and Precautions (5.6)]\n- Drug interactions and agents that may increase methotrexate exposure [Section: Drug Interactions (7)]\n\n### Novel Findings\n- Autophagy-dependent ferritinophagy and ferroptosis mediated by HMGB1 implicated in MTX-induced hepatotoxicity; inhibition of HMGB1 or ferritinophagy attenuated liver injury [5]\n- Association between SLCO1B1 T521C polymorphism (C allele) and increased risk of MTX-induced hepatotoxicity in high-dose MTX oncology regimens (meta-analysis finding ~1.9-fold risk) [3]\n- In vitro and pharmacoepidemiologic evidence that aldehyde oxidase (AOX) inhibition/interaction can increase hepatic MTX accumulation and hepatotoxicity (FAERS signal reported by authors) [6]\n- Real-world cohort evidence that glucarpidase administration after high-dose MTX-associated AKI is associated with reduced transaminitis (grade ≥2) and improved renal recovery (multi-center observational study) [1]\n\n### Known / Expected Findings\n- Hepatotoxicity (transaminase elevations, fibrosis risk in context of metabolic liver disease) – already described in label (Section: Adverse Reactions / Warnings) [multiple label chunks returned by KB]\n- Methotrexate-associated pneumonitis and other pulmonary adverse events (case reports, reviews) – documented in label and multiple case series; should be considered a labelled risk (Section: Adverse Reactions)\n\n### Signal Assessment\n- Evidence quality: Literature includes mechanistic preclinical studies (e.g., ferroptosis/ferritinophagy models), genetic meta-analysis (SLCO1B1), in vitro + pharmacoepidemiology, and multicenter observational cohorts; quality ranges from preclinical mechanistic to moderate-quality observational clinical evidence.\n- Signal strength: OpenFDA FAERS disproportionality shows statistically significant disproportionality for hepatotoxicity (ROR=5.27, 95% CI 5.05–5.49) and for pneumonitis (ROR=3.08, 95% CI 2.95–3.21) — both meet the conservative threshold (ROR ≥2 and lower CI >1).\n- Confidence: Moderate-to-high for detection of disproportionality in FAERS; moderate for novel mechanistic and pharmacogenomic findings (requires further clinical validation).\n\n\n\n---\n\n## Regulatory Recommendation\n\n- Regulatory/PHV action: Escalate to Safety Team for targeted review of MTX hepatotoxicity mechanisms and potential risk-mitigation actions: (1) consider reviewing label/KB for mention of SLCO1B1 and AOX interactions and update if warranted; (2) consider a targeted pharmacogenomics literature review to assess clinical actionability of SLCO1B1 genotyping in high-dose MTX oncology regimens; (3) consider surveillance or educational materials regarding drug interactions that inhibit AOX and potential increased MTX hepatotoxicity.\n- Rationale: Statistically significant FAERS disproportionality for both hepatotoxicity and pneumonitis plus multiple recent publications describing plausible mechanistic pathways (ferroptosis/HMGB1), genetic predisposition (SLCO1B1), and drug–drug interaction concerns (AOX inhibitors) that are not explicitly documented in the baseline KB chunks retrieved.\n\n\n---\n\n*This report was automatically generated by VigiLenseAI. The literature section above is rendered directly from live PubMed API results — all PMIDs are verified and link to real publications. The agent analysis section reflects AI reasoning over retrieved data. All findings must be reviewed by a qualified pharmacovigilance professional before regulatory submission (ICH E2D / CIOMS VI).*\n',
                "steps": [   {   'module': 'query_knowledge_base',
        'prompt': {'drug_name': 'Methotrexate', 'query': 'hepatotoxicity', 'top_k': 5},
        'response': {'drug': 'Methotrexate', 'drug_in_formulary': True, 'found': 5}},
    {   'module': 'get_drug_profile',
        'prompt': {'drug_name': 'Methotrexate'},
        'response': {   'active_ingredients': ['methotrexate'],
                        'brand_names': ['Methotrexate'],
                        'drug_class': 'Folic Acid Metabolism Inhibitors',
                        'drug_name': 'Methotrexate',
                        'mechanism': 'Mechanism not available via RxNorm.',
                        'rxcui': '6851',
                        'source': 'rxnorm'}},
    {   'module': 'fetch_pubmed_advanced',
        'prompt': {   'max_results': 50,
                      'min_year': 2020,
                      'query_term': '"methotrexate" AND ("hepatotoxicity" OR "drug-induced liver injury" OR "liver '
                                    'injury" OR "hepatitis" OR "transaminase elevation")'},
        'response': {   'count': 196,
                        'query_term': '"methotrexate" AND ("hepatotoxicity" OR "drug-induced liver injury" OR "liver '
                                      'injury" OR "hepatitis" OR "transaminase elevation")'}},
    {   'module': 'fetch_pubmed_advanced',
        'prompt': {   'max_results': 50,
                      'min_year': 2020,
                      'query_term': '"methotrexate" AND ("pulmonary toxicity" OR "pneumonitis" OR "interstitial lung '
                                    'disease" OR "pulmonary fibrosis" OR "pulmonary" )'},
        'response': {   'count': 200,
                        'query_term': '"methotrexate" AND ("pulmonary toxicity" OR "pneumonitis" OR "interstitial lung '
                                      'disease" OR "pulmonary fibrosis" OR "pulmonary" )'}},
    {   'module': 'query_knowledge_base',
        'prompt': {'drug_name': 'Methotrexate', 'query': 'ferroptosis', 'top_k': 5},
        'response': {'drug': 'Methotrexate', 'drug_in_formulary': True, 'found': 5}},
    {   'module': 'query_knowledge_base',
        'prompt': {'drug_name': 'Methotrexate', 'query': 'SLCO1B1 polymorphism', 'top_k': 5},
        'response': {'drug': 'Methotrexate', 'drug_in_formulary': True, 'found': 5}},
    {   'module': 'query_knowledge_base',
        'prompt': {'drug_name': 'Methotrexate', 'query': 'aldehyde oxidase OR AOX', 'top_k': 5},
        'response': {'drug': 'Methotrexate', 'drug_in_formulary': True, 'found': 5}},
    {   'module': 'fetch_fda_adverse_events',
        'prompt': {'adverse_event': 'Hepatotoxicity', 'drug_name': 'Methotrexate'},
        'response': {   'a': 2512,
                        'adverse_event': 'Hepatotoxicity',
                        'b': 497043,
                        'c': 19362,
                        'd': 20173773,
                        'drug_name': 'Methotrexate',
                        'source': 'OpenFDA FAERS API'}},
    {   'module': 'fetch_fda_adverse_events',
        'prompt': {'adverse_event': 'Pneumonitis', 'drug_name': 'Methotrexate'},
        'response': {   'a': 2217,
                        'adverse_event': 'Pneumonitis',
                        'b': 497338,
                        'c': 29217,
                        'd': 20163918,
                        'drug_name': 'Methotrexate',
                        'source': 'OpenFDA FAERS API'}},
    {   'module': 'calculate_disproportionality',
        'prompt': {   'cases_drug_event': 2512,
                      'cases_drug_other': 497043,
                      'cases_other_event': 19362,
                      'cases_other_other': 20173773},
        'response': {   'ci_95': [5.05, 5.49],
                        'interpretation': 'Statistically significant safety signal detected (ROR ≥ 2.0 and lower CI > '
                                          '1.0).',
                        'ror': 5.27,
                        'statistically_significant': True}},
    {   'module': 'calculate_disproportionality',
        'prompt': {   'cases_drug_event': 2217,
                      'cases_drug_other': 497338,
                      'cases_other_event': 29217,
                      'cases_other_other': 20163918},
        'response': {   'ci_95': [2.95, 3.21],
                        'interpretation': 'Statistically significant safety signal detected (ROR ≥ 2.0 and lower CI > '
                                          '1.0).',
                        'ror': 3.08,
                        'statistically_significant': True}},
    {   'module': 'generate_pharmacovigilance_report',
        'prompt': {   'adverse_event': 'Hepatotoxicity and Pulmonary toxicity (pneumonitis/ILD)',
                      'ci_95': [5.05, 5.49],
                      'confidence_score': 80,
                      'disproportionality_source': 'OpenFDA FAERS Database',
                      'drug_name': 'Methotrexate',
                      'is_significant': True,
                      'ror': 5.27},
        'response': {'generated_at': '2026-08-15T21:00:46.409538+00:00', 'status': 'generated'}},
    {   'module': 'submit_final_report',
        'prompt': {   'confidence_score': 80,
                      'evidence_chain': [   'query_knowledge_base (Methotrexate, hepatotoxicity)',
                                            'get_drug_profile (Methotrexate)',
                                            'fetch_pubmed_advanced (methotrexate hepatotoxicity search)',
                                            'fetch_pubmed_advanced (methotrexate pulmonary toxicity search)',
                                            'query_knowledge_base (Methotrexate, ferroptosis)',
                                            'query_knowledge_base (Methotrexate, SLCO1B1 polymorphism)',
                                            'query_knowledge_base (Methotrexate, aldehyde oxidase)',
                                            'fetch_fda_adverse_events (Methotrexate, Hepatotoxicity)',
                                            'fetch_fda_adverse_events (Methotrexate, Pneumonitis)',
                                            'calculate_disproportionality (hepatotoxicity)',
                                            'calculate_disproportionality (pneumonitis)',
                                            'generate_pharmacovigilance_report'],
                      'novel_signal_detected': True,
                      'reasoning': 'OpenFDA disproportionality analysis returned significant RORs for methotrexate '
                                   'with hepatotoxicity (ROR=5.27, 95% CI 5.05\x0205.49) and pneumonitis (ROR=3.08, '
                                   '95% CI 2.95\x0203.21). PubMed literature retrieved (2020\x02026 window) identified '
                                   'mechanistic and pharmaco-genetic findings not documented in the internal KB label '
                                   'chunks: ferritinophagy/ferroptosis mediated by HMGB1 (PMID: 38082504), SLCO1B1 '
                                   'T521C association with MTX hepatotoxicity (PMID: 34726639), and AOX inhibition '
                                   'increasing MTX hepatic accumulation (PMID: 38706380). The internal KB already '
                                   'documents hepatotoxicity and pulmonary toxicity in label sections; therefore these '
                                   'adverse events are known. However the mechanistic (ferroptosis/HMGB1), '
                                   'pharmacogenomic (SLCO1B1), and AOX DDI evidence represent novel, potentially '
                                   'actionable signals requiring safety-team review. Recommendation: escalate for '
                                   'targeted evaluation and consideration of KB/label updates and '
                                   'pharmacogenomic/interaction guidance.',
                      'recommended_action': 'Escalate to Safety Team: (1) Conduct a focused safety review of the new '
                                            'mechanistic and pharmacogenomic evidence (PMIDs cited) and evaluate need '
                                            'for label/KB updates; (2) Consider targeted pharmacoepidemiology or '
                                            'prospective studies to validate SLCO1B1 and AOX interaction risks, '
                                            'particularly in high-dose MTX contexts; (3) Consider educational outreach '
                                            'to prescribers on drug-drug interactions that may inhibit AOX and on '
                                            'risk-stratified monitoring (liver tests, elastography) in patients with '
                                            'metabolic risk factors.',
                      'severe_events_found': ['hepatotoxicity', 'pneumonitis/ILD'],
                      'signal_level': 'drug-specific'},
        'response': {   'confidence_score': 80,
                        'evidence_chain': [   'query_knowledge_base (Methotrexate, hepatotoxicity)',
                                              'get_drug_profile (Methotrexate)',
                                              'fetch_pubmed_advanced (methotrexate hepatotoxicity search)',
                                              'fetch_pubmed_advanced (methotrexate pulmonary toxicity search)',
                                              'query_knowledge_base (Methotrexate, ferroptosis)',
                                              'query_knowledge_base (Methotrexate, SLCO1B1 polymorphism)',
                                              'query_knowledge_base (Methotrexate, aldehyde oxidase)',
                                              'fetch_fda_adverse_events (Methotrexate, Hepatotoxicity)',
                                              'fetch_fda_adverse_events (Methotrexate, Pneumonitis)',
                                              'calculate_disproportionality (hepatotoxicity)',
                                              'calculate_disproportionality (pneumonitis)',
                                              'generate_pharmacovigilance_report'],
                        'novel_signal_detected': True,
                        'reasoning': 'OpenFDA disproportionality analysis returned significant RORs for methotrexate '
                                     'with hepatotoxicity (ROR=5.27, 95% CI 5.05\x0205.49) and pneumonitis (ROR=3.08, '
                                     '95% CI 2.95\x0203.21). PubMed literature retrieved (2020\x02026 window) '
                                     'identified mechanistic and pharmaco-genetic findings not documented in the '
                                     'internal KB label chunks: ferritinophagy/ferroptosis mediated by HMGB1 (PMID: '
                                     '38082504), SLCO1B1 T521C association with MTX hepatotoxicity (PMID: 34726639), '
                                     'and AOX inhibition increasing MTX hepatic accumulation (PMID: 38706380). The '
                                     'internal KB already documents hepatotoxicity and pulmonary toxicity in label '
                                     'sections; therefore these adverse events are known. However the mechanistic '
                                     '(ferroptosis/HMGB1), pharmacogenomic (SLCO1B1), and AOX DDI evidence represent '
                                     'novel, potentially actionable signals requiring safety-team review. '
                                     'Recommendation: escalate for targeted evaluation and consideration of KB/label '
                                     'updates and pharmacogenomic/interaction guidance.',
                        'recommended_action': 'Escalate to Safety Team: (1) Conduct a focused safety review of the new '
                                              'mechanistic and pharmacogenomic evidence (PMIDs cited) and evaluate '
                                              'need for label/KB updates; (2) Consider targeted pharmacoepidemiology '
                                              'or prospective studies to validate SLCO1B1 and AOX interaction risks, '
                                              'particularly in high-dose MTX contexts; (3) Consider educational '
                                              'outreach to prescribers on drug-drug interactions that may inhibit AOX '
                                              'and on risk-stratified monitoring (liver tests, elastography) in '
                                              'patients with metabolic risk factors.',
                        'severe_events_found': ['hepatotoxicity', 'pneumonitis/ILD'],
                        'signal_level': 'drug-specific'}}],
            }
        ]
    }


# ── History ───────────────────────────────────────────────────────────────────

@app.get("/api/history")
def get_history(limit: int = 10):
    """Return the most recent agent runs from Supabase agent_logs."""
    if not supabase:
        return {"runs": []}
    try:
        result = (
            supabase.table("agent_logs")
            .select("id, created_at, user_prompt, response, steps")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        runs = []
        for row in result.data:
            prompt   = row.get("user_prompt", "") or ""
            response = row.get("response",    "") or ""
            steps    = row.get("steps")            or []

            abort_code     = None
            is_full_report = False
            for step in steps:
                if not isinstance(step, dict):
                    continue
                if step.get("module") == "abort_investigation":
                    abort_code = (step.get("prompt") or {}).get("abort_code")
                if step.get("module") == "submit_final_report":
                    is_full_report = True

            runs.append({
                "id":             row.get("id"),
                "created_at":     row.get("created_at"),
                "prompt_preview": prompt[:100] + ("…" if len(prompt) > 100 else ""),
                "response":       response,
                "abort_code":     abort_code,
                "is_full_report": is_full_report,
            })
        return {"runs": runs}
    except Exception as exc:
        return {"runs": [], "error": str(exc)}


# ── Agent Execution ───────────────────────────────────────────────────────────

@app.post("/api/execute", response_model=ExecuteResponse)
def execute_agent(request: ExecuteRequest):
    """Run the ReAct agent loop and return a structured pharmacovigilance report."""
    try:
        final_report, steps = run_react_loop(request.prompt)
        # Prefer: (1) formatted Markdown report, (2) user-friendly reason/reasoning text, (3) raw JSON last resort
        final_answer = (
            final_report.get("report_markdown")
            or final_report.get("reasoning")
            or json.dumps(final_report, indent=2, ensure_ascii=False)
        )
        log_to_supabase(request.prompt, final_answer, steps)
        return ExecuteResponse(status="ok", response=final_answer, steps=steps)

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return ExecuteResponse(status="error", error=str(exc))
