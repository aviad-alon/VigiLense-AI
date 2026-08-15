<div align="center">

# VigiLenseAI

An autonomous pharmacovigilance signal detection agent built on the ReAct (Reason → Act → Observe) architecture.

[![Python](https://img.shields.io/badge/Python-3.11+-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Pinecone](https://img.shields.io/badge/Pinecone-RAG-000000?style=flat-square)](https://pinecone.io/)
[![Supabase](https://img.shields.io/badge/Supabase-PostgreSQL-3ECF8E?style=flat-square&logo=supabase&logoColor=white)](https://supabase.com/)
[![Vercel](https://img.shields.io/badge/Deploy-Vercel-000000?style=flat-square&logo=vercel&logoColor=white)](https://vercel.com/)

[Overview](#overview) • [Features](#features) • [Getting Started](#getting-started) • [Tools](#tools) • [API Reference](#api-reference) • [Deployment](#deployment)

</div>

---

## Overview

VigiLenseAI investigates potential adverse drug events by autonomously querying medical literature, FDA databases, and an internal knowledge base. Given a natural-language query, the agent iterates through up to 15 Reason → Act → Observe cycles, then produces a structured pharmacovigilance report in CIOMS/ICH E2D format — complete with numbered PubMed citations and a statistical signal assessment (Reporting Odds Ratio + 95% CI).

The agent enforces a **zero-fabrication policy**: every claim in the final report must be traceable to a tool response. A built-in guardrail system halts investigations that fall outside the system's scope before any expensive reasoning begins.

## Features

- **Autonomous ReAct loop** — iterative reasoning with tool calls, terminating on `submit_final_report` or a guardrail trigger
- **Evidence-grounded reports** — CIOMS/ICH E2D format with numbered PubMed citations; no citation can appear that wasn't retrieved by a tool
- **RAG knowledge base** — Pinecone vector search over FDA label documents for 8 formulary drugs
- **Portfolio guardrails** — 6 abort codes reject out-of-scope queries immediately: `drug_not_in_portfolio`, `query_too_vague`, `multiple_drugs_detected`, `non_medical_query`, `drug_not_recognized`, `no_literature_found`
- **Disproportionality analysis** — Reporting Odds Ratio with 95% CI, computed from OpenFDA FAERS counts
- **Execution tracing** — every step of the agent's reasoning is returned to the UI for full transparency
- **Investigation history** — all runs persisted to Supabase and queryable via `/api/history`

## Architecture

![VigiLenseAI System Architecture](api/architecture.png)

The full interactive diagram is available at [`architecture.html`](architecture.html).

| Layer | Technology |
|-------|-----------|
| Frontend | Vanilla HTML/CSS/JS, served by FastAPI |
| Backend | FastAPI + Uvicorn |
| LLM | LLMod.ai — `NBUECSE-gpt-5-mini` (OpenAI-compatible) |
| Vector DB | Pinecone (RAG over FDA label documents) |
| Relational DB | Supabase (PostgreSQL — `agent_logs` table) |
| Deployment | Vercel (`@vercel/python`) |

## Getting Started

### Prerequisites

- Python 3.11+
- [Pinecone](https://pinecone.io/) account with an index named `vigilense`
- [Supabase](https://supabase.com/) project with an `agent_logs` table
- [LLMod.ai](https://llmod.ai/) API key (or any OpenAI-compatible endpoint)

### Local setup

1. Clone the repository:

   ```bash
   git clone https://github.com/aviad-alon/VigiLense-AI.git
   cd VigiLense-AI
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Create a `.env` file at the project root (see [Environment variables](#environment-variables) below).

4. (First run only) Seed the Pinecone knowledge base with the bundled drug profiles:

   ```bash
   python seed_db.py
   ```

5. Start the development server:

   ```bash
   uvicorn api.index:app --reload
   ```

   The app is now available at `http://localhost:8000`.

> [!NOTE]
> If you ever need to rebuild the Pinecone index from scratch (e.g. after changing the embedding model), run `python recreate_index.py` before re-running `seed_db.py`.

### Environment variables

Create a `.env` file at the project root with the following keys:

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | API key for the LLM endpoint |
| `OPENAI_BASE_URL` | Base URL of the OpenAI-compatible API (default: `https://api.llmod.ai`) |
| `PINECONE_API_KEY` | Pinecone API key |
| `PINECONE_INDEX_NAME` | Pinecone index name (default: `vigilense`) |
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_SECRET_KEY` | Supabase service role (secret) key |

## Project structure

```
├── api/
│   ├── index.py          # FastAPI app — all HTTP routes and data models
│   ├── agent.py          # ReAct loop, system prompt, citation integrity scrubber
│   ├── tools.py          # 10 pharmacovigilance tools
│   ├── config.py         # Shared SDK clients (OpenAI, Pinecone, Supabase)
│   └── architecture.png  # Architecture diagram (served at /api/model_architecture)
├── data/                 # Drug profile JSON files used to seed Pinecone
├── tests/                # End-to-end test suite
├── index.html            # Single-page frontend
├── architecture.html     # Interactive architecture diagram
├── seed_db.py            # Seeds Pinecone with drug profile data
├── recreate_index.py     # Recreates the Pinecone index
├── requirements.txt
└── vercel.json           # Vercel deployment configuration
```

## Tools

The agent has access to 10 tools across 5 categories.

> [!IMPORTANT]
> `query_knowledge_base` is always the mandatory first step. If the drug is not in the formulary, the agent aborts immediately with `drug_not_in_portfolio` — no external API calls are made.

### Knowledge base

**`query_knowledge_base`** — Performs a semantic vector search (RAG) over the internal Pinecone index, which stores FDA label content and safety summaries for formulary drugs. Establishes the known-risk baseline before any external retrieval begins.

**`check_past_signals`** — Queries the Supabase `agent_logs` table to surface any previous investigations on the same drug. Helps the agent understand what signals have already been detected internally.

### Literature

**`fetch_pubmed_advanced`** — Searches PubMed using boolean query syntax (`"drug" AND ("ae1" OR "ae2")`) and retrieves abstracts from a configurable date range (default: 2020–present). The LLM screens all results for relevance before they are added to the report.

**`search_drug_class_effects`** — Searches PubMed by drug class rather than by a specific drug name. Useful for contextualizing a finding within a broader pharmacological class (e.g. "SGLT2 inhibitors AND ketoacidosis").

### Drug profile

**`get_drug_profile`** — Resolves a drug name against OpenFDA to retrieve the current FDA label: active ingredients, drug class, mechanism of action, and brand names. Falls back to RxNorm for name normalization when the OpenFDA lookup is ambiguous.

### Statistics

**`fetch_fda_adverse_events`** — Queries the OpenFDA FAERS database for real-world case counts (drug + adverse event vs. everything else). Returns the 2×2 contingency table values needed for disproportionality analysis.

**`calculate_disproportionality`** — Computes the Reporting Odds Ratio (ROR) and its 95% confidence interval from a 2×2 contingency table. An ROR lower bound > 1 is treated as a potential disproportionate signal.

### Reporting

**`generate_pharmacovigilance_report`** — Assembles the structured Markdown report in CIOMS/ICH E2D format. Sections include: FDA label baseline, novel findings from literature, known/expected findings, and signal assessment. All PubMed citations are rendered as numbered references pointing to real PMIDs retrieved during the investigation.

### Control

**`submit_final_report`** — Signals that the investigation is complete. Terminates the ReAct loop and returns the generated report to the caller.

**`abort_investigation`** — Terminates the loop early when a guardrail condition is met. Supported abort codes: `drug_not_in_portfolio`, `query_too_vague`, `multiple_drugs_detected`, `non_medical_query`, `drug_not_recognized`, `no_literature_found`.

## Formulary

The bundled knowledge base covers 8 drugs: **Adalimumab**, **Atorvastatin**, **Lisinopril**, **Metformin**, **Methotrexate**, **Sertraline**, **Sildenafil**, and **Warfarin**. Queries about any other drug will be rejected at the first step.

## API reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Serves the frontend |
| `POST` | `/api/execute` | Run the ReAct agent on a query |
| `GET` | `/api/agent_info` | Agent description, purpose, and prompt examples |
| `GET` | `/api/team_info` | Team details |
| `GET` | `/api/model_architecture` | Architecture diagram (`image/png`) |
| `GET` | `/api/history` | Recent investigations from Supabase |

### POST /api/execute

**Request body:**
```json
{
  "prompt": "Review the latest PubMed findings on Metformin and lactic acidosis for novel signals."
}
```

**Response:**
```json
{
  "report_markdown": "## Pharmacovigilance Report — Metformin ...",
  "reasoning": null,
  "steps": [
    { "step": 1, "thought": "...", "action": "query_knowledge_base", "observation": "..." },
    { "step": 2, "thought": "...", "action": "fetch_pubmed_advanced",  "observation": "..." }
  ]
}
```

If the agent triggers a guardrail, `report_markdown` is `null` and `reasoning` contains the abort message and code.

## Deployment

The project is pre-configured for Vercel.

1. Push your fork to GitHub.
2. Import the repository at [vercel.com/new](https://vercel.com/new).
3. Add all [environment variables](#environment-variables) in the Vercel project settings.
4. Click **Deploy** — `vercel.json` routes every request to the FastAPI app via `@vercel/python`.

> [!TIP]
> Vercel's free Hobby plan is sufficient for this project. No additional configuration is required beyond setting the environment variables.

## Resources

- [PubMed API](https://www.ncbi.nlm.nih.gov/home/develop/api/) — biomedical literature retrieval
- [OpenFDA API](https://open.fda.gov/apis/) — drug labels and FAERS adverse event database
- [RxNorm API](https://lhncbc.nlm.nih.gov/RxNav/APIs/RxNormAPIs.html) — drug name normalization
- [ICH E2D Guideline](https://www.ich.org/page/pharmacovigilance-guidelines) — post-approval pharmacovigilance reporting standard
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — the agent architecture this project is based on
