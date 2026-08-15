# VigiLenseAI - Project Context
**Last updated: Session 10 (August 2026)**

---

## Project Overview
Pharmacovigilance AI agent for a university course (Open University - AI Agents Seminar).
Monitors PubMed literature for a specific drug, compares findings against known data in Pinecone, and alerts on novel risks.

**Deadline: 17/8/2026**

---

## Tech Stack
- **LLM:** LLMod.ai (OpenAI-compatible API, base URL: `https://api.llmod.ai`)
- **Chat model:** `NBUECSE-gpt-5-mini`
- **Embedding model:** `NBUECSE-text-embedding-3-small`
- **Vector DB:** Pinecone (index: `vigilense`, dimensions: **1536**, metric: cosine, serverless AWS us-east-1)
- **Primary DB:** Supabase (tables: `agent_logs`, `past_signals`)
- **Deployment:** Vercel (serverless, max 300s timeout)
- **Budget:** $9 total

---

## API Keys (.env)
```
OPENAI_API_KEY=<your-openai-api-key>
OPENAI_BASE_URL=https://api.llmod.ai

PINECONE_API_KEY=<your-pinecone-api-key>
PINECONE_INDEX_NAME=vigilense

SUPABASE_URL=<your-supabase-url>
SUPABASE_PUBLISHABLE_KEY=<your-supabase-publishable-key>
SUPABASE_SECRET_KEY=<your-supabase-secret-key>
```

---

## Course Requirements (from project.pdf)
- `GET /api/team_info` — team details JSON
- `GET /api/agent_info` — agent purpose, description, prompt templates
- `GET /api/model_architecture` — PNG architecture diagram (file: `api/architecture.png` — NOT YET CREATED)
- `POST /api/execute` — main agent endpoint
- Response format: `{ status, error, response, steps[] }`
- Each step: `{ module, prompt, response }`
- Frontend GUI at root URL `/`

---

## Run Locally
```bash
cd "/Users/aviadalon/Desktop/הפתוחה/סדנת סוכני בינה/VigiLenseAI"
python3 -m uvicorn api.index:app --host 127.0.0.1 --port 8000 --reload
# then open http://127.0.0.1:8000
```
All packages installed via `pip3 install -r requirements.txt` ✅

---

## Project Structure (current)
```
VigiLenseAI/
├── .env                    ✅ API keys set
├── index.html              ✅ Frontend GUI (complete)
├── seed_db.py              ✅ Pinecone seeder (already run)
├── recreate_index.py       ✅ One-time utility (already run)
├── vercel.json             ✅ Vercel deployment config
├── requirements.txt        ✅ Python dependencies
├── CONTEXT.md              ✅ This file
├── data/                   ✅ 8 FDA drug JSON files
│   ├── atorvastatin.json
│   ├── metformin.json
│   ├── warfarin.json
│   ├── sildenafil.json
│   ├── methotrexate.json
│   ├── lisinopril.json
│   ├── sertraline.json
│   └── adalimumab.json
└── api/                    ✅ Backend (modular structure)
    ├── config.py           ✅ SDK clients + constants
    ├── tools.py            ✅ SESSION 10 — 9 tools + LLM screening pipeline
    ├── agent.py            ✅ SESSION 10 — ReAct + guardrails + literature pipeline
    ├── index.py            ✅ FastAPI app + routes
    └── architecture.png    ❌ MISSING — needs to be created
```

---

## What's Done

### Pinecone ✅
- Index `vigilense` recreated with **1536 dimensions** (was wrong 512 before)
- `seed_db.py` ran successfully: **36 vectors** from 8 drugs × 5 sections each
- Sections embedded: `boxed_warning`, `adverse_reactions`, `warnings_and_cautions`, `contraindications`, `drug_interactions`
- Metadata fields per vector: `drug` (lowercase), `section`, `text`, `source_type`

### Supabase ✅
- Table `agent_logs` — created and verified working
  - Fields: `id`, `created_at`, `user_prompt`, `response`, `steps (jsonb)`
- Table `past_signals` — ✅ **EXISTS** (confirmed Session 4)

### Frontend — index.html ✅
- Dark navy (#080E1A) + glassmorphism design
- VigiLenseAI gradient title (centered, no logo)
- 8 neon 3D floating pills animation (realistic, with shine/seam/shadow)
- Drug selection: 4×2 grid of clean (non-colorful) buttons
- Clicking drug → auto-fills textarea with preset query
- Textarea also allows free-text input
- Run Agent button (centered)
- "Monitoring: [Drug]" indicator appears below Run Agent after selection
- Loading spinner, error box, results section with accordion steps
- **Session 6:** Article links open in new tab via custom `marked.js` renderer
- **Session 6:** `.prose-report a` CSS (cyan, underline) for clickable PubMed links

### Backend — api/config.py ✅
- Initializes all SDK clients: OpenAI/LLMod, Pinecone, Supabase
- Defines model name constants: `CHAT_MODEL`, `EMBED_MODEL`, `PINECONE_INDEX`
- No changes needed

### Backend — api/tools.py ✅ (SESSION 9)

**Architecture:** API-Only pipeline. Zero hardcoded dictionaries. All data from live APIs.

**Constants (PubMed):**
- `PUBMED_FETCH_TIMEOUT = 20` — longer timeout for efetch XML
- `SCREENING_BATCH_SIZE = 50` — articles per LLM screening call
- `MAX_PUBMED_SCREEN = 200` — max articles fetched when screening is active
- `HTTP_TIMEOUT = 10` — increased from 5s (session 9) for RxNorm reliability

**Internal helpers:**
- `_pubmed_fetch(term, max_results, min_year=2020)` — two-step: esearch count (retmax=0) → esearch PMIDs → efetch XML. Returns `(articles, audit_info)`. audit_info: `{date_range, total_found, total_fetched}`. sort=relevance. Full abstract (no truncation).
- `_screen_articles_llm(articles, investigation_context)` — LLM batch screening. Sends batches of 50 articles to LLM with investigation context. Returns only relevant articles. Fails open on error.

**9 tools + dispatcher:**

#### 1. `get_drug_profile(drug_name)`
- OpenFDA → RxNorm → safety fallback
- Returns: `drug_name`, `active_ingredients`, `drug_class`, `brand_names`, `mechanism`, `source`
- `source="fallback"` → drug NOT recognized → agent should abort

#### 2. `calculate_disproportionality(a, b, c, d)`
- ROR + 95% CI. Significance: ROR ≥ 2.0 AND ci_lower > 1.0

#### 3. `fetch_pubmed_advanced(query_term, max_results=10, min_year=2020, investigation_context=None)`
- When `investigation_context` provided: fetches up to 200 articles, runs `_screen_articles_llm`
- When not provided: fetches up to `max_results` (legacy behavior)
- audit_info gains `total_relevant` field when screening runs
- Returns: `{query_term, count, results, audit}`
- **ALWAYS include drug name in query_term. ALWAYS pass investigation_context.**
- **SESSION 9: query_term MUST use PubMed boolean grouping:** `"drug" AND ("ae1" OR "ae2" OR "ae3")`. Multi-word terms must be quoted. Without parentheses, OR is global → 400K+ noise results.

#### 4. `search_drug_class_effects(drug_class, adverse_event, max_results=10, min_year=2020, investigation_context=None)`
- Same screening behavior as fetch_pubmed_advanced when investigation_context provided

#### 5. `check_past_signals(drug_name, adverse_event=None)`
- Supabase `past_signals` query — graceful fallback if table missing

#### 6. `query_knowledge_base(drug_name, query, section=None, top_k=3)`
- Pinecone semantic search
- Returns `drug_in_formulary: bool`
  - `True` = drug found in Pinecone KB → continue investigation
  - `False` = drug NOT in portfolio → **immediate `drug_not_in_portfolio` abort** (SESSION 10)

#### 7. `abort_investigation(abort_code, reason)`
- Terminates loop for guardrail violations
- `abort_code` enum: `drug_not_recognized` | `multiple_drugs_detected` | `non_medical_query` | `query_too_vague` | `no_literature_found` | `drug_not_in_portfolio`
- NOTE: `drug_not_in_formulary` was REMOVED as a standalone abort — replaced by compound `drug_not_in_portfolio` condition (see SESSION 10)

#### 8. `generate_pharmacovigilance_report(...)`
- CIOMS/ICH E2D Markdown report
- New field: `article_summaries` — array of `{pmid, relevance_summary}` per retrieved article
- Python injects deterministic `literature_section` before dispatch (LLM never writes citations)
- Returns: `{status, report_markdown, generated_at}`

#### 9. `submit_final_report(...)`
- Terminates the ReAct loop normally

**`dispatch(fn_name, fn_args)`** routes all tools. Strips unexpected keys via `_filter`.

---

### Backend — api/agent.py ✅ (SESSION 9)

**`_build_literature_section(articles, summaries, audit_entries)`:**
- Filters to only articles with validated LLM summaries — no raw abstract fallback
- Audit header: "Found N · Fetched N · Screened N (100%) · X included"
- Numbered articles: `**1.** **Title**\nauthor · journal · year · [PMID link]`
- `**Key Drug-Specific Findings:**` label (not raw abstract)
- `---` separator between articles

**`_scrub_pmids(text, valid_pmids)`:** removes hallucinated PMID citations from LLM text.

**`run_react_loop(user_prompt)` state variables:**
- `collected_articles` — real PubMed articles (after drug_keywords filter)
- `valid_pmids` — whitelist for PMID scrubber
- `drug_keywords` — built from `get_drug_profile` result: drug_name + brand_names + active_ingredients (all lowercase). Used to filter `fetch_pubmed_advanced` articles.
- `pubmed_audit_entries` — audit metadata from each PubMed call
- `MAX_ITERATIONS = 15`

**Literature injection (before generate_pharmacovigilance_report dispatch):**
1. Extract `article_summaries` from LLM args
2. Validate each pmid against `valid_pmids`
3. Call `_build_literature_section(collected_articles, summaries_dict, pubmed_audit_entries)`
4. Scrub hallucinated PMIDs from `summary_findings`

**SYSTEM_PROMPT key rules:**
- fetch_pubmed_advanced: ALWAYS include drug name in query_term + ALWAYS pass investigation_context (drug name, active ingredients, adverse event, demographic context)
- investigation_context triggers full 200-article LLM screening pipeline
- `drug_in_formulary=false` → do NOT abort, note in summary_findings and continue
- article_summaries: INCLUDE only articles with explicit drug-specific findings; EXCLUDE pregnancy/neonatal-only/irrelevant population papers; detailed extraction (study design, efficacy, safety, subgroups, mechanism)
- Demographic terms: add to query when user specifies age group (Adolescent[Mesh], etc.)
- **SESSION 9: summary_findings** — explicit bullet points (`-`) required in every section
- **SESSION 9: PubMed boolean syntax** — `"drug" AND ("ae1" OR "ae2" OR "ae3")` mandatory; without parentheses OR is global → 400K+ noise

**Guardrails (abort conditions):**
- `query_too_vague` — no clear drug name
- `multiple_drugs_detected` — two+ distinct drugs
- `non_medical_query` — not drug safety related
- `drug_not_recognized` — source="fallback" AND name NOT recognizable as a real drug (gibberish/food). Known drugs (Sildenafil, Aspirin etc.) with fallback → do NOT abort
- `no_literature_found` — both PubMed tools returned count=0
- `drug_not_in_portfolio` **(SESSION 10)** — `query_knowledge_base` returns drug_in_formulary=false → **immediate abort**, no further tools called. The 8 drugs in data/ are the company portfolio; anything outside gets this abort. User-friendly message explains the drug is not in the company's pharmacovigilance portfolio.

**SESSION 9 Sildenafil fix (tools.py):**
- Root cause: OpenFDA stores "SILDENAFIL CITRATE" — exact-phrase match on "Sildenafil" fails
- Fix 1: `_get_profile_from_openfda` now tries two queries: exact-phrase match → unquoted term fallback
- Fix 2: `HTTP_TIMEOUT` raised 5s → 10s for RxNorm reliability
- Fix 3: SYSTEM_PROMPT guardrail relaxed — known drugs with source="fallback" → proceed, not abort

---

### Backend — api/index.py ✅
- All 5 routes: `GET /`, `GET /api/team_info`, `GET /api/agent_info`, `GET /api/model_architecture`, `POST /api/execute`

---

## Supabase Tables

### `agent_logs` ✅
```sql
id           bigint (primary key)
created_at   timestamp with time zone
user_prompt  text
response     text
steps        jsonb
```

### `past_signals` ✅ (exists)
```sql
investigation_id uuid default gen_random_uuid() primary key,
created_at       timestamp with time zone default now(),
drug_name        text,
adverse_event    text,
verdict          text,   -- 'escalated' / 'discarded' / 'monitoring'
previous_ror     numeric
```

---

## What's Left To Do
- [ ] **Create `api/architecture.png`** — required by course
- [ ] **Deploy to Vercel** — push to GitHub, import project, set all env vars
- [ ] **End-to-end test** — run agent on each of the 8 drugs
- [ ] **Budget check** — stay within $9

---

## Drug Portfolio (8 drugs in data/)
| Drug | Category |
|------|----------|
| Atorvastatin | Statin / cholesterol |
| Metformin | Diabetes |
| Warfarin | Anticoagulant |
| Sildenafil | PDE5 inhibitor |
| Methotrexate | Chemotherapy / immunosuppressant |
| Lisinopril | ACE inhibitor |
| Sertraline | SSRI antidepressant |
| Adalimumab | TNF inhibitor biologic |

---

## Drug Queries (JS — in index.html)
```js
const DRUG_QUERIES = {
  'Atorvastatin':  'Analyze recent PubMed literature on Atorvastatin adverse effects and identify any novel safety signals not in our database.',
  'Metformin':     'Review the latest PubMed findings on Metformin side effects, especially in elderly patients, and flag any emerging risks.',
  'Warfarin':      'Search recent PubMed abstracts for novel Warfarin drug interactions or bleeding risk signals beyond known data.',
  'Sildenafil':    'Analyze recent PubMed literature on Sildenafil adverse cardiovascular effects and flag any novel safety signals.',
  'Methotrexate':  'Review recent PubMed findings on Methotrexate toxicity, including hepatotoxicity and pulmonary risks, for novel signals.',
  'Lisinopril':    'Search recent PubMed abstracts on Lisinopril adverse effects and identify any novel angioedema or renal risk signals.',
  'Sertraline':    'Analyze recent PubMed literature on Sertraline adverse effects in adolescents and flag any novel safety concerns.',
  'Adalimumab':    'Review the latest PubMed findings on Adalimumab serious infections and malignancy risks for novel safety signals.',
};
```

---

## Notes
- Vercel routes: all traffic → `api/index.py`
- `api/index.py` serves `index.html` from `../index.html` (parent directory)
- Model name for chat: `NBUECSE-gpt-5-mini`
- Tool calling requires OpenAI-compatible `tools` parameter — LLMod.ai supports it
- HTTP_TIMEOUT = 5 seconds for standard API calls; PUBMED_FETCH_TIMEOUT = 20s for efetch
- `_pubmed_fetch` is internal (not exposed to LLM), shared by fetch_pubmed_advanced and search_drug_class_effects
- `_screen_articles_llm` is internal — called automatically inside PubMed tools when investigation_context is provided
- Local server: `python3 -m uvicorn api.index:app --host 127.0.0.1 --port 8000 --reload`
- `drug_not_in_portfolio` (SESSION 10): fires immediately when drug_in_formulary=false — e.g. "Acamol", "Aspirin", any drug not in the 8-drug portfolio. No further tools called.
- TC-02 updated to expect drug_not_in_portfolio (Aspirin not in portfolio). TC-03/TC-05 accept drug_not_in_portfolio OR drug_not_recognized.
- Zero-hallucination architecture: all PMID citations are Python-injected from real tool results; LLM writes only analytical summaries
