"""
config.py — SDK clients and shared constants.
Imported by tools.py and agent.py.
"""

import os
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
from pinecone import Pinecone
from supabase import create_client, Client

# ── Debug trace flag ─────────────────────────────────────────────────────────
# Set to True to print step-by-step crash diagnostics.
# Flip to False before project submission for clean output.
DEBUG_TRACE = True

# Shared mutable trace log — populated by _trace() in agent.py and tools.py.
# Reset at the start of each run_react_loop call. Never import as "from config import trace_log"
# (that copies the reference at import time) — always access as config.trace_log.
trace_log: list = []

# ── Model names ───────────────────────────────────────────────────────────────
CHAT_MODEL     = "NBUECSE-gpt-5-mini"
EMBED_MODEL    = "NBUECSE-text-embedding-3-small"
PINECONE_INDEX = os.getenv("PINECONE_INDEX_NAME", "vigilense")

# ── LLM client (LLMod.ai — OpenAI-compatible) ─────────────────────────────────
llm_client = None
_api_key   = os.getenv("OPENAI_API_KEY")
if _api_key:
    llm_client = OpenAI(
        api_key  = _api_key,
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.llmod.ai"),
    )

# ── Pinecone client ───────────────────────────────────────────────────────────
pc = None
_pinecone_key = os.getenv("PINECONE_API_KEY")
if _pinecone_key:
    pc = Pinecone(api_key=_pinecone_key)

# ── Supabase client ───────────────────────────────────────────────────────────
supabase: Client | None = None
_supa_url = os.getenv("SUPABASE_URL")
_supa_key = os.getenv("SUPABASE_SECRET_KEY")
if _supa_url and _supa_key:
    supabase = create_client(_supa_url, _supa_key)
