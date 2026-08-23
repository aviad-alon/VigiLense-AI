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

# ── Model names ───────────────────────────────────────────────────────────────

CHAT_MODEL  = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"

PINECONE_INDEX = os.getenv("PINECONE_INDEX_NAME", "vigilense")

# ── LLM client ────────────────────────────────────────────────────────────────

llm_client = None
_api_key   = os.getenv("OPENAI_API_KEY")
if _api_key:
    llm_client = OpenAI(
        api_key  = _api_key,
        base_url = "https://api.openai.com/v1",
        timeout  = 90.0,
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
