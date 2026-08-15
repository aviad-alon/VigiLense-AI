import os
import json
import glob
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

# --- Configuration ---
DATA_DIR = "data/*.json"
EMBED_MODEL = "NBUECSE-text-embedding-3-small"
BATCH_SIZE = 100
MAX_CHARS = 6000  # safe limit for embedding model

# Sections to embed per drug (most relevant for pharmacovigilance)
SECTIONS_TO_EMBED = [
    "boxed_warning",
    "adverse_reactions",
    "warnings_and_cautions",
    "contraindications",
    "drug_interactions",
]

# --- OpenAI client (used for embeddings via llmod) ---
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

# --- Pinecone client ---
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))


def load_drugs(data_dir: str) -> list:
    """
    Load all drug JSON files from the data directory.

    Each file contains an FDA drug label with structured sections
    (adverse_reactions, warnings, contraindications, etc.).

    Args:
        data_dir: Glob pattern pointing to the JSON files (e.g. "data/*.json").

    Returns:
        A list of dicts, one per drug file.
    """
    files = glob.glob(data_dir)
    drugs = []
    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as f:
            drugs.append(json.load(f))
    print(f"Loaded {len(drugs)} drug files: {[d.get('drug_name') for d in drugs]}")
    return drugs


def prepare_chunks(drugs: list) -> list:
    """
    Iterate over all drugs and create one chunk per relevant section.

    For each drug, extracts the pharmacovigilance-relevant sections
    (adverse reactions, warnings, contraindications, etc.) and wraps
    each section in a dict with metadata.

    Args:
        drugs: List of drug dicts loaded from JSON files.

    Returns:
        A flat list of dicts, one dict per section chunk.
        Example: [{"drug_name": "Warfarin", "section": "adverse_reactions",
                   "search_name": "warfarin", "text": "..."}, ...]
    """
    all_chunks = []

    for drug in drugs:
        drug_name = drug.get("drug_name", "Unknown")
        search_name = drug.get("search_name", drug_name.lower())
        sections = drug.get("sections", {})

        for section_name in SECTIONS_TO_EMBED:
            content = sections.get(section_name)
            if not content or not content.strip():
                continue

            text = f"Drug: {drug_name}\nSection: {section_name.replace('_', ' ').title()}\n\n{content}"

            all_chunks.append({
                "id": f"{search_name}_{section_name}",
                "drug_name": drug_name,
                "search_name": search_name,
                "section": section_name,
                "source": drug.get("source", "FDA Drug Label"),
                "text": text[:MAX_CHARS],
            })

    return all_chunks


def embed_chunks(chunks: list) -> list:
    """
    Embed all chunks using the LLMod embedding model, in batches.

    Sends chunks in groups of BATCH_SIZE to avoid large single requests.
    Each chunk dict is updated in-place with a "vector" field containing
    the 1536-dimensional embedding returned by the model.

    Args:
        chunks: List of chunk dicts (must have a "text" field each).

    Returns:
        The same list of chunk dicts, each now containing a "vector" field.
    """
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c["text"] for c in batch]

        response = client.embeddings.create(model=EMBED_MODEL, input=texts)

        for j, item in enumerate(response.data):
            batch[j]["vector"] = item.embedding

        print(f"Embedded {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)} chunks")

    return chunks


def upsert_to_pinecone(chunks: list) -> None:
    """
    Upload embedded chunks to Pinecone in batches.

    Each chunk is stored as a vector with an ID and metadata so it can be
    retrieved and traced back to its source drug and section later.

    The vector ID format is "{search_name}_{section}" (e.g. "warfarin_adverse_reactions").
    Metadata includes: drug_name, search_name, section, source, and the original text.

    Args:
        chunks: List of chunk dicts, each must have "vector", "id",
                "drug_name", "search_name", "section", and "text" fields.
    """
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        vectors = [
            {
                "id": c["id"],
                "values": c["vector"],
                "metadata": {
                    "drug_name": c["drug_name"],
                    "search_name": c["search_name"],
                    "section": c["section"],
                    "source": c["source"],
                    "text": c["text"],
                },
            }
            for c in batch
        ]
        index.upsert(vectors=vectors)
        print(f"Upserted {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)} chunks to Pinecone")


if __name__ == "__main__":
    drugs = load_drugs(DATA_DIR)

    # Step 1: split each drug into section chunks
    all_chunks = prepare_chunks(drugs)
    print(f"Total chunks: {len(all_chunks)}")

    # Step 2: embed all chunks using the embedding model
    all_chunks = embed_chunks(all_chunks)

    # Step 3: upload vectors + metadata to Pinecone
    upsert_to_pinecone(all_chunks)
    print("Done! All drug knowledge uploaded to Pinecone.")
