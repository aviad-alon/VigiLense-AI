"""
One-time script: delete the existing Pinecone index (512-dim) and recreate
it with the correct 1536 dimensions to match NBUECSE-text-embedding-3-small.
Run this ONCE before seed_db.py.
"""

import os, time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "vigilense")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# 1. Delete existing index
print(f"Deleting index '{INDEX_NAME}'...")
pc.delete_index(INDEX_NAME)
print("Deleted.")

# 2. Recreate with correct dimension
print(f"Creating index '{INDEX_NAME}' with 1536 dimensions...")
pc.create_index(
    name=INDEX_NAME,
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)

# 3. Wait until ready
print("Waiting for index to be ready", end="", flush=True)
while not pc.describe_index(INDEX_NAME).status.ready:
    print(".", end="", flush=True)
    time.sleep(2)
print("\nIndex is ready!")
