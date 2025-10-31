# src/normalize_resave_embeddings.py
"""Normalize (L2) all embeddings in a Chroma collection and re-upsert them.
Safe: reads embeddings in batches, normalizes, then upserts back.
Usage: python src/normalize_resave_embeddings.py
"""
import numpy as np
from pathlib import Path
import math
import sys

# try to import chromadb with compatibility for PersistentClient / Client(Settings)
try:
    import chromadb
    from chromadb.config import Settings
except Exception as e:
    print("Missing chromadb or incompatible version:", e)
    raise SystemExit(1)

PERSIST_DIR = Path("./db/chroma")
COLLECTION_NAME = "mental_health"
BATCH_SIZE = 512  # adjust if you have low RAM

def get_client():
    # try PersistentClient first (newer API), then fallback to Client(Settings)
    try:
        client = chromadb.PersistentClient(path=str(PERSIST_DIR))
        return client
    except Exception:
        try:
            client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=str(PERSIST_DIR)))
            return client
        except Exception as e:
            print("Failed to create chroma client:", e)
            raise

def normalize_all():
    client = get_client()
    try:
        col = client.get_or_create_collection(name=COLLECTION_NAME)
    except Exception as e:
        print("Failed to get/create collection:", e)
        raise SystemExit(1)

    # Try to fetch embeddings (and rely on returned 'ids' if present).
    # Do NOT request include=["ids"] because some Chromadb versions reject that.
    print("Reading embeddings from collection (may take a while)...")
    try:
        res = col.get(include=["embeddings", "metadatas", "documents"])
    except TypeError:
        # older/newer API mismatch - try alternate call
        res = col.get()
    except Exception as e:
        print("Error calling collection.get():", e)
        raise SystemExit(1)

    embs = res.get("embeddings", None)
    ids = res.get("ids", None)
    if embs is None:
        print("No embeddings found in collection. Aborting.")
        raise SystemExit(1)
    if ids is None:
        # If ids are missing, try to create numeric ids (warning)
        print("Warning: collection.get() did not return 'ids'. Attempting to create synthetic ids.")
        ids = [f"item_{i}" for i in range(len(embs))]

    total = len(ids)
    print(f"Found {total} items. Normalizing in batches of {BATCH_SIZE}...")

    # Process in batches to limit memory spike
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch_ids = ids[start:end]
        batch_embs = np.array(embs[start:end], dtype=float)
        # L2 normalize rows
        norms = np.linalg.norm(batch_embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        batch_embs = batch_embs / norms

        # upsert back (idempotent) — keep documents/metadatas if available
        batch_docs = None
        batch_metas = None
        if res.get("documents"):
            batch_docs = res["documents"][start:end]
        if res.get("metadatas"):
            batch_metas = res["metadatas"][start:end]

        # Upsert requires ids + embeddings; include docs/metas if present to avoid losing them
        try:
            if batch_docs is not None and batch_metas is not None:
                col.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_metas, embeddings=batch_embs.tolist())
            elif batch_docs is not None:
                col.upsert(ids=batch_ids, documents=batch_docs, embeddings=batch_embs.tolist())
            elif batch_metas is not None:
                col.upsert(ids=batch_ids, metadatas=batch_metas, embeddings=batch_embs.tolist())
            else:
                col.upsert(ids=batch_ids, embeddings=batch_embs.tolist())
        except Exception as e:
            print(f"Upsert failed for batch {start}:{end} ->", e)
            raise

        print(f"Normalized & upserted items {start}..{end-1}")

    # attempt to persist if client provides persist()
    try:
        client.persist()
    except Exception:
        try:
            client._client.persist()  # fallback internal
        except Exception:
            pass

    print("Normalization complete.")

if __name__ == "__main__":
    normalize_all()
