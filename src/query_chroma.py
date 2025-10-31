#!/usr/bin/env python3
"""
Interactive query script for Chroma created by embed_and_store_chroma.py
Run: python src/query_chroma.py
"""
import os
from pathlib import Path
import argparse
import logging

PROJECT_DIR = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_DIR / ".cache"
DB_DEFAULT = PROJECT_DIR / "db" / "chroma"
LOG_DIR = PROJECT_DIR / "logs"

# ensure cache dir exists and keep it inside project/
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
os.environ.setdefault("TORCH_HOME", str(CACHE_DIR))

# logging
logger = logging.getLogger("day5_query")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
ch.setFormatter(fmt)
logger.addHandler(ch)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--persist-dir", type=str, default=str(DB_DEFAULT))
    p.add_argument("--collection-name", type=str, default="mental_health")
    p.add_argument("--model-name", type=str, default="paraphrase-multilingual-MiniLM-L12-v2")
    p.add_argument("--k", type=int, default=5)
    return p.parse_args()

def main():
    args = parse_args()
    persist_dir = Path(args.persist_dir)
    if not persist_dir.exists():
        logger.error("Persist dir not found: %s. Run embed_and_store_chroma.py first.", persist_dir)
        raise SystemExit(1)

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        logger.exception("Missing sentence-transformers: %s", e)
        raise SystemExit(1)

    try:
        import chromadb
    except Exception as e:
        logger.exception("Missing chromadb: %s", e)
        raise SystemExit(1)

    # init client
    try:
        client = chromadb.PersistentClient(path=str(persist_dir))
    except Exception:
        from chromadb.config import Settings
        client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=str(persist_dir)))

    # get collection
    try:
        if hasattr(client, "get_or_create_collection"):
            collection = client.get_or_create_collection(name=args.collection_name)
        else:
            try:
                collection = client.get_collection(args.collection_name)
            except Exception:
                collection = client.create_collection(args.collection_name)
    except Exception as e:
        logger.exception("Failed to get/create collection: %s", e)
        raise SystemExit(1)

    model = SentenceTransformer(args.model_name)
    logger.info("Ready. Type query (type 'exit' to quit).")

    while True:
        q = input("\nQuery> ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            logger.info("Exiting.")
            break
        emb = model.encode(q, convert_to_numpy=True)
        try:
            results = collection.query(query_embeddings=[emb.tolist()], n_results=args.k)
        except Exception as e:
            logger.exception("Query failed: %s", e)
            continue

        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0] if "distances" in results else None

        print(f"\nTop {len(ids)} results:")
        for i, _id in enumerate(ids):
            print(f"\n[{i+1}] id: {_id}")
            md = metadatas[i] if i < len(metadatas) else {}
            print("metadata:", md)
            snippet = docs[i] if i < len(docs) else ""
            print("snippet:", snippet[:400].replace("\n", " ") + ("..." if len(snippet) > 400 else ""))
            if distances:
                try:
                    print("score:", distances[i])
                except Exception:
                    pass

if __name__ == "__main__":
    main()
