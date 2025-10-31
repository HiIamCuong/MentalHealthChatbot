#!/usr/bin/env python3
"""
Embed chunks and upsert into Chroma (Day 5).
- Everything written under project/ (cache, db, logs).
- Run: python src/embed_and_store_chroma.py [--args]
"""
import os
from pathlib import Path
import json
import time
import argparse
import logging

# project layout
PROJECT_DIR = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_DIR / ".cache"
DB_DEFAULT = PROJECT_DIR / "db" / "chroma"
LOG_DIR = PROJECT_DIR / "logs"
CHUNKS_DEFAULT = PROJECT_DIR / "data" / "chunks_token" / "chunks.jsonl"

# ensure project subfolders exist
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
(DB_DEFAULT).mkdir(parents=True, exist_ok=True)

# keep transformer/huggingface cache inside project/.cache
os.environ.setdefault("HF_HOME", str(CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
os.environ.setdefault("TORCH_HOME", str(CACHE_DIR))

# logging
logger = logging.getLogger("day5")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# small utilities
def load_chunks(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                logger.warning("failed parse chunk line: %s", e)

def batchify(iterable, n):
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch

def parse_args():
    p = argparse.ArgumentParser(description="Embed chunks -> upsert into Chroma (persistent in project/db)")
    p.add_argument("--chunks", type=str, default=str(CHUNKS_DEFAULT), help="path to chunks.jsonl")
    p.add_argument("--persist-dir", type=str, default=str(DB_DEFAULT), help="where to persist chroma DB (under project/)")
    p.add_argument("--collection-name", type=str, default="mental_health")
    p.add_argument("--model-name", type=str, default="paraphrase-multilingual-MiniLM-L12-v2")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--truncate-chars", type=int, default=2000)
    p.add_argument("--dry-run", action="store_true", help="compute embeddings but do NOT upsert")
    p.add_argument("--max-items", type=int, default=0, help="if >0 process only up to this many chunks")
    return p.parse_args()

def main():
    args = parse_args()
    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        logger.error("Chunks file not found: %s", chunks_path)
        raise SystemExit(1)

    logger.info("Using HF cache at %s", CACHE_DIR)
    logger.info("Persist dir: %s", args.persist_dir)
    logger.info("Loading sentence-transformers model: %s", args.model_name)

    # import heavy deps after env var setup
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        logger.exception("Missing package sentence_transformers. Please 'pip install sentence-transformers'. Error: %s", e)
        raise SystemExit(1)

    try:
        import chromadb
    except Exception as e:
        logger.exception("Missing package chromadb. Please 'pip install chromadb'. Error: %s", e)
        raise SystemExit(1)

    # initialize model
    model = SentenceTransformer(args.model_name)

    # prepare chroma client (new API: PersistentClient). Fallback to legacy Client if needed.
    persist_dir = Path(args.persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    try:
        # New API
        client = chromadb.PersistentClient(path=str(persist_dir))
        logger.info("Using chromadb.PersistentClient (path=%s)", persist_dir)
    except Exception as e:
        logger.warning("PersistentClient not available or failed: %s. Trying legacy Client(Settings(...))", e)
        try:
            from chromadb.config import Settings
            client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=str(persist_dir)))
            logger.info("Using legacy chromadb.Client(Settings(...))")
        except Exception as e2:
            logger.exception("Failed to initialize chroma client: %s", e2)
            raise SystemExit(1)

    # get or create collection robustly
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

    # limit generator if needed
    chunks_iter = load_chunks(chunks_path)
    if args.max_items and args.max_items > 0:
        def limited(gen, limit):
            i = 0
            for x in gen:
                if i >= limit:
                    break
                yield x
                i += 1
        chunks_iter = limited(chunks_iter, args.max_items)

    total = 0
    t0 = time.time()

    for batch in batchify(chunks_iter, args.batch_size):
        docs = []
        ids = []
        metadatas = []
        for c in batch:
            txt = c.get("text", "") or ""
            if args.truncate_chars and args.truncate_chars > 0:
                txt = txt[: args.truncate_chars]
            docs.append(txt)
            ids.append(c.get("id"))
            metadatas.append({
                "doc_id": c.get("doc_id", ""),
                "start_char": c.get("start_char", ""),
                "end_char": c.get("end_char", ""),
                "start_token": c.get("start_token", ""),
                "end_token": c.get("end_token", ""),
                "local_path": c.get("local_path", ""),
                "created_at": c.get("created_at", ""),
            })

        # compute embeddings
        emb = model.encode(docs, batch_size=args.batch_size, show_progress_bar=False, convert_to_numpy=True)
        logger.info("Computed embeddings for batch size %d -> emb.shape=%s", len(ids), emb.shape)
        total += len(ids)

        if args.dry_run:
            logger.info("[DRY-RUN] Skipping upsert for this batch.")
            continue

        # upsert into chroma
        try:
            # prefer upsert (idempotent)
            collection.upsert(ids=ids, metadatas=metadatas, documents=docs, embeddings=emb.tolist())
        except Exception as e:
            logger.warning("upsert failed (%s). Trying add()", e)
            try:
                collection.add(ids=ids, metadatas=metadatas, documents=docs, embeddings=emb.tolist())
            except Exception as e2:
                logger.exception("Failed to add embeddings: %s", e2)
                raise

    elapsed = time.time() - t0
    logger.info("[DONE] processed %d chunks in %.1fs (%.2f chunks/s)", total, elapsed, (total / elapsed) if elapsed > 0 else 0.0)

    # persist if available
    try:
        if hasattr(client, "persist"):
            client.persist()
            logger.info("Client.persist() called.")
    except Exception as e:
        logger.warning("client.persist() failed: %s", e)

    # collection count
    try:
        count = collection.count()
        logger.info("collection.count() = %s", count)
    except Exception:
        logger.info("collection.count() not available or failed to fetch.")

if __name__ == "__main__":
    main()
