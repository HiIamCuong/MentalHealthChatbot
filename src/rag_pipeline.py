# src/rag_pipeline.py
"""
RAG pipeline (safe): retrieve from Chroma -> optional rerank -> LLM call (OpenAI or fallback).
- Ensures HF cache is inside project/.cache BEFORE importing HF libs.
- Uses only project/.cache and project/db/chroma (no files outside project).
- No background processes.
"""

import os
from pathlib import Path

# -----------------------------
# 1) PROJECT CACHE CONFIG (MUST BEFORE HF IMPORTS)
# -----------------------------
PROJECT_DIR = Path(__file__).resolve().parents[1]  # project root
CACHE_DIR = PROJECT_DIR / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# set huggingface / transformers / torch cache to project/.cache
os.environ.setdefault("HF_HOME", str(CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
os.environ.setdefault("TORCH_HOME", str(CACHE_DIR))

# -----------------------------
# 2) imports (safe now)
# -----------------------------
from sentence_transformers import SentenceTransformer
import chromadb
import tiktoken
from typing import List

# optional openai
try:
    import openai
except Exception:
    openai = None

# optional reranker; re_ranker.py provided separately
try:
    from re_ranker import ReRanker
except Exception:
    ReRanker = None

# -----------------------------
# 3) CONFIG
# -----------------------------
PERSIST_DIR = PROJECT_DIR / "db" / "chroma"
PERSIST_DIR.mkdir(parents=True, exist_ok=True)  # ensure folder exists
COLLECTION_NAME = "mental_health"
RETRIEVE_K = 20
FINAL_K = 5
ENCODER_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# -----------------------------
# 4) HELPERS
# -----------------------------
def init_clients():
    """
    Initialize embedding encoder and chroma collection.
    SentenceTransformer will download to project/.cache if needed.
    """
    enc = SentenceTransformer(ENCODER_NAME)
    client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    coll = client.get_or_create_collection(name=COLLECTION_NAME)
    return enc, coll

def build_prompt(question: str, contexts: List[dict]) -> str:
    ctx_texts = []
    for i, c in enumerate(contexts, 1):
        lp = c.get("local_path", "<unknown>")
        sc = c.get("start_char", "?")
        ec = c.get("end_char", "?")
        snippet = c.get("text", "").strip()
        if len(snippet) > 2000:
            snippet = snippet[:2000] + "..."
        src = f"[{i}] {Path(lp).name} (chars {sc}-{ec})"
        ctx_texts.append(f"Source {src}:\n{snippet}\n")
    system = (
        "You are an empathetic assistant. You must not provide medical diagnosis. "
        "If user is in crisis, instruct them to contact emergency services immediately. "
        "Always cite sources by number when referencing facts."
    )
    prompt = system + "\n\n" + "CONTEXT:\n" + "\n\n".join(ctx_texts) + "\n\nUser question: " + question
    return prompt

def call_llm(prompt: str, max_tokens: int = 512, model: str = "gpt-4o-mini") -> str:
    """
    If openai package present and OPENAI_API_KEY set -> call OpenAI ChatCompletion.
    Otherwise return a safe offline demo string (no network).
    """
    if openai is None:
        return (
            "Demo mode (openai package not installed). Install 'openai' and set OPENAI_API_KEY "
            "to enable real LLM calls.\n\nPrompt preview:\n" + (prompt[:1200] + "..." if len(prompt) > 1200 else prompt)
        )
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return (
            "Demo mode (OPENAI_API_KEY not set). Set OPENAI_API_KEY to enable real LLM responses.\n\n"
            "Prompt preview:\n" + (prompt[:1200] + "..." if len(prompt) > 1200 else prompt)
        )
    openai.api_key = key
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[{"role":"system","content":"You are a helpful assistant."},
                  {"role":"user","content":prompt}],
        max_tokens=max_tokens,
        temperature=0.2
    )
    return resp["choices"][0]["message"]["content"]

# -----------------------------
# 5) RAG pipeline
# -----------------------------
def retrieve_and_answer(question: str, use_rerank: bool = True, reranker_model: str = None):
    enc, coll = init_clients()
    q_emb = enc.encode(question, convert_to_numpy=True).tolist()

    # retrieve from chroma
    results = coll.query(query_embeddings=[q_emb], n_results=RETRIEVE_K)
    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    candidates = []
    for d, md in zip(docs, metadatas):
        if not d:
            continue
        rec = {"text": d}
        if isinstance(md, dict):
            rec.update(md)
        candidates.append(rec)

    # optional rerank
    if use_rerank and candidates and ReRanker is not None:
        try:
            rr = ReRanker(model_name=reranker_model) if reranker_model else ReRanker()
            texts = [c["text"] for c in candidates]
            idxs, scores = rr.rerank(question, texts)
            ordered = [candidates[i] for i in idxs]
        except Exception as e:
            print("Reranker failed, falling back to retrieval order. Error:", e)
            ordered = candidates
    else:
        ordered = candidates

    top_contexts = ordered[:FINAL_K]
    prompt = build_prompt(question, top_contexts)
    answer = call_llm(prompt)
    return answer, top_contexts

# -----------------------------
# 6) CLI demo (no background)
# -----------------------------
if __name__ == "__main__":
    q = input("Query> ").strip()
    if not q:
        print("No query provided. Exiting.")
    else:
        ans, ctx = retrieve_and_answer(q, use_rerank=False)
        print("\n=== ANSWER ===\n")
        print(ans[:2000] + ("..." if len(ans) > 2000 else ""))
        print("\n=== SOURCES ===\n")
        for i, c in enumerate(ctx, start=1):
            print(f"[{i}] doc_id={c.get('doc_id','?')} src={c.get('local_path','?')} chars={c.get('start_char','?')}-{c.get('end_char','?')}")
