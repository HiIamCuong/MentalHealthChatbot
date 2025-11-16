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
    from google import genai
    from google.genai.errors import APIError
    HAS_GEMINI = True
except Exception:
    HAS_GEMINI = False
    genai = None

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

GEMINI_API_KEY = <YOUR_API_KEY>
GEMINI_MODEL = "gemini-2.5-flash"
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

def call_llm(prompt: str, max_tokens: int = 512, model: str = GEMINI_MODEL) -> str:
    """
    If openai package present and OPENAI_API_KEY set -> call OpenAI ChatCompletion.
    Otherwise return a safe offline demo string (no network).
    """
    # Check 1: Thư viện đã được cài chưa?
    if not HAS_GEMINI:
        return ("Demo mode (Thư viện google-genai chưa cài). Install: pip install google-genai.")
        
    # Khởi tạo client với API Key hardcode
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        return f"Demo mode (Lỗi khởi tạo client): {e}"

    # Gọi API Gemini
    try:
        response = client.models.generate_content(
            model=model,
            contents=[
                {"role": "user", "parts": [{"text": prompt}]}
            ],
            config={"max_output_tokens": max_tokens, "temperature": 0.2}
        )
        if response.text:
            return response.text.strip()
        else:
            # Trả về thông báo lỗi nếu LLM không tạo ra được text (trả về None hoặc chuỗi rỗng)
            return f"Demo mode (LLM không tạo ra được câu trả lời. LLM Response: {response.candidates[0].finish_reason if response.candidates else 'Unknown reason'})"
    
    except APIError as e:
        return f"Demo mode (Lỗi API Gemini): {e}"
    except Exception as e:
        return f"Demo mode (Lỗi kết nối/chung): {e}"
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
    MAX_PROMPT_TOKENS = 3500  # Ngưỡng an toàn cho toàn bộ Input 
    
    tokenizer = tiktoken.get_encoding("cl100k_base") 
    
    current_contexts = []
    
    # System prompt
    system_text = "You are an empathetic assistant. You must not provide medical diagnosis. If user is in crisis, instruct them to contact emergency services immediately. Always cite sources by number when referencing facts."
    system_tokens = len(tokenizer.encode(system_text))
    
    # Question
    question_tokens = len(tokenizer.encode("User question: " + question))
    
    # Token tối đa còn lại cho tất cả các context
    context_budget = MAX_PROMPT_TOKENS - system_tokens - question_tokens - 10 # 10 token dự phòng
    
    total_context_tokens = 0
    
    for c in top_contexts:
        # Ước tính token cho mỗi context (bao gồm cả metadata khi build prompt)
        context_text_and_meta = f"Source [X] {Path(c.get('local_path','?')).name} (chars {c.get('start_char', '?')}-{c.get('end_char', '?')}):\n{c.get('text', '')}\n"
        chunk_tokens = len(tokenizer.encode(context_text_and_meta))
        
        if total_context_tokens + chunk_tokens < context_budget:
            current_contexts.append(c)
            total_context_tokens += chunk_tokens
        else:
            # Nếu vượt ngân sách, dừng lại 
            break

    final_contexts = current_contexts # Cập nhật danh sách context đã được cắt gọn
    
    if not final_contexts and top_contexts:
        print(f"[WARN] Contexts bị cắt hết do vượt quá {MAX_PROMPT_TOKENS} tokens. LLM sẽ trả lời mà không có Context.")
    prompt = build_prompt(question, final_contexts)
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

