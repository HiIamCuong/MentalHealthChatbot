# src/rag_pipeline.py
"""
RAG pipeline (safe): retrieve from Chroma -> optional rerank -> LLM call (Groq or fallback).
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

try:
    from groq import Groq
    from groq import APIError as GroqAPIError
    HAS_GROQ = True
except Exception:
    HAS_GROQ = False
    Groq = None
    GroqAPIError = None
    print("WARNING: Thư viện 'groq' chưa cài đặt. Chạy: pip install groq")

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


GROQ_API_KEY = <YOUR_GROQ_API_KEY>
GROQ_MODEL = "llama-3.1-8b-instant"

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

def call_llm(prompt: str, max_tokens: int = 512, model: str = GROQ_MODEL) -> str:
    # 1. KIỂM TRA API KEY TRƯỚC
    if not GROQ_API_KEY:
        return "Demo mode (Lỗi: Vui lòng dán GROQ_API_KEY vào mục CONFIG.)"
        
    if not HAS_GROQ:
        return "Demo mode (Lỗi: Thư viện 'groq' chưa được cài đặt.)"
        
    try:
        # 2. KHỞI TẠO CLIENT VÀ GỌI API
        client = Groq(api_key=GROQ_API_KEY)

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "user", "content": prompt} # Gửi prompt đã được xây dựng
                ],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

    except GroqAPIError as e:
        # 3. XỬ LÝ LỖI API CỦA GROQ
        status_code = getattr(e, 'status_code', 'Unknown')
        print(f"[ERROR] Lỗi Groq API ({status_code}): {e.message}")
        return f"Demo mode (Lỗi Groq API): Lỗi {status_code}. Vui lòng kiểm tra API Key hoặc hạn mức."
        
    except Exception as e:
        # 4. XỬ LÝ LỖI CHUNG (Kết nối,...)
        print(f"[ERROR] Lỗi kết nối/chung Groq: {e}")
        return f"Demo mode (Lỗi kết nối Groq): {e}"

    # 5. ĐẢM BẢO LUÔN CÓ KẾT QUẢ TRẢ VỀ NẾU CÓ LỖI XẢY RA
    return "Demo mode (LLM call failed unexpectedly.)"
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
