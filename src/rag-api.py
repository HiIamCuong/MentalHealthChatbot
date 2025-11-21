# -*- coding: utf-8 -*- #
# src/rag_api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import sys
import traceback

# Thử import pipeline, nếu lỗi thì báo rõ ràng
try:
    from rag_pipeline import retrieve_and_answer
except ImportError as e:
    print(f"LỖI IMPORT: Không thể load 'rag_pipeline'. Chi tiết: {e}")

app = FastAPI(title="MentalHealth RAG API")

# --- CẤU HÌNH CORS (CHO PHÉP HTML KẾT NỐI) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép mọi file HTML kết nối
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép POST, GET
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    rerank: bool = False

@app.post("/query")
async def query(req: QueryRequest):
    # 1. Kiểm tra câu hỏi rỗng
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Empty question")

    print(f"Đã nhận câu hỏi: {req.question}") # In ra terminal để bạn biết có tin nhắn đến

    try:
        # 2. Gọi Pipeline xử lý
        answer, contexts = retrieve_and_answer(req.question, use_rerank=req.rerank)
        
        # 3. Trả về kết quả
        return {
            "answer": answer, 
            "sources": [
                {
                    "doc_id": c.get("doc_id"), 
                    "start_char": c.get("start_char"), 
                    "local_path": c.get("local_path")
                } for c in contexts
            ]
        }
    except Exception as e:
        # Nếu lỗi, in chi tiết ra màn hình đen (Terminal) để sửa
        traceback.print_exc() 
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("Server đang khởi động... Mời bạn mở file index.html")
    uvicorn.run(app, host="0.0.0.0", port=8000)
