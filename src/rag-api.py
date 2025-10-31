# src/rag_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_pipeline import retrieve_and_answer
import uvicorn

app = FastAPI(title="MentalHealth RAG API")

class QueryRequest(BaseModel):
    question: str
    rerank: bool = True

@app.post("/query")
async def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Empty question")
    # basic safety check — block certain keywords? (tùy bạn)
    answer, contexts = retrieve_and_answer(req.question, use_rerank=req.rerank)
    return {"answer": answer, "sources": [{"doc_id": c.get("doc_id"), "start_char": c.get("start_char"), "local_path": c.get("local_path")} for c in contexts]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
