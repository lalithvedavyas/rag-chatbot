from fastapi import FastAPI
from pydantic import BaseModel
from app.rag_pipeline import load_qa_pipeline

app = FastAPI()
qa = load_qa_pipeline()

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask(q: Query):
    result = qa(q.question)
    return {"answer": result["result"]}
