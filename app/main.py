from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from app.rag import query_rag, build_vectordb_if_missing


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs at startup
    build_vectordb_if_missing()
    yield
    # Runs at shutdown (optional cleanup here)


app = FastAPI(lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "RAG API running 🚀"}

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
async def query_api(request: QueryRequest):
    answer = query_rag(request.question)
    return {"answer": answer}
