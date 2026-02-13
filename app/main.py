from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from app.rag import query_rag, build_vectordb_if_missing
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs at startup
    build_vectordb_if_missing()
    yield
    # Runs at shutdown (optional cleanup here)


app = FastAPI(lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str

# Add this CORS middleware
origins = [
    "http://localhost:5173",  # your dev frontend
    "https://rag-ui-production.up.railway.app/",  # production frontend
    "https://rag-ui-production.up.railway.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # allow your frontend domain
    allow_credentials=True,
    allow_methods=["*"],         # allow POST, GET, OPTIONS, etc.
    allow_headers=["*"]          # allow Content-Type, Authorization, etc.
)



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
