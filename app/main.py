import os
import shutil
from fastapi import FastAPI, UploadFile, File
from app.models import QueryRequest, QueryResponse
from app.rag import ingest_pdf, query_rag

app = FastAPI(title="RAG API")

UPLOAD_DIR = "/data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


UPLOAD_DIR = "/tmp/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

from fastapi import FastAPI

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ingest-pdf")
async def ingest_pdf_api(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    ingest_pdf(file_path)
    return {"message": "PDF ingested successfully"}

@app.post("/query", response_model=QueryResponse)
def query_api(request: QueryRequest):
    answer = query_rag(request.question)
    return {"answer": answer}
