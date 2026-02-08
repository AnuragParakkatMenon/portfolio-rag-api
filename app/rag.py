import os
from pypdf import PdfReader
from openai import OpenAI
from app.vector_store import VectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



os.makedirs("/data", exist_ok=True)

embeddings = OpenAIEmbeddings()

def ingest_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.create_documents([text])

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("/data/faiss")


def query_rag(question: str):
    vectorstore = FAISS.load_local(
        "/data/faiss",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = vectorstore.similarity_search(question, k=4)
    return docs


# Vector store (384 = embedding dimension)
vector_store = VectorStore(dim=384)

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def ingest_pdf(file_path: str):
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()

    chunks = chunk_text(text)
    embeddings = embedding_model.encode(chunks)

    vector_store.add(embeddings, chunks)

def query_rag(question: str) -> str:
    query_embedding = embedding_model.encode(question)
    contexts = vector_store.search(query_embedding)

    prompt = f"""
You are a helpful assistant.
Answer the question using ONLY the context below in the language provided by the user by default usee English.
If the answer is not in the context, say "I don't know".

Context:
{contexts}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content
