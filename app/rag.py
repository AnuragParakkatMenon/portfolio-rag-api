import os
from pypdf import PdfReader
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
load_dotenv()


api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not set")

client = OpenAI(api_key=api_key)
embeddings = OpenAIEmbeddings()

DATA_DIR = "/data"
FAISS_PATH = f"{DATA_DIR}/faiss"

os.makedirs(DATA_DIR, exist_ok=True)


def build_vectordb_if_missing():
    if os.path.exists(f"{FAISS_PATH}/index.faiss"):
        print("✅ Vector DB already exists.")
        return

    print("🔄 Building vector DB...")

    reader = PdfReader("Anurag-Menon.pdf")  # 👈 your PDF must be committed
    text = ""

    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.create_documents([text])

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(FAISS_PATH)

    print("✅ Vector DB built successfully.")


def query_rag(question: str) -> str:
    vectorstore = FAISS.load_local(
        FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = vectorstore.similarity_search(question, k=4)
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
Answer like you are a Senior developer whose resume has been uploaded as a PDF. Be precise and technical with your answer and answer like you have 6years of experient in front end development especially react,typescript, fastapi etc".

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content
