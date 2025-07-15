import os
from fastapi import FastAPI, UploadFile, File, Form
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

load_dotenv()

app = FastAPI()

CHROMA_PATH = "chroma"
DATA_PATH = "data/books"
EMBED_MODEL = "all-MiniLM-L6-v2"

def get_embedding_db():
    embedding_function = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    return db

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    # 1. Save PDF
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    file_path = os.path.join(DATA_PATH, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # 2. Load/split/embed PDF, add to Chroma
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=100, length_function=len, add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    db = get_embedding_db()
    db.add_documents(chunks)
    db.persist()
    return {"message": f"{file.filename} uploaded and indexed.", "chunks": len(chunks)}

@app.post("/query/")
async def query_db(question: str = Form(...)):
    db = get_embedding_db()
    results = db.similarity_search_with_relevance_scores(question, k=3)
    # Debug print
    print("Top retrieved chunks and scores:")
    for doc, score in results:
        print(f"Score: {score:.2f}")
        print(doc.page_content[:500])
        print("---")
    if not results or results[0][1] < 0.3:
        return {"answer": "The provided documents do not contain sufficient information to answer this question."}

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    PROMPT_TEMPLATE = """
You are a factual, no-nonsense AI assistant trained to answer questions about the DWEXO enterprise management platform.

- Respond only with information present in the vector database from the DWEXO corporate presentation or verifiable expert knowledge.
- If the question cannot be fully answered based on the available information, explicitly state: "The provided documents do not contain sufficient information to answer this question."
- When possible, specify if your answer refers to a product module, service, or support level.
- Be concise and avoid unnecessary details or speculation.

Context:
{context}

Question:
{question}
"""
    prompt = PROMPT_TEMPLATE.format(context=context_text, question=question)

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is not set in the environment or .env file!")
    model = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama3-70b-8192",
    )
    response = model.predict(prompt)
    return {
        "answer": response,
        "sources": [doc.metadata.get("source", None) for doc, _ in results]
    }
