# 📁 embed.py
import os
from fastapi import APIRouter, UploadFile, File, Form
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from .utils import DATA_PATH, get_embedding_db as _get_embedding_db, calculate_md5, MIN_SCORE
from langdetect import detect
import re

router = APIRouter()

# Cache the embedding DB
cached_db = None
def get_cached_db():
    global cached_db
    if cached_db is None:
        cached_db = _get_embedding_db()
    return cached_db

# 🧠 In-memory short-term chat memory
chat_memory = []

@router.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    os.makedirs(DATA_PATH, exist_ok=True)
    dest = os.path.join(DATA_PATH, file.filename)
    with open(dest, "wb") as f:
        f.write(await file.read())
    return {"message": f"{file.filename} saved to {DATA_PATH}"}

@router.post("/embed_new_pdfs/")
def embed_new_pdfs():
    db = get_cached_db()
    existing = db.get(include=["metadatas"])["metadatas"]
    seen_hash = {m.get("file_hash") for m in existing if m.get("file_hash")}

    new_chunks = []
    for fname in os.listdir(DATA_PATH):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(DATA_PATH, fname)
        h = calculate_md5(path)
        if h in seen_hash:
            continue

        docs = PyPDFLoader(path).load()
        for d in docs:
            d.metadata["source"] = fname
            d.metadata["file_hash"] = h

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300, chunk_overlap=100,
            length_function=len, add_start_index=True
        )
        chunks = [c for c in splitter.split_documents(docs) if c.page_content.strip()]
        new_chunks.extend(chunks)

    if not new_chunks:
        return {"message": "No new PDFs to embed."}

    db.add_documents(new_chunks)
    return {"message": f"Embedded {len(new_chunks)} new chunks."}

@router.post("/query/")
async def query_db(
    question: str = Form(...),
    debug: bool = Form(False),
):
    global chat_memory
    normalized = question.lower().strip()

    # 🧠 Handle memory recall (flexible)
    if re.search(r"what.*(ask|say)|chnowa.*(9olt|soutlek)", normalized):
        if not chat_memory:
            return {"answer": "You haven’t asked anything yet.", "sources": []}
        memory_lines = "\n".join(f"- {q}" for q in chat_memory)
        return {
            "answer": f"You previously asked:\n{memory_lines}",
            "sources": []
        }

    # 💾 Save question to memory
    chat_memory.append(normalized)

    greetings_map = {
        "hi": "Hi! I'm the assistant of DWEXO. How can I help you?",
        "hello": "Hello! I'm the assistant of DWEXO. What can I help you with?",
        "aselma": "Aslema! 😊 I'm the DWEXO assistant. Kif nجم نعاونك?",
        "salam": "Salam! I'm the DWEXO assistant. How can I help you?",
        "labes": "Labes! Rani m3ak 7ata l5ir. T7eb ts2el 3la chi?",
        "good morning": "Good morning! I'm the DWEXO assistant. What can I do for you?",
        "good evening": "Good evening! I'm the DWEXO assistant. Need help?"
    }

    dialect_map = {
        "ahkili 3ala dwexo": {
            "en": "what are the functionalities of dwexo?",
            "tn": "DWEXO platform t3awedk tsayyer l'operations mta3ek. 3andha CRM, Eshop, Pro, Enterprise..."
        },
        "chnowa tamel dwexo": {
            "en": "what are the functionalities of dwexo?",
            "tn": "DWEXO t3awnek b barcha 7alloul: CRM, e-shop, Pro w Enterprise. T7eb t3raf aktar?"
        },
        "chnowa dwexo": {
            "en": "what is dwexo?",
            "tn": "DWEXO hiya plateforme ta3 gestion li tsayyer les entreprises b surfa." 
        }
    }

    # Direct greeting
    if normalized in greetings_map:
        return {"answer": greetings_map[normalized], "sources": []}

    # Direct dialect match
    if normalized in dialect_map:
        lang = detect(normalized)
        if lang == "ar":
            return {"answer": dialect_map[normalized]["tn"], "sources": []}
        question = dialect_map[normalized]["en"]

    db = get_cached_db()
    raw = db.similarity_search_with_relevance_scores(question, k=3)

    seen, results = set(), []
    for doc, score in raw:
        txt = doc.page_content.strip()
        if txt and txt not in seen:
            seen.add(txt)
            results.append((doc, score))

    retrieved = [
        {
            "source": doc.metadata.get("source"),
            "score": round(score, 3),
            "excerpt": doc.page_content.replace("\n", " ")[:200]
        }
        for doc, score in results
    ]
    if debug:
        return {"retrieved": retrieved}

    if not results or results[0][1] < MIN_SCORE:
        return {
            "answer": "The provided documents do not contain sufficient information to answer this question.",
            "retrieved": retrieved
        }

    top3 = results[:3]
    context = "\n\n---\n\n".join(doc.page_content for doc, _ in top3)

    prompt = ChatPromptTemplate.from_template(
        """
أنت مساعد ذكي تتكلم باللهجة التونسية. هدفك هو مساعدة الناس على فهم منصة DWEXO بطريقة مبسطة.

السياق:
{context}

السؤال:
{question}
"""
    ).format(context=context, question=question)

    model = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama3-70b-8192")
    answer = model.predict(prompt)

    return {
        "answer": answer,
        "sources": [doc.metadata.get("source") for doc, _ in top3]
    }
