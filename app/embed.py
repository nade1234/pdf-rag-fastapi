import os
from fastapi import APIRouter, UploadFile, File
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .utils import DATA_PATH, get_embedding_db, calculate_md5

router = APIRouter()

@router.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    os.makedirs(DATA_PATH, exist_ok=True)
    dest = os.path.join(DATA_PATH, file.filename)
    with open(dest, "wb") as f:
        f.write(await file.read())
    return {"message": f"{file.filename} saved to {DATA_PATH}"}

@router.post("/embed_new_pdfs/")
def embed_new_pdfs():
    db        = get_embedding_db()
    existing  = db.get(include=["metadatas"])["metadatas"]
    seen_hash = {m["file_hash"] for m in existing if m.get("file_hash")}

    new_chunks = []
    for fname in os.listdir(DATA_PATH):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(DATA_PATH, fname)
        h    = calculate_md5(path)
        if h in seen_hash:
            continue

        docs = PyPDFLoader(path).load()
        for d in docs:
            d.metadata["source"]    = fname
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
