import os, hashlib
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Configuration constants
CHROMA_PATH = "chroma"
DATA_PATH   = "data/books"
EMBED_MODEL = "all-MiniLM-L6-v2"
MIN_SCORE   = 0.1

def get_embedding_db() -> Chroma:
    emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=emb)

def calculate_md5(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()
