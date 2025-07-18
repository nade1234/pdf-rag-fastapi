import os
from fastapi import FastAPI
from .embed import router as embed_router
from .query import router as query_router
from .utils import get_embedding_db

app = FastAPI()

# Mount the routers
app.include_router(embed_router)
app.include_router(query_router)

@app.get("/list_indexed/")
def list_indexed():
    db    = get_embedding_db()
    metas = db.get(include=["metadatas"])["metadatas"]
    files = sorted({m.get("source") for m in metas if m.get("source")})
    return {"indexed_files": files}
