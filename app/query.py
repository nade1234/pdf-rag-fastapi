import os
from dotenv import load_dotenv
from typing import Optional
from fastapi import APIRouter, Form
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from .utils import get_embedding_db, MIN_SCORE

# Load .env so that GROQ_API_KEY is available in os.environ
load_dotenv()

router = APIRouter()

@router.post("/query/")
async def query_db(
    question: str  = Form(...),
    debug:    bool = Form(False),
):
    db  = get_embedding_db()
    raw = db.similarity_search_with_relevance_scores(question, k=7)

    # Deduplicate by text
    seen, results = set(), []
    for doc, score in raw:
        txt = doc.page_content.strip()
        if txt and txt not in seen:
            seen.add(txt)
            results.append((doc, score))

    # Prepare debug response
    retrieved = [
        {
            "source":  doc.metadata.get("source"),
            "score":   round(score, 3),
            "excerpt": doc.page_content.replace("\n", " ")[:200]
        }
        for doc, score in results
    ]
    if debug:
        return {"retrieved": retrieved}

    # If below threshold
    if not results or results[0][1] < MIN_SCORE:
        return {
            "answer":    "The provided documents do not contain sufficient information to answer this question.",
            "retrieved": retrieved
        }

    # Build prompt from top‑3 chunks
    top3    = results[:3]
    context = "\n\n---\n\n".join(doc.page_content for doc, _ in top3)
    prompt  = ChatPromptTemplate.from_template(
        """
You are a factual, no-nonsense AI assistant trained to answer questions about the DWEXO enterprise management platform.

Context:
{context}

Question:
{question}
"""
    ).format(context=context, question=question)

    # Fetch API key and call the LLM
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in your environment")
    model  = ChatGroq(groq_api_key=api_key, model_name="llama3-70b-8192")
    answer = model.predict(prompt)

    return {
        "answer":  answer,
        "sources": [doc.metadata.get("source") for doc, _ in top3]
    }
