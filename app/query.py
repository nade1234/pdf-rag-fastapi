import os
from dotenv import load_dotenv
from typing import Optional
from fastapi import APIRouter, Form
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from .utils import get_embedding_db, MIN_SCORE, send_notification_email

# Load .env so that GROQ_API_KEY is available in os.environ
load_dotenv()

router = APIRouter()

@router.post("/query/")
async def query_db(
    question: str = Form(...),
    debug: bool = Form(False),
):
    # 1. Handle greetings
    greetings_map = {
    "hi": "Hi! I'm the assistant of DWEXO. How can I help you?",
    "hello": "Hello! I'm the assistant of DWEXO. What can I help you with?",
    "hey": "Hey there! I'm the DWEXO assistant. Ready to assist you!",
    "how are you": "I'm the DWEXO assistant — always ready to help!",
    "good morning": "Good morning! I'm the DWEXO assistant. What can I do for you?",
    "good evening": "Good evening! I'm the DWEXO assistant. Need help?"
    }

    normalized = question.lower().strip()
    for g in greetings_map:
        if g in normalized:
            return {"answer": greetings_map[g], "sources": []}

    # 2. Regular RAG flow
    db = get_embedding_db()
    raw = db.similarity_search_with_relevance_scores(question, k=7)

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
        # Send notification email when question cannot be answered
        try:
            send_notification_email(question)
        except Exception as e:
            print(f"Email notification failed: {e}")
        
        return {
            "answer": "The provided documents do not contain sufficient information to answer this question.",
            "retrieved": retrieved
        }

    top3 = results[:3]
    context = "\n\n---\n\n".join(doc.page_content for doc, _ in top3)
    prompt = ChatPromptTemplate.from_template(
        """
You are a factual, no-nonsense AI assistant trained to answer questions about the DWEXO enterprise management platform.

Context:
{context}

Question:
{question}
"""
    ).format(context=context, question=question)

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in your environment")

    model = ChatGroq(groq_api_key=api_key, model_name="llama3-70b-8192")
    answer = model.predict(prompt)

    return {
        "answer": answer,
        "sources": [doc.metadata.get("source") for doc, _ in top3]
    }