import argparse
import os
import json
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

# Load environment variables (for GROQ_API_KEY)
load_dotenv()

CHROMA_PATH = "chroma"
EMBED_MODEL = "all-MiniLM-L6-v2"
PROMPT_TEMPLATE = """
You are a factual, no-nonsense AI assistant trained to answer questions about the DWEXO enterprise management platform.

- Respond only with information present in the vector database or verifiable expert knowledge.
- If the question cannot be fully answered based on the available information, explicitly state: "The provided documents do not contain sufficient information to answer this question."
- When possible, specify if your answer refers to a product module, service, or support level.
- Be concise.

Context:
{context}

Question:
{question}
"""

def main():
    parser = argparse.ArgumentParser(description="Query the DWEXO RAG system")
    parser.add_argument("query_text", type=str, help="The question to ask the RAG system.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="If set, prints raw retrieved chunks and scores instead of the final answer."
    )
    args = parser.parse_args()

    # 1️⃣ Initialize Chroma
    embedding_function = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # 2️⃣ Retrieve top-k candidates (k=7 for better coverage)
    raw_results = db.similarity_search_with_relevance_scores(args.query_text, k=7)

    # 3️⃣ Deduplicate identical chunks
    seen_texts = set()
    results = []
    for doc, score in raw_results:
        text = doc.page_content.strip()
        if text and text not in seen_texts:
            seen_texts.add(text)
            results.append((doc, score))

    # 4️⃣ Build debug payload
    retrieved = [
        {
            "source": doc.metadata.get("source"),
            "score": round(score, 3),
            "excerpt": doc.page_content.replace("\n", " ")[:200]
        }
        for doc, score in results
    ]
    if args.debug:
        print(json.dumps({"retrieved": retrieved}, indent=2, ensure_ascii=False))
        return

    # 5️⃣ Apply a minimum score threshold
    MIN_SCORE = 0.1
    if not results or results[0][1] < MIN_SCORE:
        print("Unable to find matching results above threshold.")
        print(json.dumps({"retrieved": retrieved}, indent=2, ensure_ascii=False))
        return

    # 6️⃣ Build the context from the top-3 results
    top3 = results[:3]
    context = "\n\n---\n\n".join(doc.page_content for doc, _ in top3)

    # 7️⃣ Format the prompt and call the LLM
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context, question=args.query_text
    )

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in the environment.")

    model = ChatGroq(groq_api_key=api_key, model_name="llama3-70b-8192")
    answer = model.predict(prompt)

    # 8️⃣ Print the final answer and the exact contexts used
    print("\nAnswer:\n", answer, "\n")
    print("Contexts:")
    print(json.dumps(retrieved[:3], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
