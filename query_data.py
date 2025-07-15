import argparse
import os
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

# Load environment variables (for GROQ_API_KEY)
load_dotenv()

CHROMA_PATH = "chroma"

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Use HuggingFace Embeddings (FREE)
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_relevance_scores(query_text, k=3)

    # Debug print: always print scores and chunk samples
    print("Top retrieved chunks and scores:")
    for doc, score in results:
        print(f"Score: {score:.2f}")
        print(doc.page_content[:500])
        print("---")

    if len(results) == 0 or results[0][1] < 0.3:
        print("Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is not set in the environment or .env file!")

    model = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama3-70b-8192",
    )
    response_text = model.predict(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)

if __name__ == "__main__":
    main()
