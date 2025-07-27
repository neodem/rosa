### rag_engine.py
from embed import embed_text
from src.config import DEBUG
from vectorstore import search_similar_chunks
from ollama_client import call_ollama
from log_response import log_answer


def answer_question(query: str) -> str:
    query_embedding = embed_text(query)
    context_chunks = search_similar_chunks(query_embedding)

    if DEBUG:
        print(f"[DEBUG] Retrieved {len(context_chunks)} context chunks")
        for chunk in context_chunks:
            print(f"[DEBUG] Score: {chunk.score:.3f} Payload: {chunk.payload}")

    context_text = "\n".join([chunk.payload['response'] for chunk in context_chunks])

    prompt = f"""
    You are a helpful assistant. Use the following context to answer the question.

    Context:
    {context_text}

    Question: {query}
    """

    stripped_prompt = prompt.strip()

    if DEBUG:
        print(f"[DEBUG] LLM Prompt:\n{stripped_prompt}\n")

    response = call_ollama(stripped_prompt)

    log_answer(query, response)
    return response