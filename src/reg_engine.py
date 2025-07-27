### rag_engine.py
from embed import embed_text
from vectorstore import search_similar_chunks
from ollama_client import call_ollama
from log_response import log_answer


def answer_question(query: str) -> str:
    query_embedding = embed_text(query)
    context_chunks = search_similar_chunks(query_embedding)
    context_text = "\n".join([chunk['payload']['text'] for chunk in context_chunks])

    prompt = f"""
    You are a helpful assistant. Use the following context to answer the question.

    Context:
    {context_text}

    Question: {query}
    """

    response = call_ollama(prompt.strip())
    log_answer(query, response)
    return response