### ollama_client.py
import requests
from src.config import OLLAMA_MODEL

def call_ollama(prompt: str) -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    )
    return response.json().get("response", "[no response]")