### log_response.py
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from embed import embed_text
from src.config import LOG_COLLECTION_NAME
import uuid, datetime

client = QdrantClient(host="localhost", port=6333)

def log_answer(query: str, response: str):
    text = f"Q: {query}\nA: {response}"
    vector = embed_text(text)
    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=vector,
        payload={
            "query": query,
            "response": response,
            "timestamp": datetime.datetime.now().isoformat()
        }
    )
    client.upsert(collection_name=LOG_COLLECTION_NAME, points=[point])
