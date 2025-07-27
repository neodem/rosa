### vectorstore.py
from qdrant_client import QdrantClient
from qdrant_client.http.models import SearchParams
from src.config import COLLECTION_NAME

SCORE_THRESHOLD = 0.75  # adjust as needed, 0-1 range for cosine similarity

client = QdrantClient(host="localhost", port=6333)

def search_similar_chunks(query_embedding: list) -> list:
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=10,
        search_params=SearchParams(hnsw_ef=128)
    )
    # Filter results by score threshold
    filtered = [r for r in results if r.score and r.score >= SCORE_THRESHOLD]
    return filtered

