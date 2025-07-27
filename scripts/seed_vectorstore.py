# scripts/seed_vectorstore.py

import uuid
from datetime import datetime
from src.vectorstore import client
from src.config import COLLECTION_NAME
from src.embed import embed_text

test_data = [
    ("What is the capital of France?", "Paris is the capital of France."),
    ("Who wrote Hamlet?", "William Shakespeare wrote Hamlet."),
    ("What is the speed of light?", "Approximately 299,792 kilometers per second."),
]

def make_point(query, response):
    return {
        "id": str(uuid.uuid4()),
        "vector": embed_text(query),
        "payload": {
            "query": query,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
    }

points = [make_point(q, a) for q, a in test_data]

client.upsert(collection_name=COLLECTION_NAME, points=points)

print(f"Seeded {len(points)} points to collection '{COLLECTION_NAME}'")
