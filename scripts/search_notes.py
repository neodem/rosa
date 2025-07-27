from sentence_transformers import SentenceTransformer
from src.vectorstore import client
from src.config import COLLECTION_NAME

model = SentenceTransformer("thenlper/gte-small")  # or your actual model

query = "how do I drive a car?"
query_vector = model.encode(query).tolist()

results = client.search(
    collection_name=COLLECTION_NAME,
    query_vector=query_vector,
    limit=5,
    with_payload=True,
)

for result in results:
    print(f"Score: {result.score}")
    print(f"Text: {result.payload.get('text')}")
    print("-----")
