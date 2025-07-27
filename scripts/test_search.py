from src.vectorstore import client
from src.config import COLLECTION_NAME

points = client.scroll(
    collection_name=COLLECTION_NAME,
    limit=5,
    with_vectors=False,
    with_payload=True,
)

for point in points[0]:
    print(f"ID: {point.id}")
    print(f"Payload: {point.payload}")
    print("-----")