# test_qdrant_query.py
from src.vectorstore import client
from src.embed import embed_text  # your embed function
from src.config import COLLECTION_NAME

def query_loop():
    print("Type your query and press Enter. Type 'exit' to quit.")

    while True:
        text = input("You: ").strip()
        if text.lower() in ("exit", "quit"):
            print("Bye!")
            break

        vector = embed_text(text)
        print(f"[DEBUG] Embedding length: {len(vector)}")

        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector,
            limit=5,
        )

        if not results:
            print("No results found.")
        else:
            print(f"Found {len(results)} results:")
            for hit in results:
                print(f"- ID: {hit.id}")
                print(f"  Score: {hit.score:.4f}")
                print(f"  Payload: {hit.payload}")


if __name__ == "__main__":
    query_loop()