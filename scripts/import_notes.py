import os
import uuid
import time
from datetime import datetime
from src.vectorstore import client
from src.embed import embed_text
from src.config import COLLECTION_NAME

# Chunk size in number of words (adjust as needed)
CHUNK_SIZE = 300

def chunk_text(text, chunk_size=CHUNK_SIZE):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i+chunk_size])

def create_points_from_file(filepath):
    points = []
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    for idx, chunk in enumerate(chunk_text(text)):
        embedding = embed_text(chunk)

        if not embedding or all(v == 0 for v in embedding):
            print(f"⚠️  Empty or zero vector for chunk index {idx} in {filepath}")
        else:
            print(f"✅ Vector length: {len(embedding)}, sample: {embedding[:5]}")

        if len(embedding) != 384:
            raise Exception("Embedding size error")

        point = {
            "id": str(uuid.uuid4()),
            "vector": embedding,
            "payload": {
                "text": chunk,
                "source_file": os.path.basename(filepath),
                "chunk_index": idx,
                "imported_at": datetime.utcnow().isoformat()
            }
        }
        points.append(point)
    return points

NOTES_FOLDER = r"F:\Dropbox\_home\docs\notes"
all_points = []
for root, _, files in os.walk(NOTES_FOLDER):
    for file in files:
        if file.endswith((".md", ".txt")):
            path = os.path.join(root, file)
            print(f"Processing {path}...")
            all_points.extend(create_points_from_file(path))

# Assuming this is in your import script or a new temporary test script

# ... (your existing import script code, after client.upsert) ...

if all_points:
    client.upsert(collection_name=COLLECTION_NAME, points=all_points, wait=True)
    print(f"Upserted {len(all_points)} points to collection '{COLLECTION_NAME}'")

    # NEW: Try to explicitly trigger optimization (experimental)
    # This might require a specific internal API call, which is not usually exposed
    # and might vary by Qdrant version.
    # THIS IS NOT GUARANTEED TO WORK AND MIGHT NOT BE THE CORRECT WAY IN 1.15.1
    # But worth a shot for diagnosis.

    # Option 1: Using the client's internal call (less likely exposed directly)
    # This is a guess - the client might not expose such a method publicly
    try:
        # Check if the client has an internal way to trigger it.
        # This is highly speculative and likely incorrect for a public API client.
        # This part of the code is merely for diagnostic purposes to see if we can
        # force *any* optimizer activity.
        print("Attempting to trigger optimizer via internal client method (if available)...")
        # client._client.collections_api.collection_update_operations_post_sync(
        #    collection_name=COLLECTION_NAME,
        #    timeout=60,
        #    update_collection=qdrant_client.http.models.UpdateCollection(
        #        optimizer_config=OptimizersConfigDiff(
        #            indexing_threshold=0 # Re-asserting the config, not really triggering an 'optimize' method
        #        )
        #    )
        # )
        # A direct "optimize now" method is unlikely in the public client API.
        # Instead, we just wait and check logs for automatic optimization.

        print("Waiting 10 seconds for background optimizer...")
        time.sleep(10) # Give Qdrant time to run its background optimizer

        collection_info_after_wait = client.get_collection(collection_name=COLLECTION_NAME)
        print(f"Indexed vectors after wait: {collection_info_after_wait.indexed_vectors_count}")

    except Exception as e:
        print(f"Failed to trigger optimizer via client method or check: {e}")

else:
    print("No notes found to import.")


# if all_points:
#     client.upsert(collection_name=COLLECTION_NAME, points=all_points, wait=True)
#     print(f"Upserted {len(all_points)} points to collection '{COLLECTION_NAME}'")
# else:
#     print("No notes found to import.")