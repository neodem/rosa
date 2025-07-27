# test_qdrant_indexing.py
import time
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, HnswConfigDiff, OptimizersConfigDiff
import uuid

# Configuration
COLLECTION_NAME = "test_collection_indexing"
EMBEDDING_DIM = 384
NUM_POINTS_TO_ADD = 1500

# Initialize client
client = QdrantClient(host="localhost", port=6333)

hnsw_config_diff = HnswConfigDiff(
    m=16,
    ef_construct=100,
    full_scan_threshold=10000,
    on_disk=False,
    max_indexing_threads=1 # Explicitly set to 1
)

optimizer_config_diff = OptimizersConfigDiff(
    deleted_threshold=0.2,
    vacuum_min_vector_number=1000,
    default_segment_number=0,
    flush_interval_sec=1,
    indexing_threshold=0,
    max_optimization_threads=1 # Explicitly set to 1
)

try:
    print(f"Recreating collection '{COLLECTION_NAME}'...")
    if client.collection_exists(collection_name=COLLECTION_NAME):
        client.delete_collection(collection_name=COLLECTION_NAME)
        print(f"Existing collection '{COLLECTION_NAME}' deleted.")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
    )
    print(f"Collection '{COLLECTION_NAME}' recreated.")

    print(f"Updating collection '{COLLECTION_NAME}' with HNSW and Optimizer configs...")
    client.update_collection(
        collection_name=COLLECTION_NAME,
        hnsw_config=hnsw_config_diff,
        optimizer_config=optimizer_config_diff
    )
    print(f"Collection '{COLLECTION_NAME}' updated.")

    print(f"Adding {NUM_POINTS_TO_ADD} dummy points...")
    points = []
    for i in range(NUM_POINTS_TO_ADD):
        point = {
            "id": str(uuid.uuid4()),
            "vector": [float(j % 100) / 100.0 for j in range(EMBEDDING_DIM)], # Dummy vector
            "payload": {"index": i}
        }
        points.append(point)

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True
    )
    print(f"Upserted {len(points)} points to collection '{COLLECTION_NAME}'.")

    print("Waiting 10 seconds for background optimizer...")
    time.sleep(10)

    collection_info = client.get_collection(collection_name=COLLECTION_NAME)
    print("\nCollection Info after upsert and wait:")
    print(f"Status: {collection_info.status.value}")
    print(f"Points Count: {collection_info.points_count}")
    print(f"Indexed Vectors Count: {collection_info.indexed_vectors_count}")
    print(f"Segments Count: {collection_info.segments_count}")
    print(f"Optimizer Config: {collection_info.config.optimizer_config.dict()}")
    print(f"HNSW Config: {collection_info.config.hnsw_config.dict()}")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Optional: Clean up the test collection
    # client.delete_collection(collection_name=COLLECTION_NAME)
    # print(f"Cleaned up test collection '{COLLECTION_NAME}'.")
    pass