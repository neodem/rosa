# scripts/init_vectorstore.py

import time # Added for the sleep
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    HnswConfig,
    OptimizersConfig,
    HnswConfigDiff,
    OptimizersConfigDiff
)
from src.vectorstore import client # Assuming this client is correctly initialized
from src.config import COLLECTION_NAME, LOG_COLLECTION_NAME

EMBEDDING_DIM = 384

# Full config objects (for clarity and for creating diffs)
full_hnsw_config = HnswConfig(
    m=16,
    ef_construct=100,
    full_scan_threshold=10000,
    on_disk=False,
    max_indexing_threads=0 # Let Qdrant auto-detect (num_cpus - 1)
)

full_optimizer_config = OptimizersConfig(
    deleted_threshold=0.2,
    vacuum_min_vector_number=1000,
    default_segment_number=0, # Let Qdrant auto-detect
    flush_interval_sec=1,     # *** Adjusted: Flush more frequently for small datasets ***
    indexing_threshold=0,     # The crucial setting: force immediate indexing
    max_optimization_threads=0 # Let Qdrant auto-detect (num_cpus - 1)
)

# Create Diff versions for updating
hnsw_config_diff = HnswConfigDiff(
    m=full_hnsw_config.m,
    ef_construct=full_hnsw_config.ef_construct,
    full_scan_threshold=full_hnsw_config.full_scan_threshold,
    on_disk=full_hnsw_config.on_disk,
    max_indexing_threads=full_hnsw_config.max_indexing_threads
)

optimizer_config_diff = OptimizersConfigDiff(
    deleted_threshold=full_optimizer_config.deleted_threshold,
    vacuum_min_vector_number=full_optimizer_config.vacuum_min_vector_number,
    default_segment_number=full_optimizer_config.default_segment_number,
    flush_interval_sec=full_optimizer_config.flush_interval_sec,
    indexing_threshold=full_optimizer_config.indexing_threshold,
    max_optimization_threads=full_optimizer_config.max_optimization_threads
)


# --- Collection 1: COLLECTION_NAME (notes) ---
print(f"Initializing collection '{COLLECTION_NAME}'...")

# Use the recommended pattern: delete if exists, then create
if client.collection_exists(collection_name=COLLECTION_NAME):
    client.delete_collection(collection_name=COLLECTION_NAME)
    print(f"Existing collection '{COLLECTION_NAME}' deleted.")

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
)
print(f"Collection '{COLLECTION_NAME}' created.")

# Update the collection's HNSW and Optimizer configurations
client.update_collection(
    collection_name=COLLECTION_NAME,
    hnsw_config=hnsw_config_diff,
    optimizer_config=optimizer_config_diff
)
print(f"✅ Collection '{COLLECTION_NAME}' updated with HNSW and Optimizer configs.")


# --- Collection 2: LOG_COLLECTION_NAME (conversation_log) ---
print(f"Initializing collection '{LOG_COLLECTION_NAME}'...")

if client.collection_exists(collection_name=LOG_COLLECTION_NAME):
    client.delete_collection(collection_name=LOG_COLLECTION_NAME)
    print(f"Existing collection '{LOG_COLLECTION_NAME}' deleted.")

client.create_collection(
    collection_name=LOG_COLLECTION_NAME,
    vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
)
print(f"Collection '{LOG_COLLECTION_NAME}' created.")

# Update the collection's HNSW and Optimizer configurations
client.update_collection(
    collection_name=LOG_COLLECTION_NAME,
    hnsw_config=hnsw_config_diff,
    optimizer_config=optimizer_config_diff
)
print(f"✅ Collection '{LOG_COLLECTION_NAME}' updated with HNSW and Optimizer configs.")

print("✅ All Qdrant collections initialized and configured.")

# Add a small delay to allow Qdrant to process initial configs
# Not strictly necessary but doesn't hurt.
time.sleep(2)