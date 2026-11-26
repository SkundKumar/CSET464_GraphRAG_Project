import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

print("--- 1. INITIALIZING PIPELINE A (NAIVE VECTOR RAG) ---")
# Using a lightweight embedding model to convert sentences into numbers
encoder = SentenceTransformer("all-MiniLM-L6-v2")

# The exact same multi-hop sentences we put into Neo4j
corpus = [
    "Interstellar is a 2014 epic science fiction film directed by Christopher Nolan.",
    "Christopher Nolan was born in Westminster, London, United Kingdom.",
    "Inception is a 2010 science fiction action film written and directed by Christopher Nolan."
]

print("--- 2. ENCODING TEXT INTO DENSE VECTORS ---")
# Convert text to dense numerical vectors
embeddings = encoder.encode(corpus)
dimension = embeddings.shape[1]

print(f"Generated {len(embeddings)} vectors of dimension {dimension}")

print("--- 3. BUILDING FAISS VECTOR INDEX ---")
# Create the flat L2 distance index
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
print(f"Total vectors indexed in FAISS: {index.ntotal}")

print("\n--- 4. RUNNING A VECTOR RETRIEVAL TEST ---")
# Ask a multi-hop question
query = "Where was the director of Interstellar born?"
query_vector = encoder.encode([query])

# Retrieve the Top-2 closest chunks based on spatial similarity
distances, indices = index.search(query_vector, k=2)

print(f"Query: '{query}'")
print("\nRetrieved Vector Chunks:")
for i in range(2):
    idx = indices[0][i]
    print(f"Rank {i+1} (Distance: {distances[0][i]:.4f}) -> {corpus[idx]}")

print("\nSUCCESS! FAISS Vector Database built and tested.")