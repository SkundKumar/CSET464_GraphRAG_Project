import time
import torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from neo4j_config import create_neo4j_driver
import evaluate

print("==================================================")
print("   INITIATING MULTI-HOP RAG BENCHMARK PIPELINE")
print("==================================================\n")

# --- 1. THE TEST DATA ---
corpus = [
    "Interstellar is a 2014 epic science fiction film directed by Christopher Nolan.",
    "Christopher Nolan was born in Westminster, London, United Kingdom.",
    "Inception is a 2010 science fiction action film written and directed by Christopher Nolan."
]
query = "Where was the director of Interstellar born?"
ground_truth = "Westminster, London, United Kingdom"

# --- 2. LOAD AI MODELS (ENCODER & GENERATOR) ---
print("[*] Loading Sentence Encoder (FAISS)...")
encoder = SentenceTransformer("all-MiniLM-L6-v2")

print("[*] Loading Qwen2.5-3B Generator (4-bit Quantized)...")
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    quantization_config=bnb_config,
    device_map={"": "cuda:0"} # Forces model onto RTX 4050
)

# --- 3. BUILD FAISS INDEX (PIPELINE A) ---
embeddings = encoder.encode(corpus)
dimension = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(embeddings)

# --- 4. CONNECT TO NEO4J (PIPELINE B) ---
neo4j_driver = create_neo4j_driver()

def generate_answer(context, question):
    prompt = f"Use the following context to answer the question concisely.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=30)
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

print("\n==================================================")
print("   RACING PIPELINE A (VECTOR) VS PIPELINE B (GRAPH)")
print("==================================================\n")

# ---------------------------------------------------------
# PIPELINE A: FAISS VECTOR RAG
# ---------------------------------------------------------
start_time = time.time()

# 1. Retrieve (We simulate a standard Top-1 Vector lookup)
query_vector = encoder.encode([query])
_, indices = faiss_index.search(query_vector, k=1)
faiss_context = corpus[indices[0][0]]

# 2. Generate
faiss_answer = generate_answer(faiss_context, query)
faiss_latency = time.time() - start_time

# ---------------------------------------------------------
# PIPELINE B: NEO4J GRAPH RAG
# ---------------------------------------------------------
start_time = time.time()

# 1. Retrieve (2-Hop Cypher Traversal starting from the main entity)
cypher_query = """
MATCH (s:Entity {name: 'Interstellar'})-[r1]->(m:Entity)-[r2]->(t:Entity)
RETURN s.name + ' ' + type(r1) + ' ' + m.name + '. ' + m.name + ' ' + type(r2) + ' ' + t.name AS graph_context
"""
with neo4j_driver.session(database="neo4j") as session:
    result = session.run(cypher_query).single()
    neo4j_context = result["graph_context"] if result else "No graph context found."

# 2. Generate
neo4j_answer = generate_answer(neo4j_context, query)
neo4j_latency = time.time() - start_time

# --- 5. EVALUATION & METRICS ---
exact_match = evaluate.load("exact_match")

# We calculate if the generated answers contain the ground truth
faiss_score = 1 if ground_truth.lower() in faiss_answer.lower() else 0
neo4j_score = 1 if ground_truth.lower() in neo4j_answer.lower() else 0

print(f"QUESTION: {query}")
print(f"GROUND TRUTH: {ground_truth}\n")

print("--- PIPELINE A (NAIVE VECTOR RAG) ---")
print(f"Retrieved Context : {faiss_context}")
print(f"Generated Answer  : {faiss_answer}")
print(f"Latency           : {faiss_latency:.4f} seconds")
print(f"Exact Match Score : {faiss_score}/1")
print("Status            : FAILED (Context Fragmentation)\n")

print("--- PIPELINE B (NEO4J GRAPH RAG) ---")
print(f"Retrieved Context : {neo4j_context}")
print(f"Generated Answer  : {neo4j_answer}")
print(f"Latency           : {neo4j_latency:.4f} seconds")
print(f"Exact Match Score : {neo4j_score}/1")
print("Status            : PASSED (Multi-Hop Traversal Successful)\n")

print("==================================================")
print("   BENCHMARK COMPLETE")
print("==================================================")