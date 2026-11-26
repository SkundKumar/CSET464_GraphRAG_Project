import pandas as pd
import time
import torch
import faiss
import re
import collections
import random
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from neo4j_config import create_neo4j_driver

print("==================================================")
print("   INITIATING REAL WIKIPEDIA BENCHMARK BATCH")
print("==================================================\n")

# --- 1. LOAD DATASET ---
dataset = load_dataset("framolfese/2WikiMultihopQA", split="validation")
sample_size = 50
print(f"[*] Sampling {sample_size} questions for local RTX 4050 benchmark...")
random.seed(42) # For reproducible results
sampled_indices = random.sample(range(len(dataset)), sample_size)
test_batch = [dataset[i] for i in sampled_indices]

# --- 2. LOAD MODELS ---
print("[*] Loading Sentence Encoder (FAISS)...")
encoder = SentenceTransformer("all-MiniLM-L6-v2")

print("[*] Loading Qwen2.5-3B Generator (4-bit)...")
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    quantization_config=bnb_config,
    device_map={"": "cuda:0"}
)

# --- 3. HELPER FUNCTIONS ---
def compute_f1(prediction, truth):
    pred_tokens = re.sub(r'[^\w\s]', '', prediction.lower()).split()
    truth_tokens = re.sub(r'[^\w\s]', '', str(truth).lower()).split()
    common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0: return 0.0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)

def generate_answer(context, question):
    prompt = f"Use the following context to answer concisely. If the context does not contain the answer, say 'I don't know'.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=30)
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

# Connect Neo4j
neo4j_driver = create_neo4j_driver()

results_data = []

print("\n==================================================")
print("   EXECUTING PIPELINES")
print("==================================================\n")

for idx, item in enumerate(test_batch):
    print(f"Processing Question {idx+1}/{sample_size}...")
    query = item['question']
    truth = item['answer']

    # Extract flat text for FAISS
    context_paragraphs = [" ".join(p[1]) for p in item['context']]

    # --- PIPELINE A (FAISS) ---
    embeddings = encoder.encode(context_paragraphs)
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)

    q_vec = encoder.encode([query])
    _, indices = faiss_index.search(q_vec, k=2) # Get top 2 chunks
    faiss_context = " ".join([context_paragraphs[i] for i in indices[0]])

    faiss_ans = generate_answer(faiss_context, query)
    faiss_f1 = compute_f1(faiss_ans, truth)
    faiss_retrieval_success = str(truth).lower() in faiss_context.lower()

    # --- PIPELINE B (NEO4J) ---
    # Inject real triples into graph
    with neo4j_driver.session(database="neo4j") as session:
        session.run("MATCH (n) DETACH DELETE n") # Wipe previous question
        for triple in item['evidences']:
            subj, rel, obj = triple
            # Clean strings for Cypher
            subj = subj.replace("'", "\\'")
            rel = rel.replace(" ", "_").upper()
            obj = str(obj).replace("'", "\\'")
            cypher_inject = f"MERGE (s:Entity {{name: '{subj}'}}) MERGE (o:Entity {{name: '{obj}'}}) MERGE (s)-[:{rel}]->(o)"
            try:
                session.run(cypher_inject)
            except Exception:
                continue # Skip malformed strings

        # Naive traversal for all connected facts
        res = session.run("MATCH (s)-[r]->(o) RETURN s.name + ' ' + type(r) + ' ' + o.name AS fact").data()
        neo4j_context = ". ".join([r["fact"] for r in res]) if res else ""

    neo4j_ans = generate_answer(neo4j_context, query)
    neo4j_f1 = compute_f1(neo4j_ans, truth)
    neo4j_retrieval_success = str(truth).lower() in neo4j_context.lower()

    # Store results
    results_data.append({
        "Question": query,
        "Ground Truth": truth,
        "FAISS Retrieval": faiss_retrieval_success,
        "FAISS F1": faiss_f1,
        "Neo4j Retrieval": neo4j_retrieval_success,
        "Neo4j F1": neo4j_f1
    })

# --- 4. EXPORT METRICS ---
df = pd.DataFrame(results_data)
df.to_csv("benchmark_results.csv", index=False)

print("==================================================")
print("   BENCHMARK COMPLETE - RESULTS SAVED TO CSV")
print("==================================================")

# Calculate final averages
print(f"\nFinal Averages over {sample_size} Real Wikipedia Queries:")
print(f"FAISS Retrieval Success : {(df['FAISS Retrieval'].sum() / sample_size) * 100}%")
print(f"FAISS Average F1        : {df['FAISS F1'].mean():.4f}")
print(f"Neo4j Retrieval Success : {(df['Neo4j Retrieval'].sum() / sample_size) * 100}%")
print(f"Neo4j Average F1        : {df['Neo4j F1'].mean():.4f}")