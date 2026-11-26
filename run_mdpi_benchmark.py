import pandas as pd
import torch
import faiss
import re
import collections
import random
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from neo4j_config import create_neo4j_driver

print("=======================================================")
print("   INITIATING MDPI BENCHMARK (STEP 2: FIX + MRR@K)")
print("=======================================================\n")

# --- 1. LOAD DATASET ---
dataset = load_dataset("framolfese/2WikiMultihopQA", split="validation")
sample_size = 50
TOP_K = 2  # Fixed baseline Top-K for local 6GB VRAM edge constraint
print(f"[*] Sampling {sample_size} questions for local RTX 4050 benchmark (Top-K={TOP_K})...")
random.seed(42)
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
def compute_exact_match(prediction, truth):
    pred_clean = re.sub(r'[^\w\s]', '', prediction.strip().lower())
    truth_clean = re.sub(r'[^\w\s]', '', str(truth).strip().lower())
    return 1 if pred_clean == truth_clean else 0

def compute_f1(prediction, truth):
    pred_tokens = re.sub(r'[^\w\s]', '', prediction.lower()).split()
    truth_tokens = re.sub(r'[^\w\s]', '', str(truth).lower()).split()
    common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)

def compute_mrr(retrieved_chunks, truth):
    truth_clean = str(truth).strip().lower()
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        if truth_clean in str(chunk).lower():
            return 1.0 / rank
    return 0.0

def generate_answer(context, question):
    prompt = (
        "Use the following context to answer concisely. "
        "If the context does not contain the answer, say 'I don't know'.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=30)
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

# --- 4. CONNECT NEO4J ---
neo4j_driver = create_neo4j_driver()

results_data = []

print("\n=======================================================")
print("   EXECUTING PIPELINES")
print("=======================================================\n")

for idx, item in enumerate(test_batch):
    print(f"Processing Question {idx+1}/{sample_size}...")
    query = item['question']
    truth = item['answer']

    # --- FIX: Properly extract real Wikipedia titles and sentences ---
    titles = item['context']['title']
    sentences_list = item['context']['sentences']
    context_paragraphs = [
        f"{title}: {' '.join(sents)}"
        for title, sents in zip(titles, sentences_list)
    ]

    # --- PIPELINE A (FAISS) ---
    embeddings = encoder.encode(context_paragraphs)
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)

    q_vec = encoder.encode([query])
    k_actual = min(TOP_K, len(context_paragraphs))
    _, indices = faiss_index.search(q_vec, k=k_actual)

    faiss_retrieved_chunks = [context_paragraphs[i] for i in indices[0]]
    faiss_context = " ".join(faiss_retrieved_chunks)

    faiss_ans = generate_answer(faiss_context, query)
    faiss_f1 = compute_f1(faiss_ans, truth)
    faiss_em = compute_exact_match(faiss_ans, truth)
    faiss_mrr = compute_mrr(faiss_retrieved_chunks, truth)
    faiss_retrieval_success = str(truth).lower() in faiss_context.lower()

    # --- PIPELINE B (NEO4J) ---
    with neo4j_driver.session(database="neo4j") as session:
        session.run("MATCH (n) DETACH DELETE n")
        for triple in item['evidences']:
            subj, rel, obj = triple
            rel_clean = re.sub(r'[^A-Za-z0-9_]', '_', str(rel)).upper()
            cypher_inject = (
                f"MERGE (s:Entity {{name: $subj}}) "
                f"MERGE (o:Entity {{name: $obj}}) "
                f"MERGE (s)-[:`{rel_clean}`]->(o)"
            )
            try:
                session.run(cypher_inject, subj=str(subj), obj=str(obj))
            except Exception:
                continue

        res = session.run(
            "MATCH (s)-[r]->(o) RETURN s.name + ' ' + type(r) + ' ' + o.name AS fact"
        ).data()
        neo4j_facts = [r["fact"] for r in res] if res else []
        neo4j_context = ". ".join(neo4j_facts)

    neo4j_ans = generate_answer(neo4j_context, query)
    neo4j_f1 = compute_f1(neo4j_ans, truth)
    neo4j_em = compute_exact_match(neo4j_ans, truth)
    neo4j_mrr = compute_mrr(neo4j_facts, truth)
    neo4j_retrieval_success = str(truth).lower() in neo4j_context.lower()

    results_data.append({
        "Question": query,
        "Ground Truth": truth,
        "FAISS Context": faiss_context,
        "Neo4j Context": neo4j_context,
        "FAISS Answer": faiss_ans,
        "Neo4j Answer": neo4j_ans,
        "FAISS Recall@K": int(faiss_retrieval_success),
        "FAISS MRR@K": faiss_mrr,
        "FAISS EM": faiss_em,
        "FAISS F1": faiss_f1,
        "Neo4j Recall@K": int(neo4j_retrieval_success),
        "Neo4j MRR@K": neo4j_mrr,
        "Neo4j EM": neo4j_em,
        "Neo4j F1": neo4j_f1
    })

neo4j_driver.close()

# --- 5. EXPORT METRICS ---
df = pd.DataFrame(results_data)
df.to_csv("mdpi_benchmark_results.csv", index=False)

print("\n=======================================================")
print("   BENCHMARK COMPLETE - RESULTS SAVED TO CSV")
print("=======================================================")

print(f"\nFinal Averages over {sample_size} Real Wikipedia Queries (Top-K={TOP_K}):")
print(f"FAISS Recall@K          : {df['FAISS Recall@K'].mean() * 100:.1f}%")
print(f"FAISS MRR@K             : {df['FAISS MRR@K'].mean():.4f}")
print(f"FAISS Exact Match (EM)  : {df['FAISS EM'].mean() * 100:.1f}%")
print(f"FAISS Average F1        : {df['FAISS F1'].mean():.4f}")
print("-" * 50)
print(f"Neo4j Recall@K          : {df['Neo4j Recall@K'].mean() * 100:.1f}%")
print(f"Neo4j MRR@K             : {df['Neo4j MRR@K'].mean():.4f}")
print(f"Neo4j Exact Match (EM)  : {df['Neo4j EM'].mean() * 100:.1f}%")
print(f"Neo4j Average F1        : {df['Neo4j F1'].mean():.4f}")