import time
import torch
import faiss
import re
import collections
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from neo4j_config import create_neo4j_driver

print("==================================================")
print("   INITIATING LAYERED THESIS BENCHMARK (PHASE 2)")
print("==================================================\n")

def compute_f1(prediction, truth):
    pred_tokens = re.sub(r'[^\w\s]', '', prediction.lower()).split()
    truth_tokens = re.sub(r'[^\w\s]', '', truth.lower()).split()
    common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0: return 0.0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)

test_cases = [
    {
        "type": "Inference (2-Hop)",
        "query": "Where was the director of Interstellar born?",
        "ground_truth": "London",
        "neo4j_start_node": "Interstellar"
    },
    {
        "type": "Comparison (Temporal)",
        "query": "Which movie was released first: Interstellar or Inception?",
        "ground_truth": "Inception",
        "neo4j_start_node": "Interstellar"
    },
    {
        "type": "Adversarial (Null)",
        "query": "Who directed the movie The Matrix?",
        "ground_truth": "matrix",
        "neo4j_start_node": "The Matrix"
    }
]

corpus = [
    "Interstellar is a 2014 epic science fiction film directed by Christopher Nolan.",
    "Christopher Nolan was born in Westminster, London, United Kingdom.",
    "Inception is a 2010 science fiction action film written and directed by Christopher Nolan."
]

print("[*] Loading Models into RTX 4050 GPU...")
encoder = SentenceTransformer("all-MiniLM-L6-v2")

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    quantization_config=bnb_config,
    device_map={"": "cuda:0"}
)

embeddings = encoder.encode(corpus)
dimension = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(embeddings)

neo4j_driver = create_neo4j_driver()

def generate_answer(context, question):
    prompt = f"Use the following context to answer concisely. If the context does not contain the answer, say 'I don't know'.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=60)
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

print("\n==================================================")
print("   EXECUTING LAYERED EVALUATION")
print("==================================================\n")

for test in test_cases:
    print(f"QUERY TYPE : {test['type']}")
    print(f"QUESTION   : {test['query']}")
    print(f"TRUTH      : {test['ground_truth']}\n")

    # --- PIPELINE A (FAISS) ---
    q_vec = encoder.encode([test["query"]])
    _, indices = faiss_index.search(q_vec, k=1)
    faiss_context = corpus[indices[0][0]]
    faiss_ans = generate_answer(faiss_context, test["query"])

    faiss_retrieval_success = test["ground_truth"].lower() in faiss_context.lower()
    faiss_f1 = compute_f1(faiss_ans, test["ground_truth"])

    print("--- PIPELINE A (NAIVE FAISS) ---")
    print(f"Layer 2 (Retrieval) : {'PASSED' if faiss_retrieval_success else 'FAILED'}")
    print(f"Layer 3 (F1-Score)  : {faiss_f1:.2f}")
    print(f"LLM Final Output    : {faiss_ans}\n")

    # --- PIPELINE B (NEO4J) ---
    cypher_query = f"""
    MATCH path = (s {{name: '{test['neo4j_start_node']}'}})-[*1..2]-(t)
    UNWIND relationships(path) AS r
    WITH DISTINCT r
    RETURN startNode(r).name + coalesce(' (Year ' + toString(startNode(r).year) + ')', '') + ' ' + type(r) + ' ' + endNode(r).name + coalesce(' (Year ' + toString(endNode(r).year) + ')', '') AS fact
    """
    with neo4j_driver.session(database="neo4j") as session:
        res = session.run(cypher_query).data()
        if res:
            neo4j_context = ". ".join([row["fact"] for row in res])
        else:
            neo4j_context = ""

    neo4j_ans = generate_answer(neo4j_context, test["query"])

    neo4j_retrieval_success = test["ground_truth"].lower() in neo4j_context.lower()
    neo4j_f1 = compute_f1(neo4j_ans, test["ground_truth"])

    print("--- PIPELINE B (NEO4J GRAPHRAG) ---")
    print(f"Layer 2 (Retrieval) : {'PASSED' if neo4j_retrieval_success else 'FAILED'}")
    print(f"Layer 3 (F1-Score)  : {neo4j_f1:.2f}")
    print(f"LLM Final Output    : {neo4j_ans}\n")
    print("-" * 50 + "\n")

print("==================================================")
print("   EVALUATION COMPLETE")
print("==================================================")