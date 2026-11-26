import pandas as pd
import torch
from bert_score import BERTScorer

print("=======================================================")
print("   STEP 4: COMPUTING BERTSCORE (SEMANTIC SIMILARITY)")
print("=======================================================\n")

# --- 1. LOAD SAVED BENCHMARK DATA ---
csv_path = "mdpi_benchmark_results.csv"
df = pd.read_csv(csv_path)

# Clean any missing/NaN values into safe strings
truths = [str(x).strip() if pd.notna(x) else "" for x in df["Ground Truth"]]
faiss_preds = [str(x).strip() if pd.notna(x) else "" for x in df["FAISS Answer"]]
neo4j_preds = [str(x).strip() if pd.notna(x) else "" for x in df["Neo4j Answer"]]

# --- 2. LOAD BERTSCORE MODEL ONCE (LOCAL GPU/CPU) ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[*] Loading BERTScore evaluator (bert-base-uncased) on {device.upper()}...")
scorer = BERTScorer(
    model_type="bert-base-uncased",
    lang="en",
    device=device,
    batch_size=16
)

# --- 3. SCORE FAISS ANSWERS ---
print("[*] Computing BERTScore for Pipeline A (FAISS)...")
f_P, f_R, f_F1 = scorer.score(faiss_preds, truths)

# --- 4. SCORE NEO4J ANSWERS ---
print("[*] Computing BERTScore for Pipeline B (Neo4j)...")
n_P, n_R, n_F1 = scorer.score(neo4j_preds, truths)

# --- 5. SAVE SCORES INTO MASTER CSV ---
df["FAISS BERTScore P"] = f_P.tolist()
df["FAISS BERTScore R"] = f_R.tolist()
df["FAISS BERTScore F1"] = f_F1.tolist()

df["Neo4j BERTScore P"] = n_P.tolist()
df["Neo4j BERTScore R"] = n_R.tolist()
df["Neo4j BERTScore F1"] = n_F1.tolist()

df.to_csv(csv_path, index=False)

print(f"\n[*] Updated '{csv_path}' with BERTScore P/R/F1 columns.")
print(f"[*] Reproducibility Hash (for IEEE paper): {scorer.hash}\n")

print("=======================================================")
print("   BERTSCORE (SEMANTIC SIMILARITY) RESULTS")
print("=======================================================")
print(f"FAISS BERTScore Precision : {df['FAISS BERTScore P'].mean():.4f}")
print(f"FAISS BERTScore Recall    : {df['FAISS BERTScore R'].mean():.4f}")
print(f"FAISS BERTScore F1        : {df['FAISS BERTScore F1'].mean():.4f}")
print("-" * 55)
print(f"Neo4j BERTScore Precision : {df['Neo4j BERTScore P'].mean():.4f}")
print(f"Neo4j BERTScore Recall    : {df['Neo4j BERTScore R'].mean():.4f}")
print(f"Neo4j BERTScore F1        : {df['Neo4j BERTScore F1'].mean():.4f}")