import pandas as pd
import re

print("=======================================================")
print("   STEP 3: COMPUTING CONTEXT PRECISION (CP) FROM CSV")
print("=======================================================\n")

# --- 1. LOAD SAVED BENCHMARK DATA ---
csv_path = "mdpi_benchmark_results.csv"
df = pd.read_csv(csv_path)

# --- 2. CONTEXT PRECISION HELPER ---
def compute_context_precision(context_text, truth):
    """
    Measures sentence/fact-level signal-to-noise ratio:
    (Number of retrieved statements containing the ground-truth answer)
    divided by (Total number of retrieved statements).
    """
    if pd.isna(context_text) or not str(context_text).strip():
        return 0.0

    # Split retrieved context into individual sentences or graph facts
    statements = [
        s.strip()
        for s in re.split(r'(?<=[.!?])\s+', str(context_text).strip())
        if s.strip()
    ]
    if not statements:
        return 0.0

    truth_clean = str(truth).strip().lower()
    relevant_count = sum(1 for stmt in statements if truth_clean in stmt.lower())

    return relevant_count / len(statements)

def count_words(context_text):
    if pd.isna(context_text) or not str(context_text).strip():
        return 0
    return len(str(context_text).split())

# --- 3. CALCULATE FOR ALL 50 QUERIES ---
df["FAISS Context Precision"] = df.apply(
    lambda row: compute_context_precision(row["FAISS Context"], row["Ground Truth"]),
    axis=1
)
df["Neo4j Context Precision"] = df.apply(
    lambda row: compute_context_precision(row["Neo4j Context"], row["Ground Truth"]),
    axis=1
)

# Bonus efficiency stat for edge-compute (6GB VRAM) defense
df["FAISS Context Words"] = df["FAISS Context"].apply(count_words)
df["Neo4j Context Words"] = df["Neo4j Context"].apply(count_words)

# --- 4. SAVE UPDATED CSV ---
df.to_csv(csv_path, index=False)

print(f"[*] Updated '{csv_path}' with Context Precision columns.\n")
print("=======================================================")
print("   CONTEXT PRECISION (CP) & EFFICIENCY RESULTS")
print("=======================================================")
print(f"FAISS Context Precision (CP) : {df['FAISS Context Precision'].mean():.4f} ({df['FAISS Context Precision'].mean() * 100:.1f}%)")
print(f"Neo4j Context Precision (CP) : {df['Neo4j Context Precision'].mean():.4f} ({df['Neo4j Context Precision'].mean() * 100:.1f}%)")
print("-" * 55)
print(f"FAISS Avg Retrieved Words    : {df['FAISS Context Words'].mean():.1f} words/query")
print(f"Neo4j Avg Retrieved Words    : {df['Neo4j Context Words'].mean():.1f} words/query")