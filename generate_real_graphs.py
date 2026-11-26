import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("[*] Generating Final Thesis Visualizations from Real Data...")

# Load the real results
df = pd.read_csv("benchmark_results.csv")
sample_size = len(df)

faiss_retrieval = (df['FAISS Retrieval'].sum() / sample_size) * 100
neo4j_retrieval = (df['Neo4j Retrieval'].sum() / sample_size) * 100

faiss_f1 = df['FAISS F1'].mean()
neo4j_f1 = df['Neo4j F1'].mean()

plt.style.use('ggplot')
colors = ['#E24A33', '#348ABD']

# --- GRAPH 1: RETRIEVAL SUCCESS ---
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(['Pipeline A (FAISS)', 'Pipeline B (Neo4j)'], [faiss_retrieval, neo4j_retrieval], color=colors, width=0.5)
ax.set_ylabel('Retrieval Success Rate (%)', fontweight='bold')
ax.set_title('Layer 2 Evaluation: Multi-Hop Evidence Retrieval', fontweight='bold', pad=15)
ax.set_ylim(0, 110)

for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('real_retrieval_comparison.png', dpi=300)
print("[*] Saved 'real_retrieval_comparison.png'")

# --- GRAPH 2: F1 SCORE ---
fig2, ax2 = plt.subplots(figsize=(8, 6))
bars2 = ax2.bar(['Pipeline A (FAISS)', 'Pipeline B (Neo4j)'], [faiss_f1, neo4j_f1], color=colors, width=0.5)
ax2.set_ylabel('Average F1-Score', fontweight='bold')
ax2.set_title('Layer 3 Evaluation: LLM Generation Accuracy', fontweight='bold', pad=15)

for bar in bars2:
    height = bar.get_height()
    ax2.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('real_f1_comparison.png', dpi=300)
print("[*] Saved 'real_f1_comparison.png'")