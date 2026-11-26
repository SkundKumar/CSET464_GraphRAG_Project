import pandas as pd

print("============================================================")
print("   STEP 7: EXTRACTING QUALITATIVE CASE STUDIES FROM CSV")
print("============================================================\n")

csv_path = "mdpi_benchmark_results.csv"
df = pd.read_csv(csv_path).copy()

# Clean string columns safely
for col in ["Question", "Ground Truth", "FAISS Answer", "Neo4j Answer", "FAISS Context", "Neo4j Context"]:
    df[col] = df[col].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

df["BERTScore_Gain"] = df["Neo4j BERTScore F1"] - df["FAISS BERTScore F1"]
df["F1_Gain"] = df["Neo4j F1"] - df["FAISS F1"]
df["Neo4j_Contains_Truth"] = df.apply(
    lambda r: r["Ground Truth"].lower() in r["Neo4j Answer"].lower() and len(r["Ground Truth"]) > 0,
    axis=1
)
df["FAISS_Contains_Truth"] = df.apply(
    lambda r: r["Ground Truth"].lower() in r["FAISS Answer"].lower() and len(r["Ground Truth"]) > 0,
    axis=1
)
df["Q_Len"] = df["Question"].str.len()

selected_indices = []
case_roles = {}

# --- CASE 1 & 2: Clear Multi-Hop Wins (FAISS Missed, Neo4j Retrieved & Answered Correctly) ---
hero_pool = df[
    (df["FAISS Recall@K"] == 0) &
    (df["Neo4j Recall@K"] == 1) &
    (df["Neo4j_Contains_Truth"]) &
    (~df["FAISS_Contains_Truth"])
].sort_values(by=["BERTScore_Gain", "F1_Gain", "Q_Len"], ascending=[False, False, True])

# Fallback if fewer than 2 strict matches
if len(hero_pool) < 2:
    hero_pool = df[
        (df["FAISS Recall@K"] == 0) &
        (df["Neo4j Recall@K"] == 1) &
        (df["BERTScore_Gain"] > 0)
    ].sort_values(by=["BERTScore_Gain", "F1_Gain"], ascending=[False, False])

for idx in hero_pool.index[:2]:
    selected_indices.append(idx)
    case_roles[idx] = "Multi-Hop Recovery (FAISS Missed vs. Neo4j Succeeded)"

# --- CASE 3: Verbosity Penalty Proof (Neo4j Right in Meaning, EM=0, Low Token F1, High BERTScore Recall) ---
verbosity_pool = df[
    (~df.index.isin(selected_indices)) &
    (df["Neo4j_Contains_Truth"]) &
    (df["Neo4j EM"] == 0) &
    (df["Neo4j F1"] < 0.35) &
    (df["Neo4j BERTScore R"] > 0.55)
].sort_values(by=["Neo4j BERTScore R", "BERTScore_Gain"], ascending=[False, False])

if len(verbosity_pool) == 0:
    verbosity_pool = df[
        (~df.index.isin(selected_indices)) &
        (df["Neo4j Recall@K"] == 1) &
        (df["BERTScore_Gain"] > 0)
    ].sort_values(by=["Neo4j BERTScore R", "BERTScore_Gain"], ascending=[False, False])

if len(verbosity_pool) > 0:
    idx3 = verbosity_pool.index[0]
    selected_indices.append(idx3)
    case_roles[idx3] = "Verbosity Penalty Proof (Correct Meaning, EM=0, High BERTScore)"

# --- BUILD OUTPUT TABLE ---
cases_df = df.loc[selected_indices].copy()
cases_df.insert(0, "Case Type", [case_roles[i] for i in selected_indices])
cases_df.insert(0, "Case #", [f"Case {n}" for n in range(1, len(cases_df) + 1)])

export_cols = [
    "Case #", "Case Type", "Question", "Ground Truth",
    "FAISS Recall@K", "FAISS Context Words", "FAISS Answer", "FAISS F1", "FAISS BERTScore F1",
    "Neo4j Recall@K", "Neo4j Context Words", "Neo4j Answer", "Neo4j F1", "Neo4j BERTScore R", "Neo4j BERTScore F1"
]
cases_df[export_cols].to_csv("ieee_case_studies.csv", index=False)

# --- WRITE MARKDOWN TABLE FOR IEEE PAPER ---
md_lines = [
    "# Qualitative Case Studies (2WikiMultihopQA, Local RTX 4050)\n",
    "| Case | Category | Question | Ground Truth | Pipeline A: FAISS Answer (Recall / F1 / BERT-F1) | Pipeline B: Neo4j Answer (Recall / F1 / BERT-F1) |",
    "| :--- | :--- | :--- | :--- | :--- | :--- |"
]

def clean_md(s):
    return str(s).replace("|", "\\|").replace("\n", " ").strip()

for _, row in cases_df.iterrows():
    f_cell = (
        f"{clean_md(row['FAISS Answer'])} "
        f"*(Rec: {int(row['FAISS Recall@K'])}, Words: {int(row['FAISS Context Words'])}, "
        f"F1: {row['FAISS F1']:.3f}, BERT-F1: {row['FAISS BERTScore F1']:.3f})*"
    )
    n_cell = (
        f"{clean_md(row['Neo4j Answer'])} "
        f"*(Rec: {int(row['Neo4j Recall@K'])}, Words: {int(row['Neo4j Context Words'])}, "
        f"F1: {row['Neo4j F1']:.3f}, BERT-F1: {row['Neo4j BERTScore F1']:.3f})*"
    )
    md_lines.append(
        f"| {row['Case #']} | {clean_md(row['Case Type'])} | {clean_md(row['Question'])} | "
        f"**{clean_md(row['Ground Truth'])}** | {f_cell} | {n_cell} |"
    )

with open("ieee_case_studies_table.md", "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines) + "\n")

# --- PRINT CLEAN SUMMARY TO TERMINAL ---
print("[*] Saved 'ieee_case_studies.csv' and 'ieee_case_studies_table.md'.\n")
for _, row in cases_df.iterrows():
    print("-" * 68)
    print(f"{row['Case #']} | {row['Case Type']}")
    print(f"Question     : {row['Question']}")
    print(f"Ground Truth : {row['Ground Truth']}")
    print(f"FAISS (Rec={int(row['FAISS Recall@K'])}, Words={int(row['FAISS Context Words'])}, F1={row['FAISS F1']:.3f}, BERT-F1={row['FAISS BERTScore F1']:.3f}):")
    print(f"  -> {row['FAISS Answer']}")
    print(f"Neo4j (Rec={int(row['Neo4j Recall@K'])}, Words={int(row['Neo4j Context Words'])}, F1={row['Neo4j F1']:.3f}, BERT-R={row['Neo4j BERTScore R']:.3f}, BERT-F1={row['Neo4j BERTScore F1']:.3f}):")
    print(f"  -> {row['Neo4j Answer']}")

print("-" * 68)
print("\n[INFO] Additional Stats Across All 50 Queries:")
print(f"  Queries where FAISS Missed (Rec=0) & Neo4j Succeeded (Rec=1): {len(df[(df['FAISS Recall@K']==0) & (df['Neo4j Recall@K']==1)])}/50")
print(f"  Queries where Neo4j Answer explicitly contained Ground Truth : {int(df['Neo4j_Contains_Truth'].sum())}/50")
print(f"  Queries where FAISS Answer explicitly contained Ground Truth : {int(df['FAISS_Contains_Truth'].sum())}/50")