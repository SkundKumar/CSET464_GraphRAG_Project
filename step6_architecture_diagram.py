import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

print("============================================================")
print("   STEP 6: GENERATING MDPI/IEEE SYSTEM ARCHITECTURE DIAGRAM")
print("============================================================\n")

# --- IEEE PUBLICATION STYLING ---
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

fig, ax = plt.subplots(figsize=(14.0, 6.8))
ax.set_xlim(0, 14.0)
ax.set_ylim(0, 6.8)
ax.axis("off")

# --- COLOR PALETTE (MATCHES FIG 1–3, PRINT-SAFE) ---
BG_PHASE1 = "#F8F9FA"
BG_PHASE2 = "#F4F6F9"
BORDER_PHASE = "#7F8C8D"

FAISS_FILL = "#EBEDEF"
FAISS_EDGE = "#5D6D7E"

NEO4J_FILL = "#EBF5FB"
NEO4J_EDGE = "#1B4F72"

SHARED_FILL = "#FDFEFE"
SHARED_EDGE = "#2C3E50"

LLM_FILL = "#FEF9E7"
LLM_EDGE = "#9A7D0A"

EVAL_FILL = "#F4ECF7"
EVAL_EDGE = "#5B2C6F"

def draw_box(x, y, w, h, text, fill, edge, fontsize=8.5, bold_first=True, lw=1.2, linestyle="-"):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.06,rounding_size=0.10",
        facecolor=fill,
        edgecolor=edge,
        linewidth=lw,
        linestyle=linestyle,
        zorder=3
    )
    ax.add_patch(box)

    lines = text.split("\n")
    if bold_first and len(lines) > 1:
        title = lines[0]
        body = "\n".join(lines[1:])
        ax.text(
            x + w / 2, y + h * 0.74, title,
            ha="center", va="center", fontsize=fontsize + 0.4,
            fontweight="bold", color="#111111", zorder=4
        )
        ax.text(
            x + w / 2, y + h * 0.35, body,
            ha="center", va="center", fontsize=fontsize - 0.3,
            color="#222222", linespacing=1.25, zorder=4
        )
    else:
        ax.text(
            x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold" if bold_first else "normal",
            color="#111111", linespacing=1.25, zorder=4
        )

def draw_arrow(p1, p2, color="#2C3E50", lw=1.3, style="-|>", rad=0.0):
    arrow = FancyArrowPatch(
        p1, p2,
        arrowstyle=style,
        mutation_scale=12,
        color=color,
        linewidth=lw,
        connectionstyle=f"arc3,rad={rad}",
        zorder=5
    )
    ax.add_patch(arrow)

def draw_polyline_arrow(points, color="#2C3E50", lw=1.3):
    xs = [p[0] for p in points[:-1]]
    ys = [p[1] for p in points[:-1]]
    if len(points) > 2:
        ax.plot(xs, ys, color=color, linewidth=lw, zorder=5)
    draw_arrow(points[-2], points[-1], color=color, lw=lw)

# --- FIGURE TITLE ---
ax.text(
    7.0, 6.50,
    "Edge-Compute Comparative RAG Architecture: Naive Vector RAG (FAISS) vs. Edge GraphRAG (Neo4j)",
    ha="center", va="center", fontsize=12, fontweight="bold", color="#111111"
)

# --- PHASE CONTAINERS ---
phase1_box = FancyBboxPatch(
    (0.20, 0.55), 6.00, 5.55,
    boxstyle="round,pad=0.05,rounding_size=0.12",
    facecolor=BG_PHASE1, edgecolor=BORDER_PHASE,
    linewidth=1.1, linestyle="--", zorder=1
)
ax.add_patch(phase1_box)
ax.text(
    3.20, 5.90, "PHASE I: OFFLINE INGESTION & KNOWLEDGE INDEXING",
    ha="center", va="center", fontsize=9.5, fontweight="bold", color="#2C3E50", zorder=2
)

phase2_box = FancyBboxPatch(
    (6.45, 0.55), 7.35, 5.55,
    boxstyle="round,pad=0.05,rounding_size=0.12",
    facecolor=BG_PHASE2, edgecolor=BORDER_PHASE,
    linewidth=1.1, linestyle="--", zorder=1
)
ax.add_patch(phase2_box)
ax.text(
    10.12, 5.90, "PHASE II: ONLINE MULTI-HOP INFERENCE & MULTI-LAYER EVALUATION",
    ha="center", va="center", fontsize=9.5, fontweight="bold", color="#2C3E50", zorder=2
)

# ============================================================
# PHASE I BOXES
# ============================================================
# Dataset Input
draw_box(
    0.40, 2.65, 1.45, 1.30,
    "2WikiMultihopQA\nValidation Split\nWikipedia Contexts\n& Evidence Triples",
    SHARED_FILL, SHARED_EDGE, fontsize=8.3
)

# Track A: Naive Vector RAG (Top)
draw_box(
    2.30, 4.25, 1.70, 1.15,
    "Dense Encoder\nall-MiniLM-L6-v2\nParagraph Chunking\n(384-d Vectors)",
    FAISS_FILL, FAISS_EDGE, fontsize=8.3
)
draw_box(
    4.40, 4.25, 1.60, 1.15,
    "FAISS Vector Store\nIndexFlatL2\nUnstructured Flat\nEmbeddings",
    FAISS_FILL, FAISS_EDGE, fontsize=8.3
)

# Track B: Edge GraphRAG (Bottom)
draw_box(
    2.30, 1.25, 1.70, 1.15,
    "DistilBERT Extractor\n(67M Params, SFT)\nRelation Extraction\nTriples: (S, R, O)",
    NEO4J_FILL, NEO4J_EDGE, fontsize=8.3
)
draw_box(
    4.40, 1.25, 1.60, 1.15,
    "Neo4j Graph DB\nCypher MERGE\nStructured Entities\n& Typed Relations",
    NEO4J_FILL, NEO4J_EDGE, fontsize=8.3
)

# Track labels inside Phase I
ax.text(4.15, 5.56, "Pipeline A: Naive Vector RAG Baseline", ha="center", va="center",
        fontsize=8.3, fontweight="bold", color=FAISS_EDGE)
ax.text(4.15, 0.85, "Pipeline B: Proposed Edge GraphRAG (Strict Extractor Isolation - Zero Cloud API)",
        ha="center", va="center", fontsize=8.1, fontweight="bold", color=NEO4J_EDGE)

# Phase I Arrows
draw_polyline_arrow([(1.85, 3.55), (2.08, 3.55), (2.08, 4.82), (2.30, 4.82)], color=FAISS_EDGE)
draw_polyline_arrow([(1.85, 3.05), (2.08, 3.05), (2.08, 1.82), (2.30, 1.82)], color=NEO4J_EDGE)
draw_arrow((4.00, 4.82), (4.40, 4.82), color=FAISS_EDGE)
draw_arrow((4.00, 1.82), (4.40, 1.82), color=NEO4J_EDGE)

# ============================================================
# PHASE II BOXES
# ============================================================
# User Query
draw_box(
    6.75, 2.75, 1.65, 1.10,
    "Multi-Hop Query\n2WikiMultihopQA\n(N = 50 Benchmark)",
    SHARED_FILL, SHARED_EDGE, fontsize=8.3
)

# Retrieval A (Top)
draw_box(
    6.68, 4.25, 1.80, 1.15,
    "Top-K Vector Search\nFAISS L2 (k = 2)\nRetrieved Context A\nAvg: 136.4 words",
    FAISS_FILL, FAISS_EDGE, fontsize=8.2
)

# Retrieval B (Bottom)
draw_box(
    6.68, 1.25, 1.80, 1.15,
    "Cypher Traversal\nGraph Fact Retrieval\nRetrieved Context B\nAvg: 16.9 words (-87.6%)",
    NEO4J_FILL, NEO4J_EDGE, fontsize=8.1
)

# Frozen Generator LLM
draw_box(
    9.10, 2.40, 2.00, 1.80,
    "Frozen Generator LLM\nQwen2.5-3B-Instruct\n4-bit NF4 Quantization\nLocal RTX 4050 (6GB)\nIndependent A / B Runs",
    LLM_FILL, LLM_EDGE, fontsize=8.3, lw=1.4
)

# Multi-Layer Evaluation Container & Sub-cards
eval_outer = FancyBboxPatch(
    (11.45, 1.15), 2.15, 4.30,
    boxstyle="round,pad=0.05,rounding_size=0.10",
    facecolor=EVAL_FILL, edgecolor=EVAL_EDGE,
    linewidth=1.3, zorder=3
)
ax.add_patch(eval_outer)
ax.text(
    12.525, 5.20, "Multi-Layer Evaluation\n(Offline / Deterministic)",
    ha="center", va="center", fontsize=8.6, fontweight="bold", color=EVAL_EDGE, zorder=4
)

draw_box(
    11.60, 3.25, 1.85, 1.55,
    "Layer 2: Retrieval\n- Recall@2 (Hit Rate)\n- MRR@2 (Rank)\n- Context Precision\n- Token/Word Overhead",
    "#FFFFFF", EVAL_EDGE, fontsize=7.9, lw=0.9
)
draw_box(
    11.60, 1.35, 1.85, 1.65,
    "Layer 3: Generation\n- Exact Match (EM)\n- Token F1-Score\n- BERTScore (P/R/F1)\n- Verbosity Analysis",
    "#FFFFFF", EVAL_EDGE, fontsize=7.9, lw=0.9
)

# ============================================================
# PHASE I -> PHASE II & INTERNAL PHASE II ARROWS
# ============================================================
# DB -> Retrieval across phase boundary
draw_arrow((6.00, 4.82), (6.68, 4.82), color=FAISS_EDGE, lw=1.4)
draw_arrow((6.00, 1.82), (6.68, 1.82), color=NEO4J_EDGE, lw=1.4)

# Query -> Retrieval A, Retrieval B, and Generator Prompt
draw_arrow((7.58, 3.85), (7.58, 4.25), color=SHARED_EDGE)
draw_arrow((7.58, 2.75), (7.58, 2.40), color=SHARED_EDGE)
draw_arrow((8.40, 3.30), (9.10, 3.30), color=SHARED_EDGE)
ax.text(8.75, 3.43, "Query", ha="center", va="bottom", fontsize=7.5, color=SHARED_EDGE)

# Retrieval A & B -> Frozen Generator
draw_polyline_arrow([(8.48, 4.82), (8.80, 4.82), (8.80, 3.85), (9.10, 3.85)], color=FAISS_EDGE)
ax.text(8.80, 4.92, "Context A", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=FAISS_EDGE)

draw_polyline_arrow([(8.48, 1.82), (8.80, 1.82), (8.80, 2.75), (9.10, 2.75)], color=NEO4J_EDGE)
ax.text(8.80, 1.70, "Context B", ha="center", va="top", fontsize=7.5, fontweight="bold", color=NEO4J_EDGE)

# Generator + Context Logs -> Evaluation Suite
draw_arrow((11.10, 3.30), (11.45, 3.30), color=EVAL_EDGE, lw=1.5)
ax.text(11.27, 3.60, "CSV\nLogs", ha="center", va="bottom", fontsize=7.3, fontweight="bold", color=EVAL_EDGE)

# Hardware Constraint Footer Note
ax.text(
    7.0, 0.22,
    "Hardware Environment: 100% Local Edge Execution on NVIDIA RTX 4050 (6GB VRAM) | Strict Separation of Extractor (DistilBERT) and Generator (Qwen2.5-3B)",
    ha="center", va="center", fontsize=8.3, style="italic", color="#444444"
)

output_file = "fig0_system_architecture.png"
fig.savefig(output_file)
plt.close(fig)
print(f"[+] Saved Architecture Diagram: {output_file} (300 DPI)")