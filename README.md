# Edge GraphRAG

This project compares a FAISS-based vector RAG pipeline with a Neo4j GraphRAG
pipeline for multi-hop question answering. It includes benchmark scripts,
evaluation and paper-generation steps, result tables, figures, and project
documents.

## Repository contents

- `run_full_benchmark.py` and `run_mdpi_benchmark.py`: benchmark runs.
- `step3_*.py` through `step8_*.py`: evaluation, chart, case-study, and paper
  generation steps.
- `benchmark_results.csv` and `mdpi_benchmark_results.csv`: saved benchmark
  results.
- `Edge_graphRAG(E23cseu1124)/`: paper source and document.
- The root-level Word documents: project definition, survey, contribution, and
  architecture materials.

## Setup

Use Python 3.10 or newer. Install dependencies from `requirements.txt`. PyTorch
and `bitsandbytes` GPU support depends on your operating system and CUDA setup;
install the PyTorch build appropriate for your machine if needed.

Run Neo4j locally, then configure its connection before running graph scripts.
The default URI is `bolt://localhost:7687` and the default username is `neo4j`.
Set the password in the environment rather than putting it in source code.

PowerShell:

```powershell
$env:NEO4J_PASSWORD = "your-local-neo4j-password"
python run_mdpi_benchmark.py
```

Optional connection variables are `NEO4J_URI` and `NEO4J_USERNAME`.

The trained relation-extractor checkpoint is intentionally not stored in this
repository because its weights exceed GitHub's regular file-size limit. Place
the checkpoint files in `relation_extractor_model/` before running
`test_model.py`. The benchmark scripts also download the public dataset and
language models when first run.
