# RAG Pipeline Evaluation — Reference Implementation
 
This folder contains a complete reference implementation showing
how to use llm-eval-framework to evaluate a real RAG pipeline,
from dataset design through failure analysis.
 
---
 
## What is this?
 
Most evaluation frameworks show you how to compute metrics.
This reference implementation shows you what to DO with them, like, 
how to design a test dataset, run baseline evaluations, measure
the impact of optimizations, and act on failure analysis.
 
---
 
## Structure
 
```
rag_pipeline_eval/
├── dataset/
│   ├── test_questions.json     # 30 curated test questions
│   └── README.md               # Dataset design decisions
├── results/
│   ├── baseline_results.json   # Scores before optimization
│   ├── baseline_summary.md     # Human-readable baseline report
│   ├── hyde_reranking_results.json  # Scores after HyDE + reranking
│   └── comparison_summary.md   # Before/after comparison
├── notebooks/
│   └── chunking_comparison.ipynb   # Strategy benchmarks
└── FAILURE_ANALYSIS.md         # Root cause analysis + recommendations
```
 
---
 
## The evaluation story
 
**Step 1: Baseline** (Commit 13)
Run the basic RAG pipeline (recursive chunking, similarity retrieval)
against all 30 questions. Establish baseline scores.
 
**Step 2: Optimize** (Commit 14)
Add HyDE retrieval and cross-encoder reranking. Run evaluation again.
Compare scores to see which metrics improved? By how much?
 
**Step 3: Analyze** (Commit 15-16)
Dig into which question categories the pipeline handles poorly.
Identify root causes. Make concrete recommendations.
 
---
 
*Evaluated using [llm-eval-framework](https://github.com/pulkitkushwaha/llm-eval-framework)*
