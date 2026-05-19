# Evaluation Dataset — RAG Pipeline
 
A curated set of 30 test questions for evaluating RAG pipeline
performance across 8 query categories.
 
---
 
## Dataset design philosophy
 
Most RAG evaluation datasets only test factual lookup, the easiest
case. This dataset is designed to stress-test the pipeline across
the full range of query types that appear in production systems.
 
A pipeline that scores well on all 8 categories is genuinely
production-ready. A pipeline that only scores well on factual
queries has significant blind spots.
 
---
 
## Query categories
 
| Category | Count | What it tests |
|---|---|---|
| `factual` | 5 | Basic retrieval: single-hop lookups |
| `multi_hop` | 5 | Combining information from multiple sections |
| `summarization` | 3 | Synthesizing across multiple paragraphs |
| `comparison` | 3 | Retrieving and comparing attributes |
| `negative` | 4 | Retrieving limiting conditions and exceptions |
| `adversarial` | 3 | Handling false premises in questions |
| `out_of_scope` | 2 | Staying grounded: not hallucinating answers |
| `ambiguous` | 2 | Handling underspecified queries |
| `technical` | 3 | Retrieving from technical documentation |
 
---
 
## Dataset format
 
```json
{
  "id": "q001",
  "category": "factual",
  "question": "What is the return policy?",
  "ground_truth": "Items can be returned within 30 days...",
  "answer": "",
  "contexts": [],
  "notes": "Design note about this question"
}
```
 
The `answer` and `contexts` fields are empty — they are populated
at evaluation time by the RAG pipeline under test.
The `ground_truth` is the reference answer used by recall-based metrics.
The `notes` field explains why the question was included and what
specific failure mode it is designed to catch.
 
---
 
## How to use
 
```python
import json
from llm_eval import Evaluator
from llm_eval.models import EvalSample
from llm_eval.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
 
# Load dataset
with open("examples/rag_pipeline_eval/dataset/test_questions.json") as f:
    questions = json.load(f)
 
# Run your RAG pipeline on each question to populate answer + contexts
samples = []
for q in questions:
    answer, contexts = my_rag_pipeline.query(q["question"])
    samples.append(EvalSample(
        question=q["question"],
        answer=answer,
        contexts=contexts,
        ground_truth=q["ground_truth"]
    ))
 
# Evaluate
evaluator = Evaluator(metrics=[
    Faithfulness(),
    AnswerRelevancy(),
    ContextPrecision(),
    ContextRecall()
])
report = evaluator.evaluate(samples)
print(report.summary())
```
 
---
 
## Benchmark results
 
See `../results/` for baseline and post-optimization evaluation
results against this dataset.
