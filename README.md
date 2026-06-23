# llm-eval-framework

[![Tests](https://github.com/pulkitkushwaha/llm-eval-framework/actions/workflows/tests.yml/badge.svg)](https://github.com/pulkitkushwaha/llm-eval-framework/actions/workflows/tests.yml)
[![Eval](https://github.com/pulkitkushwaha/llm-eval-framework/actions/workflows/eval.yml/badge.svg)](https://github.com/pulkitkushwaha/llm-eval-framework/actions/workflows/eval.yml)

I remember watching a YouTube video regarding the same evaluation topic, and the teacher said a line which stuck with me like "Mitochondria being the powerhouse of the cell." She said that "You cannot improve something you cannot measure." And that was it. It opened my way of thinking towards evaluation from a developer's perspective.

Hence, I have tried to create a standalone evaluation framework for RAG pipelines and LLM systems: built for engineers who want to measure their systems before shipping them, not after something breaks in production.

> "Shipping a RAG pipeline without evaluation is like deploying an API without tests. You might get away with it for a while. But you won't know when it breaks, why it broke, or how to fix it."

---

## Why is evaluation hard to skip?

Most RAG pipelines fail silently. The system returns an answer, which looks reasonable, it's grammatically correct, it sounds confident. But it's wrong. Or it's right for the wrong reason. Or it was right last week and wrong this week because someone updated the knowledge base.

The problem is that "it looks good" is not a metric. Human spot-checks don't scale. And by the time a user complains, the damage is already done.

Evaluation solves this by giving you **quantitative signals** that answer the questions you actually care about in production:

- Is my pipeline making things up, or grounding answers in retrieved content?
- Are my retrieved chunks actually relevant to the question being asked?
- Is my retriever finding everything it needs, or missing critical information?
- Are my answers actually addressing what the user asked?

These aren't academic questions. They're the difference between a RAG system that works and one that quietly erodes user trust.

---

## Quick start

```python
from llm_eval import Evaluator
from llm_eval.models import EvalSample
from llm_eval.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from llm_eval.reporters import MarkdownReporter

# Define your test samples
samples = [
    EvalSample(
        question="What is the return policy for online orders?",
        answer="Online orders can be returned within 30 days of purchase.",
        contexts=["Our return policy allows returns within 30 days for all purchases."],
        ground_truth="Items purchased online can be returned within 30 days."
    )
]

# Run evaluation
evaluator = Evaluator(
    metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()]
)
report = evaluator.evaluate(samples)
print(report.summary())

# Save as markdown (great for PR comments)
MarkdownReporter().save(report, "results/eval_report.md")
```

**CLI:**
```bash
python -m llm_eval \
  --dataset examples/rag_pipeline_eval/dataset/test_questions.json \
  --metrics faithfulness,answer_relevancy,context_precision,context_recall \
  --output markdown
```

**Expected output:**
```
╔══════════════════════════════════════════════════════╗
║           LLM Evaluation Report                      ║
╠══════════════════════════════════════════════════════╣
║ Samples evaluated:     30                            ║
║ Metrics run:           4                             ║
╠══════════════════════════════════════════════════════╣
║ Faithfulness           0.87              Good        ║
║ Answer Relevancy       0.91              Excellent   ║
║ Context Precision      0.76              Moderate    ║
║ Context Recall         0.82              Good        ║
╠══════════════════════════════════════════════════════╣
║ Overall Score          0.84              Good        ║
╚══════════════════════════════════════════════════════╝
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  llm-eval-framework                     │
│                                                         │
│  ┌──────────┐   ┌──────────────┐   ┌────────────────┐  │
│  │EvalSample│──▶│   Evaluator  │──▶│   EvalReport   │  │
│  │ (input)  │   │ (orchestrate)│   │   (output)     │  │
│  └──────────┘   └──────┬───────┘   └───────┬────────┘  │
│                        │                   │            │
│               ┌────────▼────────┐  ┌───────▼────────┐  │
│               │   BaseMetric    │  │   Reporters    │  │
│               │   (abstract)    │  │ JSON/Markdown  │  │
│               └────────┬────────┘  └────────────────┘  │
│                        │                               │
│         ┌──────────────┼──────────────┐               │
│         │              │              │               │
│  ┌──────▼─────┐ ┌──────▼─────┐ ┌─────▼──────┐        │
│  │Faithfulness│ │  Answer    │ │  Context   │  ...   │
│  │            │ │ Relevancy  │ │ Precision  │        │
│  └────────────┘ └────────────┘ └────────────┘        │
└─────────────────────────────────────────────────────────┘
```

---

## What this framework measures

There are four fundamental failure modes in a RAG pipeline: **Faithfulness, Answer Relevancy, Context Precision** and **Context Recall**. Each maps to a metric.

---

### 1. Faithfulness: *did the model make things up?*

**What it measures:**
Whether every claim in the generated answer is actually supported by the retrieved context. A faithful answer contains only information that can be directly traced back to what was retrieved, nothing more.

**How it works:**
The answer is decomposed into atomic claims, i.e. individual factual statements. Each claim is then checked against the retrieved chunks using an LLM call: *"Is this claim supported by the following context?"* The faithfulness score is the fraction of claims that pass this check.

```
Faithfulness = (number of claims supported by context) / (total claims in answer)
```

**Score range:** 0.0 to 1.0. Higher is better.

**What a low score means:**
The model is hallucinating: generating content that isn't in the retrieved context. This is the most dangerous failure mode. A faithfulness score below 0.7 in production means your users are receiving fabricated information presented as fact.

**Where this metric fails:**
Faithfulness only checks whether claims are grounded, it doesn't check whether the retrieved context itself is accurate. If your knowledge base contains incorrect information, a perfectly faithful answer can still be wrong. Faithfulness measures retrieval-generation alignment, not factual correctness against the real world.

**When to prioritize it:**
Always, but especially in high-stakes domains like legal, medical, financial, or compliance systems where hallucinated content has real consequences.

---

### 2. Answer Relevancy: *did the model actually answer what was asked?*

**What it measures:**
Whether the generated answer directly addresses the user's question. A highly relevant answer is focused, complete, and on-topic. A low-relevancy answer might be faithful to the context but tangential to the actual question.

**How it works:**
The answer is used to generate multiple reverse questions: *"What question would this answer be responding to?"* These reverse questions are then compared to the original question using embedding similarity. High similarity means the answer is on-topic, low similarity means the answer drifted.

```
Answer Relevancy = mean(cosine_similarity(reverse_question_i, original_question))
```

**Score range:** 0.0 to 1.0. Higher is better.

**What a low score means:**
The model retrieved something related but not quite right, and generated an answer that technically uses the context but doesn't address what the user actually wanted. Common in vague queries or when the retriever pulls adjacent-but-not-relevant chunks.

**Where this metric fails:**
Answer relevancy doesn't penalize incomplete answers, meaning an answer that correctly addresses part of the question but misses half of it can still score well. It measures topical alignment, not completeness. Pair it with context recall to catch this gap.

**When to prioritize it:**
When your users are asking complex or multi-part questions. Low answer relevancy often points to a retrieval problem — the right documents aren't being surfaced.

---

### 3. Context Precision: *is your retriever finding signal or noise?*

**What it measures:**
Whether the retrieved chunks are actually relevant to the question. High context precision means your retriever is doing a good job: it's returning content that's useful for answering the question. Low precision means you're flooding the context window with irrelevant content.

**How it works:**
Each retrieved chunk is evaluated for relevance to the question: *"Is this chunk useful for answering this question?"* Context precision is the fraction of retrieved chunks that are relevant, weighted by their position (chunks ranked higher should be more relevant).

```
Context Precision = weighted mean of relevance scores across retrieved chunks
```

**Score range:** 0.0 to 1.0. Higher is better.

**What a low score means:**
Your retriever is returning chunks that look semantically similar to the query but don't actually contain the information needed to answer it. This wastes context window space, increases token costs, and confuses the LLM, often leading to hallucination or irrelevant answers.

**Where this metric fails:**
Context precision only evaluates the chunks you retrieved. It has no visibility into what you didn't retrieve. A retriever that returns 3 perfectly relevant chunks out of 3 scores 1.0, even if there were 10 other relevant chunks it missed entirely. That's context recall's job.

**When to prioritize it:**
When you're debugging retrieval quality — especially if faithfulness is high but answer relevancy is low. Low context precision often points to embedding model quality or chunking strategy problems.

---

### 4. Context Recall: *is your retriever missing critical information?*

**What it measures:**
Whether all the information needed to answer the question correctly was present in the retrieved chunks. High recall means your retriever found everything — low recall means it missed chunks that contained critical information.

**How it works:**
The ground truth answer is decomposed into atomic statements. Each statement is checked against the retrieved context: *"Can this statement be attributed to the retrieved chunks?"* Context recall is the fraction of ground truth statements that can be found in the retrieved context.

```
Context Recall = (ground truth statements found in context) / (total ground truth statements)
```

**Score range:** 0.0 to 1.0. Higher is better.

**What a low score means:**
Your retriever is missing relevant chunks. The information exists in your knowledge base, but your retrieval strategy isn't finding it. Common causes: chunk size too small, embedding model not capturing semantic meaning well, top-k too low, or poor document structure causing relevant content to be split across chunks.

**Where this metric fails:**
Context recall requires ground truth answers. You need to know what the correct answer should be before you can measure whether the context contains it. This makes it more expensive to evaluate at scale. It also can't tell you whether the missed information was in your knowledge base at all. It can only tell you whether it was in what you retrieved.

**When to prioritize it:**
When users report that the system "doesn't know" things that should be in the knowledge base. Low context recall is almost always a retrieval problem: chunking strategy, embedding model, or top-k settings.

---

## The metric matrix: reading your scores together

We have selected 4 evaluation metrics, as no single metric tells the full story — always remember this. It is the combination of those metrics that gives the bigger picture of what is exactly going on. Co-relating these answers is what gives more insights about the current system performance. Here's how to interpret combinations:

| Faithfulness | Answer Relevancy | Context Precision | Context Recall | Diagnosis |
|---|---|---|---|---|
| ✅ High | ✅ High | ✅ High | ✅ High | Pipeline is working well |
| ❌ Low | ✅ High | ✅ High | ✅ High | LLM is hallucinating despite good retrieval |
| ✅ High | ❌ Low | ✅ High | ✅ High | Answers are grounded but off-topic — prompt issue |
| ✅ High | ✅ High | ❌ Low | ✅ High | Retriever returning noise — embedding or chunking issue |
| ✅ High | ✅ High | ✅ High | ❌ Low | Retriever missing chunks — top-k or chunk size issue |
| ❌ Low | ❌ Low | ❌ Low | ❌ Low | Fundamental retrieval failure — start troubleshooting with chunking |
| ✅ High | ❌ Low | ❌ Low | ❌ Low | LLM is faithful to bad context — retrieval is the problem |

I hope this example gives you an idea of how to analyse and derive insights from the metrics we have been calculating.

---

## RAGAS vs custom metrics: when to use each

This framework supports both. Here's when to reach for each:

**Use RAGAS when:**
- You want a battle-tested, well-documented baseline
- You're evaluating a general-purpose RAG pipeline
- You want to compare your pipeline against published benchmarks
- Speed of setup matters more than customization

**Use custom metrics when:**
- Your domain has specific requirements RAGAS doesn't cover (e.g. citation accuracy, policy compliance, structured output correctness)
- You need to evaluate aspects of your pipeline that aren't retrieval or generation quality, i.e, latency, cost, format adherence
- You want full control over the LLM used for evaluation
- You're building evaluation into a CI/CD pipeline and need deterministic, lightweight checks

**Use both when:**
- You want RAGAS scores for benchmarking and comparison
- Plus custom metrics for domain-specific requirements
- This framework lets you mix them in the same evaluation run

---

## Reference implementation — evaluating a real RAG pipeline

The `examples/rag_pipeline_eval/` folder contains a complete end-to-end evaluation of the [rag-pipeline](https://github.com/pulkitkushwaha/rag-pipeline) repo across multiple retrieval strategies.

### Results summary

**Baseline** (recursive chunking + similarity retrieval):

| Metric | Score | Status |
|---|---|---|
| Faithfulness | `0.7823` | ✅ Pass |
| Answer Relevancy | `0.7654` | ✅ Pass |
| Context Precision | `0.6891` | ❌ Fail |
| Context Recall | `0.6201` | ❌ Fail |
| **Overall** | **`0.7142`** | 🟡 **Marginal** |

**Optimized** (sentence-window chunking + HyDE + cross-encoder reranking):

| Metric | Score | Delta | Status |
|---|---|---|---|
| Faithfulness | `0.8534` | +9.1% | ✅ Pass |
| Answer Relevancy | `0.8312` | +8.6% | ✅ Pass |
| Context Precision | `0.7989` | +15.9% | ✅ Pass |
| Context Recall | `0.7789` | +25.6% | ✅ Pass |
| **Overall** | **`0.8156`** | **+14.2%** | ✅ **Good** |

All 4 metrics now pass the 0.7 threshold. Context recall improved most dramatically — driven by switching to sentence-window chunking.

**Key finding:** Chunking strategy was the single biggest lever. Switching from recursive to sentence-window chunking drove a 25.6% improvement in context recall — more than HyDE and reranking combined. This only became visible through systematic evaluation.

See the full analysis:
- [`results/baseline_summary.md`](examples/rag_pipeline_eval/results/baseline_summary.md)
- [`results/comparison_summary.md`](examples/rag_pipeline_eval/results/comparison_summary.md)
- [`FAILURE_ANALYSIS.md`](examples/rag_pipeline_eval/FAILURE_ANALYSIS.md)
- [`notebooks/chunking_comparison.ipynb`](examples/rag_pipeline_eval/notebooks/chunking_comparison.ipynb)

---

## CI/CD integration

The GitHub Actions workflow runs evaluation on every PR and posts results as a comment. Exit code 1 if overall score < 0.7 — acts as a quality gate that blocks merges on score regressions.

```yaml
# .github/workflows/eval.yml
- name: Run evaluation
  run: |
    python -m llm_eval \
      --dataset examples/rag_pipeline_eval/dataset/test_questions.json \
      --metrics faithfulness,answer_relevancy \
      --output json \
      --save ci-results/report.json
```

---

## Extending with custom metrics

```python
from llm_eval.base import BaseMetric
from llm_eval.models import EvalSample, MetricResult

class CitationAccuracy(BaseMetric):
    name = "citation_accuracy"
    threshold = 0.8

    def score(self, sample: EvalSample) -> MetricResult:
        score = self._compute_citation_match(sample.answer, sample.contexts)
        return MetricResult(
            metric_name=self.name,
            score=score,
            reasoning=f"Citation match: {score:.3f}",
            threshold=self.threshold
        )

# Use alongside built-in metrics
evaluator = Evaluator(metrics=[Faithfulness(), CitationAccuracy()])
```

---

## What this library provides

```
llm-eval-framework/
├── llm_eval/
│   ├── evaluator.py        # Unified Evaluator class — runs any combination of metrics
│   ├── models.py           # EvalSample, MetricResult, EvalReport data models
│   ├── base.py             # BaseMetric abstract class — extend to add custom metrics
│   ├── metrics/
│   │   ├── faithfulness.py         # Faithfulness metric implementation
│   │   ├── answer_relevancy.py     # Answer relevancy metric implementation
│   │   ├── context_precision.py    # Context precision metric implementation
│   │   ├── context_recall.py       # Context recall metric implementation
│   │   └── ragas_wrapper.py        # RAGAS integration wrapper
│   ├── reporters/
│   │   ├── json_reporter.py        # Machine-readable JSON output
│   │   └── markdown_reporter.py    # Human-readable markdown tables
│   └── cli.py              # CLI interface
├── examples/
│   └── rag_pipeline_eval/  # Reference implementation — evaluating a real RAG pipeline
│       ├── dataset/        # 30 curated test questions with ground truth
│       ├── results/        # Baseline and post-optimization eval results
│       └── notebooks/      # Chunking strategy comparison and failure analysis
├── tests/                  # Unit tests for all metrics
├── ARCHITECTURE.md         # System design and decisions
└── .github/workflows/      # CI/CD pipelines
```

---

## Design principles

**Pipeline-agnostic:** Works with any RAG implementation, be it LangChain, LlamaIndex, Haystack, or custom. As long as you can produce (question, answer, contexts, ground_truth) tuples, the evaluator works.

**Metric-agnostic:** Built-in metrics, RAGAS metrics, and custom metrics all run through the same interface. Mix and match freely.

**CI/CD friendly:** JSON output format is designed to be consumed by GitHub Actions, Jenkins, or any CI pipeline. Run evals on every PR, catch regressions before they reach production.

**Honest about limitations:** Every metric in this framework has documented failure modes. Evaluation is not a solved problem. This framework helps you measure, but it can't replace domain expertise and human judgment for high-stakes decisions.

---

## Lessons learned

**1. Measure context recall first.**
In our evaluation, context recall was the weakest metric and had the highest impact when improved. Most engineers optimize generation before diagnosing retrieval — this is backwards. Measure first.

**2. Chunking strategy matters more than retrieval algorithm.**
Sentence-window chunking improved recall by 25.6%. Adding HyDE improved it by roughly 8%. Get chunking right before adding retrieval complexity.

**3. Some failures are not retrieval problems.**
Adversarial queries (false premises) and ambiguous queries scored poorly even after retrieval optimization. These require upstream guardrails and agentic query expansion — not better chunking. See [llm-guardrails](https://github.com/pulkitkushwaha/llm-guardrails) and [multi-agent-system](https://github.com/pulkitkushwaha/multi-agent-system).

**4. Automated evaluation enables faster iteration.**
With the CLI and CI/CD integration, each retrieval change produces a new eval report in minutes. This feedback loop is what makes optimization systematic rather than intuition-driven.

---

## Related repos

| Repo | How it relates |
|---|---|
| [rag-pipeline](https://github.com/pulkitkushwaha/rag-pipeline) | The pipeline evaluated in `examples/` |
| [llm-guardrails](https://github.com/pulkitkushwaha/llm-guardrails) | Fixes failures this framework identifies |
| [llm-security-playbook](https://github.com/pulkitkushwaha/llm-security-playbook) | Security testing for RAG systems |
| [multi-agent-system](https://github.com/pulkitkushwaha/multi-agent-system) | Agentic fixes for system-design failures |

---

## Status

| Component | Status |
|---|---|
| Base metric interface + data models | ✅ Done |
| Faithfulness metric | ✅ Done |
| Answer relevancy metric | ✅ Done |
| Context precision metric | ✅ Done |
| Context recall metric | ✅ Done |
| RAGAS integration wrapper | ✅ Done |
| JSON + Markdown reporters | ✅ Done |
| CLI interface | ✅ Done |
| Reference implementation (RAG pipeline eval) | ✅ Done |
| GitHub Actions CI/CD workflow | ✅ Done |

---

*Part of the [ai-engineering-portfolio](https://github.com/pulkitkushwaha/ai-engineering-portfolio) — built by [Pulkit Kushwaha](https://linkedin.com/in/pulkit-kushwaha)*
