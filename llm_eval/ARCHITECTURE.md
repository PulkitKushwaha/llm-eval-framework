# ARCHITECTURE.md

## System Architecture
 
This document explains the technical design of `llm-eval-framework`: the components, their responsibilities, the data flow, and the reasoning behind key design decisions. It is intended for contributors, reviewers, and engineers who want to understand how the system works under the hood.
 
---
 
## Design Principles
 
Four principles shaped every architectural decision in this framework:
 
**1. Pipeline-agnostic by design**
The framework must work with any RAG implementation, be it LangChain, LlamaIndex, Haystack, or completely custom. This means the input contract (what the framework accepts) must be the minimum viable interface: question, answer, contexts, ground_truth. Nothing framework-specific ever enters the evaluation layer.
 
**2. Contract-first development**
Data models and abstract interfaces are defined before implementations. The `EvalSample`, `MetricResult`, and `EvalReport` models define the language the entire system speaks. The `BaseMetric` abstract class defines the contract every metric must follow. This forces consistency and makes the system predictable.
 
**3. Separation of concerns**
Evaluation logic (metrics), orchestration (Evaluator), and output formatting (reporters) are completely separate layers. A metric never knows how its results will be displayed. The Evaluator never knows how a metric computes its score. Reporters never know anything about evaluation logic.
 
**4. Fail gracefully, not silently**
A single failing metric should never crash an entire evaluation run. Errors are caught, logged with context, and represented as zero-score results. The run completes. The failure is visible. This is essential for CI/CD integration where a flaky metric shouldn't block a deployment pipeline.
 
---
 
## Component Overview
 
```
┌──────────────────────────────────────────────────────────────┐
│                    llm-eval-framework                        │
│                                                              │
│  ┌─────────────┐     ┌──────────────┐      ┌─────────────┐   │
│  │  EvalSample │────▶│   Evaluator  │────▶│  EvalReport │   │
│  │  (input)    │     │ (orchestrate)│      │  (output)   │   │
│  └─────────────┘     └──────┬───────┘      └──────┬──────┘   │
│                             │                     │          │
│                    ┌────────▼────────┐    ┌───────▼───────┐  │
│                    │   BaseMetric    │    │   Reporters   │  │
│                    │   (abstract)    │    │ JSON/Markdown │  │
│                    └────────┬────────┘    └───────────────┘  │
│                             │                                │
│              ┌──────────────┼──────────────┐                 │
│              │              │              │                 │
│     ┌────────▼───┐  ┌───────▼────┐  ┌──────▼─────┐           │
│     │Faithfulness│  │  Answer    │  │  Context   │  ...      │
│     │            │  │ Relevancy  │  │ Precision  │           │
│     └────────────┘  └────────────┘  └────────────┘           │
└──────────────────────────────────────────────────────────────┘
```
 
---
 
## Data Flow
 
A complete evaluation run flows through the system in this order:
 
```
1. INPUT
   User provides List[EvalSample]
   Each sample: {question, answer, contexts, ground_truth}
 
2. VALIDATION
   Evaluator validates each sample against each metric's requirements
   e.g. ContextRecall.requires_ground_truth = True -> fails fast if missing
 
3. SCORING
   For each metric:
     metric.batch_score(samples) -> List[MetricResult]
   Each MetricResult: {score, reasoning, passed, threshold}
 
4. AGGREGATION
   Evaluator assembles SampleEvalResults (all metric scores per sample)
   Then assembles EvalReport (aggregated across all samples)
   EvalReport.compute_aggregates() -> metric averages + overall score
 
5. OUTPUT
   EvalReport passed to Reporter
   JSONReporter → results.json
   MarkdownReporter → results.md (for PR comments, README badges)
```
 
---
 
## Component Responsibilities
 
### `models.py`: Data contracts
 
Defines the data shapes that flow through the entire system. Uses Pydantic for validation. Invalid inputs are rejected at the boundary, not deep inside metric logic.
 
| Model | Responsibility |
|---|---|
| `EvalSample` | Input contract: what every metric receives |
| `MetricResult` | Output of one metric on one sample |
| `SampleEvalResult` | All metric results for one sample |
| `EvalReport` | Final aggregated report for an entire run |
 
**Why Pydantic?**
Pydantic gives us runtime validation, clear error messages, and easy JSON serialization — all essential for a library that needs to fail clearly and integrate with CI/CD pipelines.
 
---
 
### `base.py`: The metric contract
 
Defines `BaseMetric` — the abstract base class every metric must inherit from. The Evaluator only ever interacts with `BaseMetric`, never with concrete metric classes directly.
 
**Key methods:**
 
| Method | Responsibility |
|---|---|
| `score(sample)` | Abstract — must be implemented by every metric |
| `batch_score(samples)` | Default loops over `score()`: override for batching efficiency |
| `validate_sample(sample)` | Checks required fields before scoring |
 
**Why an abstract base class over duck typing?**
Python duck typing would work but it provides no guidance to contributors adding custom metrics. An abstract base class makes the contract explicit, catches missing implementations at class definition time (not at runtime), and makes the codebase self-documenting. Anyone reading `BaseMetric` immediately understands exactly what a metric must do.
 
**Why is `batch_score()` not abstract?**
Most metrics can be implemented as single-sample scorers. Making `batch_score()` abstract would force every metric to implement batching logic, a boilerplate for simple metrics. The default loops over `score()`. Metrics that benefit from batching (e.g. those making embedding API calls) can override it for efficiency.
 
---
