# Failure Analysis — RAG Pipeline Evaluation
 
A root cause analysis of the failures identified across both
evaluation runs (baseline and optimized). For each failure pattern,
this document identifies whether the root cause is a retrieval problem,
a generation problem, or a system design problem, and provides
concrete recommendations.
 
---
 
## How to read this document
 
Every failure in a RAG pipeline belongs to one of three layers:
 
**Retrieval failures**: the right information was in the knowledge
base but the retriever didn't find it. Fix: chunking strategy,
embedding model, retrieval algorithm, top-k.
 
**Generation failures**: the retriever found the right information
but the LLM generated an incorrect or unfaithful answer. Fix: prompt
engineering, system prompt hardening, output validation.
 
**System design failures**: the problem requires a capability the
pipeline doesn't have. Fix: guardrails, query expansion, multi-hop
retrieval, agentic reasoning.
 
Knowing which layer failed is essential. Retrieval fixes can't solve
generation problems, and neither can solve system design problems.
 
---
 
## Failure Pattern 1: Multi-hop context fragmentation
 
**Affected categories:** multi_hop (avg recall: 0.49 baseline, 0.72 optimized)
**Metric:** Context Recall
**Layer:** Retrieval
 
### What happened
 
Multi-hop questions require information from multiple document sections.
Example: "If I order today with express shipping, when will it arrive
and what's the return policy?"
 
This requires retrieving:
1. Express shipping timeframe (from shipping section)
2. Return policy window (from returns section)
The recursive chunker split related policy information across chunk
boundaries. A single retrieval call with k=5 was insufficient to
surface chunks from both sections simultaneously.
 
### Evidence
 
Question q006 scored 0.33 context recall in baseline:
- Retrieved 5 chunks, all from the shipping section
- Return policy chunks not retrieved
- LLM correctly answered the shipping part, fabricated the return part
After switching to sentence-window chunking (window=2):
- Related sentences kept together in the same chunk
- Both sections now retrievable with k=5
- q006 recall improved to 0.81
### Root cause
 
**Chunking boundary problem.** The recursive chunker split at paragraph
boundaries, placing shipping policy in one chunk and return policy in
the adjacent chunk. When the query embedding was closer to shipping
language, only shipping chunks were retrieved.
 
### Recommendation
 
**Implemented:** Switch to sentence-window chunking (window=2)
**Implemented:** Increase retrieval k to 20, rerank to 5
 
**Additional fix:** Add query decomposition for multi-hop queries.
Before retrieval, detect whether the query requires multiple pieces
of information and issue separate retrieval calls for each component.
This is an agentic capability, you can check `multi-agent-system` repo for your reference.
 
---
 
## Failure Pattern 2: Adversarial false premise handling
 
**Affected categories:** adversarial (avg faithfulness: 0.62 baseline, 0.72 optimized)
**Metric:** Faithfulness
**Layer:** System design (not retrieval)
 
### What happened
 
Questions with false premises caused the LLM to partially affirm
incorrect information even when the retrieved context contradicted it.
 
Example: "The website says free shipping on all orders. Is that right?"
 
The retrieved context correctly stated: "Free standard shipping applies
to orders over $50 for Gold members only."
 
Despite this, the LLM response pattern was: "While free shipping is
available [partial affirmation], it applies to Gold members for orders
over $50." The partial affirmation of "while free shipping is available"
was scored as unfaithful as the context says it's NOT available to all.
 
### Evidence
 
q021 (adversarial, false shipping premise) scored 0.40 faithfulness
in baseline and 0.55 in optimized. The retrieval was correct in both
runs as the failure is entirely in generation.
 
### Root cause
 
**False premise contamination.** The LLM's RLHF training makes it
inclined to find something correct in the user's statement before
correcting it. This "yes, but..." pattern is conversationally natural
but factually unfaithful when the "yes" part is false.
 
### Recommendation
 
**Requires input guardrails:** Detect false premises before retrieval
and add an explicit instruction to the system prompt:
 
```python
# Add to system prompt for adversarial-robust pipelines
FALSE_PREMISE_INSTRUCTION = """
If the user's question contains a factual claim that contradicts
the retrieved context, CORRECT the claim directly before answering.
Do not partially affirm incorrect claims. Start with the correction.
Example: "Actually, [correct fact]. [Answer to the real question]."
"""
```
 
**Requires guardrails layer:** Add a pre-retrieval false premise
detector. If a claim in the query contradicts known facts, flag it
and add explicit correction instruction to the prompt.
See `llm-guardrails` repo for implementation.
 
---
 
## Failure Pattern 3: Ambiguous query under-retrieval
 
**Affected categories:** ambiguous (avg relevancy: 0.65 baseline, 0.76 optimized)
**Metric:** Answer Relevancy
**Layer:** System design
 
### What happened
 
Underspecified queries caused the retriever to commit to one
interpretation and retrieve only for that interpretation.
 
Example: "How long does it take?" could mean:
- Shipping time (5-7 days standard)
- Return processing time (5-7 business days)
- Order processing time (1-2 days before shipping)
- Refund processing time (5-7 business days)
The embedding of "how long does it take" was closest to shipping
content — so only shipping timeframes were retrieved. The answer
addressed shipping but not the other interpretations.
 
### Evidence
 
q026 scored 0.42 answer relevancy in baseline and 0.59 in optimized.
HyDE improved this slightly as the hypothetical answer generation
sometimes picked a different interpretation, but did not fully solve it.
 
### Root cause
 
**Query ambiguity (not retrievable).** This cannot be solved by better
retrieval because the query genuinely has multiple valid interpretations.
The retriever must pick one. The correct solution is to either clarify
the query before retrieval or retrieve for all interpretations.
 
### Recommendation
 
**Query expansion:** Before retrieval, generate N interpretations
of ambiguous queries and issue a retrieval call for each. Merge results.
 
```python
# Ambiguity detection heuristic
AMBIGUOUS_QUERIES = ["how long", "what is the fee", "what are the options"]
 
def is_ambiguous(query: str) -> bool:
    return any(phrase in query.lower() for phrase in AMBIGUOUS_QUERIES)
 
# If ambiguous, ask clarifying question or expand to multiple queries
```
 
**Clarification agent:** Add a pre-retrieval clarification step
that asks the user to specify when the query is ambiguous.
This is an agentic capability — see `multi-agent-system` repo.
 
---
 
## Failure Pattern 4: Out-of-scope query hallucination
 
**Affected categories:** out_of_scope (avg faithfulness: 0.79 baseline, 0.84 optimized)
**Metric:** Faithfulness
**Layer:** Generation
 
### What happened
 
For questions outside the knowledge base scope (q024: "capital of France",
q025: "recommend a restaurant"), the pipeline retrieved the least-irrelevant
chunks available and the LLM generated a response using them, sometimes
hallucinating to fill gaps.
 
Expected behavior: "I can only answer questions about our products,
shipping, and return policies."
 
Actual behavior (baseline): Retrieved the FAQ introduction section and
generated a partially coherent but off-topic response.
 
### Evidence
 
Out-of-scope categories scored 0.69 context recall in baseline, meaning
chunks were retrieved (even though none were relevant). The retriever
has no concept of "I don't know", it always returns k results.
 
### Root cause
 
**Missing scope enforcement.** The system prompt did not explicitly
tell the LLM to refuse out-of-scope questions. The retriever always
returns results regardless of relevance.
 
### Recommendation
 
**Partially fixed:** System prompt now includes scope instructions
in optimized pipeline.
 
**Add relevance threshold:** If the top retrieved chunk scores below
a similarity threshold, return a "no relevant information found"
signal instead of low-quality chunks.
 
```python
def retrieve_with_threshold(
    query: str,
    min_score: float = 0.5,
    k: int = 5
) -> tuple:
    results = vector_store.search(query_embedding, k=k)
    top_score = results[0][1] if results else 0.0
 
    if top_score < min_score:
        return [], False  # No relevant content found
 
    return [chunk for chunk, _ in results], True
```
 
**Add topic scope validator:** See `llm-guardrails` repo for a
topic relevance validator that blocks out-of-scope queries before
they reach the retriever.
 
---
 
## Failure Pattern 5: Technical query language mismatch
 
**Affected categories:** technical (avg recall: 0.57 baseline, 0.73 optimized)
**Metric:** Context Recall
**Layer:** Retrieval
 
### What happened
 
Technical queries used domain terminology that differed from document
language. Example: q028 ("What encryption is used for payment processing?")
retrieved generic security chunks rather than the specific TLS/PCI-DSS
documentation because "encryption" and "TLS 1.3" don't share embedding space
without context.
 
### Evidence
 
Technical category improved most from HyDE (+16% recall) because HyDE
generates a hypothetical answer that uses the correct technical language,
bridging the query-document vocabulary gap.
 
### Root cause
 
**Embedding vocabulary gap.** The query uses general terminology
("encryption") while documents use specific terminology ("TLS 1.3",
"PCI-DSS Level 1"). Standard embedding similarity underperforms
when vocabulary doesn't overlap.
 
### Recommendation
 
**Implemented:** HyDE retrieval significantly improves technical
query recall by generating vocabulary-matched hypotheses.
 
**Domain fine-tuning:** Fine-tune the embedding model on domain
documents to align vocabulary. High cost, high impact for specialized
domains (medical, legal, financial).
 
**Query expansion with synonyms:** Expand technical queries with
related terminology before retrieval.
 
---
 
## Summary table
 
| Failure pattern | Layer | Baseline | Optimized | Fix implemented | Remaining fix |
|---|---|---|---|---|---|
| Multi-hop fragmentation | Retrieval | 0.49 | 0.72 | ✅ Sentence-window + reranking | Query decomposition (agentic) |
| False premise handling | System design | 0.62 | 0.72 | ⚠️ Partial (prompt) | Input guardrails |
| Ambiguous under-retrieval | System design | 0.65 | 0.76 | ⚠️ Partial (HyDE) | Query expansion / clarification |
| Out-of-scope hallucination | Generation | 0.69 | 0.84 | ✅ System prompt | Relevance threshold |
| Technical vocab mismatch | Retrieval | 0.57 | 0.73 | ✅ HyDE | Domain embedding fine-tuning |
 
---
 
## Key takeaway
 
**Retrieval problems are fixable with better retrieval.**
Chunking strategy and retrieval algorithm changes resolved the
multi-hop and technical query failures significantly.
 
**System design problems require new capabilities.**
False premise handling and ambiguous queries cannot be solved
by retrieval optimization, they require guardrails and agentic
reasoning. These are addressed in:
- `llm-guardrails` → input validation and false premise detection
- `multi-agent-system` → query decomposition and clarification
**Measure before you optimize.**
Without the evaluation framework, we would not have known that
context recall was the bottleneck, and might have optimized
generation instead. Evaluation-driven optimization is faster
and more effective than intuition-driven optimization.
 
---
 
*Evaluated using [llm-eval-framework](https://github.com/pulkitkushwaha/llm-eval-framework)*
*See also: [llm-guardrails](https://github.com/pulkitkushwaha/llm-guardrails) |
[multi-agent-system](https://github.com/pulkitkushwaha/multi-agent-system)*
