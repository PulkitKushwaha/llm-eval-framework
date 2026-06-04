# Before vs After — RAG Pipeline Optimization
 
Comparing baseline (recursive chunking + similarity retrieval) against
optimized pipeline (sentence-window chunking + HyDE + cross-encoder reranking).
 
---
 
## Score comparison
 
| Metric | Baseline | Optimized | Delta | Improvement |
|---|---|---|---|---|
| Faithfulness | `0.7823` | `0.8534` | +0.0711 | +9.1% ✅ |
| Answer Relevancy | `0.7654` | `0.8312` | +0.0658 | +8.6% ✅ |
| Context Precision | `0.6891` | `0.7989` | +0.1098 | +15.9% ✅ |
| Context Recall | `0.6201` | `0.7789` | +0.1588 | +25.6% ✅ |
| **Overall** | **`0.7142`** | **`0.8156`** | **+0.1014** | **+14.2% ✅** |
 
**All 4 metrics now pass the 0.7 threshold.** Baseline had 2 failing.
 
---
 
## Per-category improvement
 
| Category | Baseline Overall | Optimized Overall | Delta |
|---|---|---|---|
| factual | 0.8680 | 0.9206 | +0.0526 |
| summarization | 0.7509 | 0.8329 | +0.0820 |
| negative | 0.7270 | 0.8234 | +0.0964 |
| multi_hop | 0.6265 | 0.7687 | +0.1422 |
| comparison | 0.6733 | 0.7928 | +0.1195 |
| technical | 0.6651 | 0.7817 | +0.1166 |
| out_of_scope | 0.6951 | 0.7981 | +0.1030 |
| adversarial | 0.5762 | 0.6873 | +0.1111 |
| ambiguous | 0.5948 | 0.7342 | +0.1394 |
 
**Biggest gains in the hardest categories**: multi-hop (+14.2%) and
ambiguous (+13.9%) improved most. These were the weakest in baseline.
 
---
 
## What drove the improvements
 
### Context Recall +25.6%: sentence-window chunking
 
Baseline recall failures were almost entirely caused by the recursive
chunker splitting related content across chunk boundaries. Switching to
sentence-window chunking (window=2) keeps each sentence in context with
its neighbors. Multi-hop queries that previously missed half the required
information now retrieve it consistently.
 
**Specific example:** q006 (express shipping + return policy) improved from
0.33 to 0.81 recall. The sentence window kept shipping policy sentences
adjacent to return policy sentences: the retriever found both in the same
chunk instead of missing one.
 
### Context Precision +15.9%: cross-encoder reranking
 
Baseline retrieved 5 chunks via cosine similarity, several were
semantically adjacent but not truly relevant. Cross-encoder reranking
scores query-chunk pairs together, promoting genuinely relevant chunks
and demoting noise. Over-fetching 20 candidates then reranking to 5
gives the reranker good material to work with.
 
### Answer Relevancy +8.6%: HyDE retrieval
 
HyDE generates a hypothetical answer and uses its embedding for search
instead of the raw query. This bridges the language gap between conversational
queries and formal document language. Particularly effective for technical
and comparison queries where the user's phrasing differs from the document.
 
### Faithfulness +9.1%: downstream of better retrieval
 
Faithfulness improved as a downstream effect of better retrieval.
When the retriever finds the right chunks, the LLM has the right
information to generate faithful answers. Fewer hallucinations occur
when the context is complete and relevant.
 
---
 
## What didn't fully improve
 
**Adversarial queries (0.69)**: still below 0.7. False premise handling
requires input-layer guardrails, not retrieval improvements. The retriever
now finds correct information, but the LLM still partially affirms false
premises in the question. Solution: add a false-premise detection layer
before retrieval (see llm-guardrails repo).
 
**Ambiguous queries (0.73)**: improved but marginal. HyDE picks one
interpretation of the ambiguous query and retrieves for that interpretation.
A proper fix requires query expansion or clarification before retrieval.
 
---
 
## Key lessons
 
1. **Context recall is the most impactful metric to improve.** A 25%
   recall improvement cascades into faithfulness, relevancy, and precision
   gains. Fix retrieval before fixing generation.
2. **Chunking strategy matters more than retrieval strategy.** Switching
   from recursive to sentence-window chunking drove more improvement than
   adding HyDE. Get chunking right first.
3. **Reranking is high-value, low-effort.** Adding cross-encoder reranking
   to an existing pipeline requires minimal code changes but delivers
   consistent precision improvements across all query types.
4. **Some failures are not retrieval problems.** Adversarial and ambiguous
   queries need upstream fixes — guardrails and query understanding — that
   no retrieval optimization can solve.
---
 
## Recommended next steps
 
| Priority | Action | Expected gain |
|---|---|---|
| High | Add false-premise detection to input layer | Adversarial +10% |
| High | Add query expansion for ambiguous queries | Ambiguous +8% |
| Medium | Increase window_size to 3 for multi-hop | Multi-hop +5% |
| Medium | Add HyDE multi-hypothesis (n=3) | Technical +3% |
| Low | Fine-tune reranker on domain data | Precision +2% |
 
---
 
*Evaluated using [llm-eval-framework](https://github.com/pulkitkushwaha/llm-eval-framework)*
