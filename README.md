# llm-eval-framework

I remember watching a youtube video regarding the same evaluation topic, and the teacher said a line which stuck with me like "Mitochondria being the powerhouse of the cell." She said that "You cannot improve something you cannot measure." And that was it, it opened my way of thinking towards evaluation from a developer's perspective. 

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

## What this framework measures
 
There are four fundamental failure modes in a RAG pipeline: **Faithfulness, Answer Relevancy, Context Precision** and **Context Recall**. Each maps to a metric.
 
---
 
### Faithfulness: *did the model make things up?*
 
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

### Answer Relevancy: *did the model actually answer what was asked?*
 
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
Answer relevancy doesn't penalize incomplete answers, meaning, an answer that correctly addresses part of the question but misses half of it can still score well. It measures topical alignment, not completeness. Pair it with context recall to catch this gap.
 
**When to prioritize it:**
When your users are asking complex or multi-part questions. Low answer relevancy often points to a retrieval problem that the right documents aren't being surfaced.
 
---
