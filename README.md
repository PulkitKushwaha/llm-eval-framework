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
