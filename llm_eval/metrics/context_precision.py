from typing import List
from llm_eval.base import BaseMetric
from llm_eval.models import EvalSample, MetricResult
 
 
class ContextPrecision(BaseMetric):
    """
    Measures what fraction of retrieved chunks are actually
    relevant to the question.
 
    How it works: 1 LLM call per retrieved chunk:
 
    For each retrieved chunk, ask the LLM:
        "Is this chunk useful for answering the question?"
        → YES or NO
 
    Score = weighted mean of relevance scores, where chunks
    ranked higher (retrieved first) carry more weight.
 
    Why position-weighted?
        Most retrievers return chunks in order of similarity.
        A chunk ranked 1st that is irrelevant is a bigger
        problem than an irrelevant chunk ranked 5th. The
        weighted scoring reflects this — precision failures
        at the top of the ranked list are penalized more.
 
    Score formula:
        For each position k where the chunk is relevant:
            precision_at_k = (relevant chunks up to k) / k
        Score = mean(precision_at_k for all relevant positions)
        If no chunks are relevant, score = 0.0
 
    Why this matters:
        Low context precision means your retriever is flooding
        the context window with noise. This increases token
        costs, confuses the LLM, and leads to hallucination
        or irrelevant answers — even when the right chunk
        exists in your knowledge base.
 
    Common causes of low context precision:
        - Embedding model not capturing semantic meaning well
        - Chunk size too large: relevant content diluted
        - top-k too high: over-retrieval of marginal chunks
        - Query and document language mismatch
 
    Args:
        llm_client  : OpenAI or Azure OpenAI client instance
        model       : Model name for relevance scoring
        threshold   : Minimum acceptable score (default: 0.7)
    """
 
    name = "context_precision"
    requires_ground_truth = False
 
    def __init__(
        self,
        llm_client=None,
        model: str = "gpt-4",
        threshold: float = 0.7
    ):
        self.llm_client = llm_client
        self.model = model
        self.threshold = threshold
 
    def score(self, sample: EvalSample) -> MetricResult:
        """
        Score context precision for a single EvalSample.
 
        Args:
            sample: EvalSample with question and contexts.
 
        Returns:
            MetricResult with context precision score.
        """
        self.validate_sample(sample)
 
        # Score each chunk for relevance
        relevance_scores = self._score_chunks(
            question=sample.question,
            chunks=sample.contexts
        )
 
        # Compute position-weighted precision
        precision_score = self._compute_weighted_precision(relevance_scores)
 
        relevant_count = sum(relevance_scores)
        total_count = len(relevance_scores)
 
        reasoning = (
            f"{relevant_count}/{total_count} retrieved chunks are relevant.\n"
            f"Position-weighted precision: {precision_score:.4f}\n"
            f"Chunk relevance (1=relevant, 0=irrelevant): {relevance_scores}"
        )
 
        return MetricResult(
            metric_name=self.name,
            score=round(precision_score, 4),
            reasoning=reasoning,
            threshold=self.threshold,
            metadata={
                "total_chunks": total_count,
                "relevant_chunks": relevant_count,
                "relevance_scores": relevance_scores
            }
        )
 
    def _score_chunks(
        self,
        question: str,
        chunks: List[str]
    ) -> List[int]:
        """
        Score each chunk for relevance to the question.
 
        Returns a binary list — 1 if relevant, 0 if not.
        Order preserved — matches input chunk order.
        """
        relevance_scores = []
 
        for i, chunk in enumerate(chunks):
            prompt = f"""Is the following context chunk useful for answering the question?
Answer with YES or NO only.
 
Question: {question}
 
Context chunk:
{chunk}
 
Answer:"""
 
            response = self._call_llm(prompt)
            is_relevant = 1 if (response and "YES" in response.upper()) else 0
            relevance_scores.append(is_relevant)
 
        return relevance_scores
 
    def _compute_weighted_precision(
        self,
        relevance_scores: List[int]
    ) -> float:
        """
        Compute position-weighted average precision.
 
        For each position where a chunk is relevant,
        compute precision@k (relevant chunks seen so far / k).
        Average precision = mean of precision@k at relevant positions.
 
        This is the standard Average Precision (AP) metric from
        information retrieval, adapted for RAG evaluation.
        """
        if not relevance_scores or sum(relevance_scores) == 0:
            return 0.0
 
        precision_at_k_values = []
        relevant_so_far = 0
 
        for k, is_relevant in enumerate(relevance_scores, start=1):
            if is_relevant:
                relevant_so_far += 1
                precision_at_k = relevant_so_far / k
                precision_at_k_values.append(precision_at_k)
 
        if not precision_at_k_values:
            return 0.0
 
        return sum(precision_at_k_values) / len(precision_at_k_values)
 
    def _call_llm(self, prompt: str) -> str:
        """LLM call with mock fallback for testing."""
        if self.llm_client is None:
            return "YES"
 
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[{self.name}] LLM call failed: {e}")
            return ""
