from typing import List
from llm_eval.base import BaseMetric
from llm_eval.models import EvalSample, MetricResult
 
 
class ContextRecall(BaseMetric):
    """
    Measures whether all information needed to answer the
    question correctly was present in the retrieved chunks.
 
    This is the only core metric that requires ground_truth.
    Without knowing what the correct answer should contain,
    we cannot measure whether the retriever found everything
    it needed to.
 
    How it works: 2 steps per sample:
 
    Step 1 — Ground truth decomposition:
        The ground truth answer is broken into atomic statements —
        individual pieces of information that must be present
        for a complete answer.
 
        Example ground truth: "Items can be returned within
        30 days. They must be unused and in original packaging."
        → Statement 1: "Items can be returned within 30 days"
        → Statement 2: "Items must be unused"
        → Statement 3: "Items must be in original packaging"
 
    Step 2 — Statement attribution:
        For each statement, check whether it can be found in
        the retrieved context. The LLM answers YES or NO.
        Score = statements found in context / total statements
 
    Why this matters:
        Context recall tells you whether your retriever is
        missing important chunks. A score of 0.6 means 40%
        of the information needed to answer correctly was
        not retrieved — even though it may exist in your
        knowledge base. This points to chunking or embedding
        problems, not generation problems.
 
    Context recall vs Context precision:
        Precision  → are the chunks you retrieved relevant?
        Recall     → did you retrieve all the relevant chunks?
        You need both. High precision + low recall means your
        retriever is picky but misses things. High recall +
        low precision means it finds everything but drowns
        the LLM in noise.
 
    Common causes of low context recall:
        - Chunk size too small — relevant info split across
          chunk boundaries and only one side retrieved
        - top-k too low — not retrieving enough chunks
        - Embedding model missing semantic connections
        - Poor document structure — related content scattered
 
    Args:
        llm_client  : OpenAI or Azure OpenAI client instance
        model       : Model name for statement attribution
        threshold   : Minimum acceptable score (default: 0.7)
    """
 
    name = "context_recall"
    requires_ground_truth = True
 
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
        Score context recall for a single EvalSample.
 
        Args:
            sample: EvalSample with question, contexts, and ground_truth.
                    ground_truth is REQUIRED for this metric.
 
        Returns:
            MetricResult with context recall score.
 
        Raises:
            ValueError: If ground_truth is not provided.
        """
        self.validate_sample(sample)
 
        # Step 1 — decompose ground truth into atomic statements
        statements = self._decompose_ground_truth(sample.ground_truth)
 
        if not statements:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                reasoning="Could not decompose ground truth into statements.",
                threshold=self.threshold
            )
 
        # Step 2 — check each statement against retrieved context
        context_text = "\n\n".join(sample.contexts)
        found_count = self._attribute_statements(statements, context_text)
 
        score = found_count / len(statements)
 
        reasoning = (
            f"{found_count}/{len(statements)} ground truth statements "
            f"found in retrieved context.\n"
            f"Statements: {statements}"
        )
 
        return MetricResult(
            metric_name=self.name,
            score=round(score, 4),
            reasoning=reasoning,
            threshold=self.threshold,
            metadata={
                "total_statements": len(statements),
                "found_in_context": found_count,
                "missing_from_context": len(statements) - found_count,
                "statements": statements
            }
        )
 
    def _decompose_ground_truth(self, ground_truth: str) -> List[str]:
        """
        Decompose ground truth answer into atomic statements.
 
        Each statement should be a single piece of information
        that can be independently verified against the context.
        """
        prompt = f"""Break the following answer into individual atomic statements.
Each statement should be a single piece of information.
Return one statement per line. No numbering, no bullets.
 
Answer: {ground_truth}
 
Statements:"""
 
        response = self._call_llm(prompt)
        if not response:
            return []
 
        statements = [
            line.strip()
            for line in response.strip().split("\n")
            if line.strip()
        ]
        return statements
 
    def _attribute_statements(
        self,
        statements: List[str],
        context: str
    ) -> int:
        """
        Check how many statements can be attributed to
        the retrieved context.
 
        Returns the count of statements found in context.
        """
        found = 0
 
        for statement in statements:
            prompt = f"""Can the following statement be directly attributed
to the context below? Answer YES or NO only.
 
Context:
{context}
 
Statement: {statement}
 
Answer:"""
 
            response = self._call_llm(prompt)
            if response and "YES" in response.upper():
                found += 1
 
        return found
 
    def _call_llm(self, prompt: str) -> str:
        """LLM call with mock fallback for testing."""
        if self.llm_client is None:
            return "YES"
 
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[{self.name}] LLM call failed: {e}")
            return ""
