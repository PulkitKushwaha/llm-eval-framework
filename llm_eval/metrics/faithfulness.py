from typing import List
from llm_eval.base import BaseMetric
from llm_eval.models import EvalSample, MetricResult
 
 
class Faithfulness(BaseMetric):
    """
    Measures whether all claims in the answer are supported
    by the retrieved context.
 
    How it works: 2 LLM calls per sample:
 
    Step 1 — Claim decomposition:
        The answer is broken into atomic claims — individual
        factual statements that can be verified independently.
        Example answer: "The return policy allows 30 days.
        Items must be unused."
        → Claim 1: "The return policy allows 30 days"
        → Claim 2: "Items must be unused"
 
    Step 2 — Claim verification:
        Each claim is checked against the retrieved context.
        The LLM answers YES or NO for each claim.
        Score = supported claims / total claims
 
    Why this matters:
        A faithfulness score below 0.7 means your pipeline
        is hallucinating — generating content not grounded
        in retrieved context. This is the most dangerous
        failure mode in production RAG systems.
 
    Args:
        llm_client  : OpenAI or Azure OpenAI client instance
        model       : Model name to use for scoring (default: gpt-4)
        threshold   : Minimum acceptable score (default: 0.7)
    """
 
    name = "faithfulness"
    requires_ground_truth = False
 
    def __init__(self, llm_client=None, model: str = "gpt-4", threshold: float = 0.7):
        self.llm_client = llm_client
        self.model = model
        self.threshold = threshold
 
    def score(self, sample: EvalSample) -> MetricResult:
        """
        Score faithfulness for a single EvalSample.
 
        Args:
            sample: EvalSample with question, answer, and contexts.
 
        Returns:
            MetricResult with faithfulness score and reasoning.
        """
        self.validate_sample(sample)
 
        # Step 1 — decompose answer into atomic claims
        claims = self._decompose_into_claims(sample.answer)
 
        if not claims:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                reasoning="Could not decompose answer into verifiable claims.",
                threshold=self.threshold
            )
 
        # Step 2 — verify each claim against retrieved context
        context_text = "\n\n".join(sample.contexts)
        supported = self._verify_claims(claims, context_text)
 
        score = supported / len(claims)
        reasoning = (
            f"{supported}/{len(claims)} claims supported by context.\n"
            f"Claims: {claims}"
        )
 
        return MetricResult(
            metric_name=self.name,
            score=round(score, 4),
            reasoning=reasoning,
            threshold=self.threshold,
            metadata={
                "total_claims": len(claims),
                "supported_claims": supported,
                "claims": claims
            }
        )
 
    def _decompose_into_claims(self, answer: str) -> List[str]:
        """
        Use LLM to decompose answer into atomic factual claims.
 
        Each claim should be a single verifiable statement —
        something that can be answered YES or NO when checked
        against a context passage.
        """
        prompt = f"""Break the following answer into individual atomic claims.
Each claim should be a single verifiable factual statement.
Return one claim per line. No numbering, no bullets.
 
Answer: {answer}
 
Claims:"""
 
        response = self._call_llm(prompt)
        if not response:
            return []
 
        claims = [
            line.strip()
            for line in response.strip().split("\n")
            if line.strip()
        ]
        return claims
 
    def _verify_claims(self, claims: List[str], context: str) -> int:
        """
        Verify each claim against the retrieved context.
 
        Returns the count of claims supported by the context.
        """
        supported = 0
 
        for claim in claims:
            prompt = f"""Given the following context, is the claim supported?
Answer with YES or NO only.
 
Context:
{context}
 
Claim: {claim}
 
Answer:"""
 
            response = self._call_llm(prompt)
            if response and "YES" in response.upper():
                supported += 1
 
        return supported
 
    def _call_llm(self, prompt: str) -> str:
        """
        Make an LLM call.
 
        Falls back to a mock response if no client is configured —
        useful for testing without API keys.
        """
        if self.llm_client is None:
            # Mock response for testing — replace with real client in production
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
