from typing import List
from llm_eval.base import BaseMetric
from llm_eval.models import EvalSample, MetricResult
 
 
class AnswerRelevancy(BaseMetric):
    """
    Measures how well the answer addresses the original question.
 
    How it works: 2 steps per sample:
 
    Step 1 — Reverse question generation:
        Given the answer, ask the LLM: "What question would
        this answer be responding to?" Generate N reverse
        questions from the answer.
 
        Why reverse questions? Because if an answer is truly
        relevant to the question, then questions generated
        FROM that answer should closely resemble the original
        question. If the answer drifted off-topic, the reverse
        questions will look different from the original.
 
    Step 2 — Embedding similarity:
        Compute cosine similarity between each reverse question
        and the original question using embeddings.
        Score = mean similarity across all reverse questions.
 
    Why this matters:
        An answer can be perfectly faithful (grounded in context)
        but still not answer what the user actually asked.
        Answer relevancy catches this failure mode — the
        "technically correct but useless" answer.
 
    Args:
        llm_client      : OpenAI or Azure OpenAI client instance
        embedder        : Embedding model with embed_query() method
        model           : Model for reverse question generation
        n_generations   : Number of reverse questions to generate (default: 3)
        threshold       : Minimum acceptable score (default: 0.7)
    """
 
    name = "answer_relevancy"
    requires_ground_truth = False
 
    def __init__(
        self,
        llm_client=None,
        embedder=None,
        model: str = "gpt-4",
        n_generations: int = 3,
        threshold: float = 0.7
    ):
        self.llm_client = llm_client
        self.embedder = embedder
        self.model = model
        self.n_generations = n_generations
        self.threshold = threshold
 
    def score(self, sample: EvalSample) -> MetricResult:
        """
        Score answer relevancy for a single EvalSample.
 
        Args:
            sample: EvalSample with question and answer.
 
        Returns:
            MetricResult with answer relevancy score.
        """
        self.validate_sample(sample)
 
        # Step 1 — generate reverse questions from the answer
        reverse_questions = self._generate_reverse_questions(
            sample.answer,
            sample.question
        )
 
        if not reverse_questions:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                reasoning="Could not generate reverse questions from answer.",
                threshold=self.threshold
            )
 
        # Step 2 — compute embedding similarity
        similarity_score = self._compute_similarity(
            sample.question,
            reverse_questions
        )
 
        reasoning = (
            f"Mean cosine similarity between original question and "
            f"{len(reverse_questions)} reverse-generated questions: "
            f"{similarity_score:.4f}\n"
            f"Reverse questions: {reverse_questions}"
        )
 
        return MetricResult(
            metric_name=self.name,
            score=round(similarity_score, 4),
            reasoning=reasoning,
            threshold=self.threshold,
            metadata={
                "reverse_questions": reverse_questions,
                "n_generations": self.n_generations,
                "original_question": sample.question
            }
        )
 
    def _generate_reverse_questions(
        self,
        answer: str,
        original_question: str
    ) -> List[str]:
        """
        Generate N questions that the given answer could be responding to.
 
        The original question is provided as context to help the LLM
        understand the domain — but the reverse questions should be
        generated from the answer alone.
        """
        prompt = f"""Given the following answer, generate {self.n_generations} 
different questions that this answer could be responding to.
Return one question per line. No numbering, no bullets.
 
Answer: {answer}
 
Questions:"""
 
        response = self._call_llm(prompt)
        if not response:
            return []
 
        questions = [
            line.strip()
            for line in response.strip().split("\n")
            if line.strip() and "?" in line
        ]
        return questions[:self.n_generations]
 
    def _compute_similarity(
        self,
        original_question: str,
        reverse_questions: List[str]
    ) -> float:
        """
        Compute mean cosine similarity between original question
        and reverse-generated questions using embeddings.
 
        Falls back to keyword overlap similarity if no embedder
        is configured — useful for testing without API keys.
        """
        if self.embedder is None:
            # Fallback — keyword overlap similarity for testing
            return self._keyword_similarity(original_question, reverse_questions)
 
        try:
            import numpy as np
 
            orig_embedding = self.embedder.embed_query(original_question)
            reverse_embeddings = [
                self.embedder.embed_query(q) for q in reverse_questions
            ]
 
            orig_vec = np.array(orig_embedding)
            similarities = []
 
            for rev_emb in reverse_embeddings:
                rev_vec = np.array(rev_emb)
                # Cosine similarity
                similarity = np.dot(orig_vec, rev_vec) / (
                    np.linalg.norm(orig_vec) * np.linalg.norm(rev_vec) + 1e-8
                )
                similarities.append(float(similarity))
 
            return sum(similarities) / len(similarities)
 
        except Exception as e:
            print(f"[{self.name}] Embedding similarity failed: {e}")
            return self._keyword_similarity(original_question, reverse_questions)
 
    def _keyword_similarity(
        self,
        original: str,
        reverse_questions: List[str]
    ) -> float:
        """
        Simple keyword overlap similarity — fallback when no embedder.
 
        Computes Jaccard similarity between original and each
        reverse question, returns the mean.
        """
        orig_words = set(original.lower().split())
        similarities = []
 
        for rq in reverse_questions:
            rq_words = set(rq.lower().split())
            intersection = orig_words & rq_words
            union = orig_words | rq_words
            if union:
                similarities.append(len(intersection) / len(union))
            else:
                similarities.append(0.0)
 
        return sum(similarities) / len(similarities) if similarities else 0.0
 
    def _call_llm(self, prompt: str) -> str:
        """LLM call with mock fallback for testing."""
        if self.llm_client is None:
            # Mock response — returns a plausible reverse question
            return "What is the return policy?\nHow many days can I return items?\nWhat are the conditions for returns?"
 
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[{self.name}] LLM call failed: {e}")
            return ""
