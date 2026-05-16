from typing import List, Optional, Dict, Any
from llm_eval.base import BaseMetric
from llm_eval.models import EvalSample, MetricResult
 
 
class RAGASMetric(BaseMetric):
    """
    Thin wrapper that runs any RAGAS metric through the
    llm-eval-framework interface.
 
    Why wrap RAGAS instead of just using it directly?
 
    RAGAS is excellent. It is well-documented, widely benchmarked,
    and actively maintained. But it has its own data format,
    its own runner, and its own output structure. Wrapping it
    means you can run RAGAS metrics alongside custom metrics
    in the same Evaluator run, get results in the same
    EvalReport format, and feed them into the same reporters
    and CI/CD pipeline.
 
    This is the integration layer, not a replacement.
 
    Supported RAGAS metrics:
        - faithfulness
        - answer_relevancy
        - context_precision
        - context_recall
        - answer_correctness
        - answer_similarity
 
    Usage:
        from llm_eval.metrics import RAGASMetric
 
        # Run RAGAS faithfulness alongside custom metrics
        evaluator = Evaluator(metrics=[
            Faithfulness(),              # our custom implementation
            RAGASMetric("faithfulness"), # RAGAS implementation for comparison
        ])
 
    Args:
        ragas_metric_name : Name of the RAGAS metric to wrap
        llm               : LangChain LLM instance for RAGAS
        embeddings        : LangChain embeddings instance for RAGAS
        threshold         : Minimum acceptable score (default: 0.7)
    """
 
    requires_ground_truth = False  # set per metric in __init__
 
    RAGAS_METRICS_REQUIRING_GT = {
        "context_recall",
        "answer_correctness",
        "answer_similarity"
    }
 
    SUPPORTED_METRICS = {
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "answer_correctness",
        "answer_similarity"
    }
 
    def __init__(
        self,
        ragas_metric_name: str,
        llm=None,
        embeddings=None,
        threshold: float = 0.7
    ):
        if ragas_metric_name not in self.SUPPORTED_METRICS:
            raise ValueError(
                f"Unsupported RAGAS metric: '{ragas_metric_name}'. "
                f"Supported: {', '.join(sorted(self.SUPPORTED_METRICS))}"
            )
 
        self.ragas_metric_name = ragas_metric_name
        self.name = f"ragas_{ragas_metric_name}"
        self.llm = llm
        self.embeddings = embeddings
        self.threshold = threshold
        self.requires_ground_truth = (
            ragas_metric_name in self.RAGAS_METRICS_REQUIRING_GT
        )
        self._ragas_metric = None
 
    def score(self, sample: EvalSample) -> MetricResult:
        """
        Score a single EvalSample using the wrapped RAGAS metric.
 
        Converts EvalSample to RAGAS dataset format, runs the
        metric, and converts the result back to MetricResult.
 
        Args:
            sample: EvalSample with required fields for this metric.
 
        Returns:
            MetricResult with RAGAS score.
        """
        self.validate_sample(sample)
 
        try:
            ragas_score = self._run_ragas(sample)
        except ImportError:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                reasoning=(
                    "RAGAS is not installed. "
                    "Install with: pip install ragas"
                ),
                threshold=self.threshold
            )
        except Exception as e:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                reasoning=f"RAGAS scoring failed: {str(e)}",
                threshold=self.threshold
            )
 
        return MetricResult(
            metric_name=self.name,
            score=round(float(ragas_score), 4),
            reasoning=f"RAGAS {self.ragas_metric_name} score: {ragas_score:.4f}",
            threshold=self.threshold,
            metadata={
                "ragas_metric": self.ragas_metric_name,
                "raw_score": ragas_score
            }
        )
 
    def _run_ragas(self, sample: EvalSample) -> float:
        """
        Convert EvalSample to RAGAS format and run the metric.
 
        RAGAS expects a Dataset with columns:
            question, answer, contexts, ground_truth (optional)
        """
        from datasets import Dataset
        import ragas
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness,
            answer_similarity
        )
 
        # Map metric name to RAGAS metric object
        metric_map = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "answer_correctness": answer_correctness,
            "answer_similarity": answer_similarity
        }
 
        ragas_metric = metric_map[self.ragas_metric_name]
 
        # Build RAGAS dataset from single sample
        data = {
            "question": [sample.question],
            "answer": [sample.answer],
            "contexts": [sample.contexts],
        }
 
        if sample.ground_truth:
            data["ground_truth"] = [sample.ground_truth]
 
        dataset = Dataset.from_dict(data)
 
        # Configure LLM and embeddings if provided
        if self.llm:
            ragas_metric.llm = self.llm
        if self.embeddings:
            ragas_metric.embeddings = self.embeddings
 
        result = evaluate(dataset, metrics=[ragas_metric])
 
        # Extract score from result
        score_key = self.ragas_metric_name
        if score_key in result:
            return result[score_key]
 
        # Fallback — try first numeric value in result
        for key, value in result.items():
            if isinstance(value, (int, float)):
                return float(value)
 
        return 0.0
 
    def batch_score(self, samples: List[EvalSample]) -> List[MetricResult]:
        """
        Override batch_score to run RAGAS on the full dataset at once.
 
        RAGAS is more efficient when run on a full dataset rather
        than sample by sample, it can batch embedding calls.
        This override takes advantage of that efficiency.
        """
        try:
            return self._run_ragas_batch(samples)
        except Exception as e:
            print(f"[{self.name}] Batch scoring failed, falling back to single: {e}")
            return super().batch_score(samples)
 
    def _run_ragas_batch(
        self,
        samples: List[EvalSample]
    ) -> List[MetricResult]:
        """
        Run RAGAS on the full dataset for efficiency.
        """
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness, answer_relevancy,
            context_precision, context_recall,
            answer_correctness, answer_similarity
        )
 
        metric_map = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "answer_correctness": answer_correctness,
            "answer_similarity": answer_similarity
        }
 
        ragas_metric = metric_map[self.ragas_metric_name]
 
        data = {
            "question": [s.question for s in samples],
            "answer": [s.answer for s in samples],
            "contexts": [s.contexts for s in samples],
        }
 
        if any(s.ground_truth for s in samples):
            data["ground_truth"] = [
                s.ground_truth or "" for s in samples
            ]
 
        dataset = Dataset.from_dict(data)
 
        if self.llm:
            ragas_metric.llm = self.llm
        if self.embeddings:
            ragas_metric.embeddings = self.embeddings
 
        result = evaluate(dataset, metrics=[ragas_metric])
 
        # Extract per-sample scores
        scores = result.to_pandas()[self.ragas_metric_name].tolist()
 
        return [
            MetricResult(
                metric_name=self.name,
                score=round(float(score), 4),
                reasoning=f"RAGAS {self.ragas_metric_name}: {score:.4f}",
                threshold=self.threshold,
                metadata={"ragas_metric": self.ragas_metric_name}
            )
            for score in scores
        ]
 
 
class RAGASEvaluator:
    """
    Convenience class for running the full RAGAS suite at once.
 
    Wraps all 4 core RAGAS metrics (faithfulness, answer_relevancy,
    context_precision, context_recall) and returns results in
    EvalReport format.
 
    Usage:
        ragas_eval = RAGASEvaluator(llm=my_llm, embeddings=my_embeddings)
        metrics = ragas_eval.get_metrics()
 
        evaluator = Evaluator(metrics=metrics)
        report = evaluator.evaluate(samples)
    """
 
    CORE_METRICS = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall"
    ]
 
    def __init__(self, llm=None, embeddings=None, threshold: float = 0.7):
        self.llm = llm
        self.embeddings = embeddings
        self.threshold = threshold
 
    def get_metrics(self) -> List[RAGASMetric]:
        """
        Return all 4 core RAGAS metrics as RAGASMetric instances
        ready to pass to the Evaluator.
        """
        return [
            RAGASMetric(
                ragas_metric_name=metric_name,
                llm=self.llm,
                embeddings=self.embeddings,
                threshold=self.threshold
            )
            for metric_name in self.CORE_METRICS
        ]
