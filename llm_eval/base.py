from abc import ABC, abstractmethod
from typing import List, Optional
from llm_eval.models import EvalSample, MetricResult
 
 
class BaseMetric(ABC):
    """
    Abstract base class for all evaluation metrics.
 
    Every metric in this framework — built-in or custom — must inherit
    from BaseMetric and implement the score() method.
 
    This contract is what allows the Evaluator to run any combination
    of metrics through the same interface without knowing their internals.
 
    To implement a custom metric:
 
        class MyCustomMetric(BaseMetric):
            name = "my_custom_metric"
            threshold = 0.75
 
            def score(self, sample: EvalSample) -> MetricResult:
                # Your scoring logic here
                my_score = ...
                return MetricResult(
                    metric_name=self.name,
                    score=my_score,
                    reasoning="Explanation of why this score was given",
                    threshold=self.threshold
                )
 
    Then pass it to the Evaluator:
 
        evaluator = Evaluator(metrics=[MyCustomMetric()])
        report = evaluator.evaluate(samples)
    """
 
    # Every subclass must define a name which will be used in reports and logging
    name: str = "base_metric"
 
    # Default threshold: subclasses can override it
    threshold: float = 0.7
 
    # Whether this metric requires ground_truth in EvalSample
    requires_ground_truth: bool = False
 
    @abstractmethod
    def score(self, sample: EvalSample) -> MetricResult:
        """
        Score a single EvalSample.
 
        Args:
            sample: An EvalSample containing question, answer, contexts,
                    and optionally ground_truth.
 
        Returns:
            A MetricResult containing the score, reasoning, and pass/fail status.
 
        Raises:
            ValueError: If the sample is missing required fields for this metric.
                        For example, context recall requires ground_truth.
        """
        raise NotImplementedError
 
    def batch_score(self, samples: List[EvalSample]) -> List[MetricResult]:
        """
        Score a list of EvalSamples.
 
        Default implementation loops over score(): subclasses can override
        this for metrics that benefit from batching (e.g. batch embedding calls).
 
        Args:
            samples: List of EvalSamples to score.
 
        Returns:
            List of MetricResults, one per sample, in the same order.
        """
        results = []
        for i, sample in enumerate(samples):
            try:
                result = self.score(sample)
            except Exception as e:
                # Log the failure but don't crash the entire evaluation run
                print(f"[{self.name}] Failed to score sample {i}: {e}")
                result = MetricResult(
                    metric_name=self.name,
                    score=0.0,
                    reasoning=f"Scoring failed: {str(e)}",
                    threshold=self.threshold
                )
            results.append(result)
        return results
 
    def validate_sample(self, sample: EvalSample) -> None:
        """
        Validate that a sample has all required fields for this metric.
        Called automatically before score() in the Evaluator.
 
        Raises:
            ValueError: If a required field is missing.
        """
        if self.requires_ground_truth and not sample.ground_truth:
            raise ValueError(
                f"Metric '{self.name}' requires ground_truth but it was not provided. "
                f"Question: '{sample.question[:50]}...'"
            )
 
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', threshold={self.threshold})"
