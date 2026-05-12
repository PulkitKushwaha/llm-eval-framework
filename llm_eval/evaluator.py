from typing import List, Optional, Dict, Any
from datetime import datetime
import time
 
from llm_eval.base import BaseMetric
from llm_eval.models import EvalSample, EvalReport, SampleEvalResult, MetricResult
 
 
class Evaluator:
    """
    The core orchestration engine of llm-eval-framework.
 
    The Evaluator accepts a list of metrics and runs them against a dataset
    of EvalSamples. It handles validation, scoring, aggregation, and returns
    a fully populated EvalReport.
 
    The Evaluator is deliberately agnostic about metric internals, it only
    speaks the BaseMetric contract. This means any combination of built-in
    metrics, RAGAS wrappers, or custom metrics can be passed in and the
    Evaluator will handle them identically.
 
    Basic usage:
        from llm_eval import Evaluator
        from llm_eval.metrics import Faithfulness, AnswerRelevancy
        from llm_eval.models import EvalSample
 
        samples = [
            EvalSample(
                question="What is the return policy?",
                answer="Items can be returned within 30 days.",
                contexts=["Our policy allows returns within 30 days."],
                ground_truth="Items purchased can be returned within 30 days."
            )
        ]
 
        evaluator = Evaluator(
            metrics=[Faithfulness(), AnswerRelevancy()]
        )
        report = evaluator.evaluate(samples)
        print(report.summary())
 
    With metadata tracking:
        evaluator = Evaluator(
            metrics=[Faithfulness(), AnswerRelevancy()],
            metadata={
                "pipeline_version": "v1.2",
                "model": "gpt-4",
                "chunking_strategy": "recursive"
            }
        )
    """
 
    def __init__(
        self,
        metrics: List[BaseMetric],
        metadata: Optional[Dict[str, Any]] = None,
        verbose: bool = True
    ):
        """
        Initialize the Evaluator.
 
        Args:
            metrics:  List of metric instances to run. Each must inherit
                      from BaseMetric and implement score().
            metadata: Optional dict for tracking run-level context:
                      pipeline version, model name, chunking strategy, etc.
                      Stored in the EvalReport for traceability.
            verbose:  If True, prints progress during evaluation.
                      Set to False for CI/CD pipelines.
 
        Raises:
            ValueError: If metrics list is empty.
        """
        if not metrics:
            raise ValueError(
                "Evaluator requires at least one metric. "
                "Pass a list of BaseMetric instances."
            )
 
        self.metrics = metrics
        self.metadata = metadata or {}
        self.verbose = verbose
        self._metric_names = [m.name for m in metrics]
 
        if self.verbose:
            print(f"Evaluator initialized with {len(metrics)} metric(s): "
                  f"{', '.join(self._metric_names)}")
 
    def evaluate(self, samples: List[EvalSample]) -> EvalReport:
        """
        Run all metrics against all samples and return an EvalReport.
 
        This is the main entry point. It orchestrates the full evaluation
        pipeline: validation → scoring → assembly → aggregation.
 
        Args:
            samples: List of EvalSamples to evaluate. Each sample must
                     contain at minimum: question, answer, contexts.
                     ground_truth is required by recall-based metrics.
 
        Returns:
            EvalReport: Fully populated report with per-sample results,
                        per-metric averages, and overall score.
 
        Raises:
            ValueError: If samples list is empty.
            ValueError: If a sample is missing required fields for a metric.
        """
        if not samples:
            raise ValueError("Cannot evaluate an empty dataset.")
 
        start_time = time.time()
 
        if self.verbose:
            print(f"\nStarting evaluation — {len(samples)} sample(s), "
                  f"{len(self.metrics)} metric(s)")
            print("-" * 50)
 
        # Step 1 — Validate all samples against all metric requirements
        self._validate_samples(samples)
 
        # Step 2 — Score all samples with all metrics
        all_metric_results = self._score_all(samples)
 
        # Step 3 — Assemble per-sample results
        sample_results = self._assemble_sample_results(samples, all_metric_results)
 
        # Step 4 — Build and aggregate the final report
        report = self._build_report(sample_results)
 
        elapsed = round(time.time() - start_time, 2)
 
        if self.verbose:
            print("-" * 50)
            print(f"Evaluation complete in {elapsed}s")
            print(report.summary())
 
        return report
 
    def _validate_samples(self, samples: List[EvalSample]) -> None:
        """
        Validate all samples against all metric requirements.
 
        Fails fast: raises ValueError on the first invalid sample
        rather than discovering the problem mid-evaluation.
 
        Args:
            samples: List of EvalSamples to validate.
 
        Raises:
            ValueError: If any sample fails validation for any metric.
        """
        if self.verbose:
            print("Validating samples...")
 
        for metric in self.metrics:
            for i, sample in enumerate(samples):
                try:
                    metric.validate_sample(sample)
                except ValueError as e:
                    raise ValueError(
                        f"Validation failed for metric '{metric.name}' "
                        f"on sample {i} (question: '{sample.question[:50]}...'): {e}"
                    )
 
        if self.verbose:
            print(f"All {len(samples)} samples passed validation")
 
    def _score_all(
        self,
        samples: List[EvalSample]
    ) -> Dict[str, List[MetricResult]]:
        """
        Run batch scoring for every metric.
 
        Returns a dict mapping metric name to its list of MetricResults.
        One MetricResult per sample, in the same order as samples.
 
        Args:
            samples: List of EvalSamples to score.
 
        Returns:
            Dict mapping metric_name → List[MetricResult]
        """
        all_results = {}
 
        for metric in self.metrics:
            if self.verbose:
                print(f"Running metric: {metric.name}...")
 
            metric_start = time.time()
            results = metric.batch_score(samples)
            metric_elapsed = round(time.time() - metric_start, 2)
 
            all_results[metric.name] = results
 
            if self.verbose:
                avg_score = sum(r.score for r in results) / len(results)
                print(f"  {metric.name}: avg={avg_score:.4f} ({metric_elapsed}s)")
 
        return all_results
 
    def _assemble_sample_results(
        self,
        samples: List[EvalSample],
        all_metric_results: Dict[str, List[MetricResult]]
    ) -> List[SampleEvalResult]:
        """
        Assemble per-sample results from the flat metric results dict.
 
        Combines all metric results for each individual sample into a
        SampleEvalResult: one per sample.
 
        Args:
            samples: Original list of EvalSamples.
            all_metric_results: Dict mapping metric_name → List[MetricResult]
 
        Returns:
            List of SampleEvalResults, one per sample.
        """
        sample_results = []
 
        for i, sample in enumerate(samples):
            # Collect this sample's result from every metric
            sample_metric_results = {
                metric_name: results[i]
                for metric_name, results in all_metric_results.items()
            }
 
            sample_result = SampleEvalResult(
                sample=sample,
                results=sample_metric_results
            )
            sample_result.compute_overall()
            sample_results.append(sample_result)
 
        return sample_results
 
    def _build_report(
        self,
        sample_results: List[SampleEvalResult]
    ) -> EvalReport:
        """
        Build the final EvalReport from assembled sample results.
 
        Args:
            sample_results: List of SampleEvalResults.
 
        Returns:
            Fully populated EvalReport with aggregated scores.
        """
        report = EvalReport(
            metric_names=self._metric_names,
            sample_results=sample_results,
            metadata={
                **self.metadata,
                "evaluator_version": "0.1.0",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        report.compute_aggregates()
        return report
 
    def add_metric(self, metric: BaseMetric) -> None:
        """
        Add a metric to the evaluator after initialization.
 
        Useful for incrementally building up an evaluation suite.
 
        Args:
            metric: A BaseMetric instance to add.
        """
        if metric.name in self._metric_names:
            raise ValueError(
                f"Metric '{metric.name}' is already registered. "
                f"Remove it first or use a different name."
            )
        self.metrics.append(metric)
        self._metric_names.append(metric.name)
 
        if self.verbose:
            print(f"Added metric: {metric.name}")
 
    def remove_metric(self, metric_name: str) -> None:
        """
        Remove a metric from the evaluator by name.
 
        Args:
            metric_name: The name of the metric to remove.
 
        Raises:
            ValueError: If the metric name is not found.
        """
        if metric_name not in self._metric_names:
            raise ValueError(
                f"Metric '{metric_name}' not found. "
                f"Available metrics: {', '.join(self._metric_names)}"
            )
        self.metrics = [m for m in self.metrics if m.name != metric_name]
        self._metric_names.remove(metric_name)
 
        if self.verbose:
            print(f"Removed metric: {metric_name}")
 
    def __repr__(self) -> str:
        return (
            f"Evaluator("
            f"metrics=[{', '.join(self._metric_names)}], "
            f"verbose={self.verbose})"
        )
