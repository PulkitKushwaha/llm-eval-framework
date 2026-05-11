from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
 
 
class EvalSample(BaseModel):
    """
    Represents a single evaluation sample, basically the atomic unit of evaluation.
 
    This is the input contract for the entire framework. Any RAG pipeline
    can be evaluated as long as it can produce these four fields.
 
    Fields:
        question     : The user's original query
        answer       : The RAG pipeline's generated response
        contexts     : The list of retrieved chunks passed to the LLM
        ground_truth : The correct answer (used by recall-based metrics)
        metadata     : Optional dict for tracking source, pipeline version, etc.
    """
    question: str = Field(..., description="The user's original query")
    answer: str = Field(..., description="The RAG pipeline's generated response")
    contexts: List[str] = Field(..., description="Retrieved chunks passed to the LLM")
    ground_truth: Optional[str] = Field(None, description="The correct answer for recall-based metrics")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional tracking metadata")
 
    @validator("contexts")
    def contexts_must_not_be_empty(cls, v):
        if not v:
            raise ValueError("contexts must contain at least one retrieved chunk")
        return v
 
    @validator("question", "answer")
    def fields_must_not_be_blank(cls, v):
        if not v.strip():
            raise ValueError("question and answer must not be blank")
        return v
 
 
class MetricResult(BaseModel):
    """
    Represents the result of running a single metric on a single EvalSample.
 
    Fields:
        metric_name  : Name of the metric that produced this result
        score        : Numeric score between 0.0 and 1.0
        reasoning    : LLM-generated explanation of why this score was given
        passed       : Boolean — did this sample meet the minimum threshold?
        threshold    : The minimum acceptable score for this metric
        metadata     : Optional dict for storing intermediate computation details
    """
    metric_name: str
    score: float = Field(..., ge=0.0, le=1.0, description="Score between 0.0 and 1.0")
    reasoning: Optional[str] = Field(None, description="Explanation of the score")
    passed: bool = Field(default=True, description="Whether score meets the threshold")
    threshold: float = Field(default=0.7, description="Minimum acceptable score")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
 
    @validator("passed", always=True, pre=False)
    def compute_passed(cls, v, values):
        if "score" in values and "threshold" in values:
            return values["score"] >= values["threshold"]
        return v
 
 
class SampleEvalResult(BaseModel):
    """
    Aggregates all MetricResults for a single EvalSample.
 
    Fields:
        sample       : The original EvalSample that was evaluated
        results      : Dict mapping metric name to its MetricResult
        overall_score: Mean score across all metrics for this sample
    """
    sample: EvalSample
    results: Dict[str, MetricResult] = Field(default_factory=dict)
    overall_score: Optional[float] = None
 
    def compute_overall(self) -> float:
        if not self.results:
            return 0.0
        self.overall_score = sum(r.score for r in self.results.values()) / len(self.results)
        return self.overall_score
 
 
class EvalReport(BaseModel):
    """
    The final evaluation report — aggregates results across all samples and metrics.
 
    This is what the Evaluator returns after a full evaluation run.
    It can be serialized to JSON or rendered as a Markdown table
    by the reporter classes.
 
    Fields:
        run_id          : Unique identifier for this evaluation run
        timestamp       : When the evaluation was run
        num_samples     : Total number of EvalSamples evaluated
        metric_names    : List of metrics that were run
        sample_results  : Full results for every sample
        metric_averages : Mean score per metric across all samples
        overall_score   : Mean score across all metrics and samples
        metadata        : Optional run-level metadata (pipeline version, model, etc.)
    """
    run_id: str = Field(default_factory=lambda: datetime.utcnow().strftime("%Y%m%d_%H%M%S"))
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    num_samples: int = 0
    metric_names: List[str] = Field(default_factory=list)
    sample_results: List[SampleEvalResult] = Field(default_factory=list)
    metric_averages: Dict[str, float] = Field(default_factory=dict)
    overall_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
 
    def compute_aggregates(self) -> "EvalReport":
        """Compute metric averages and overall score from sample results."""
        if not self.sample_results:
            return self
 
        self.num_samples = len(self.sample_results)
 
        # Compute per-metric averages
        for metric_name in self.metric_names:
            scores = [
                sr.results[metric_name].score
                for sr in self.sample_results
                if metric_name in sr.results
            ]
            if scores:
                self.metric_averages[metric_name] = round(sum(scores) / len(scores), 4)
 
        # Compute overall score
        if self.metric_averages:
            self.overall_score = round(
                sum(self.metric_averages.values()) / len(self.metric_averages), 4
            )
 
        return self
 
    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            f"Evaluation Report — {self.timestamp}",
            f"Samples: {self.num_samples} | Metrics: {len(self.metric_names)}",
            "-" * 50,
        ]
        for metric, avg in self.metric_averages.items():
            status = "Looks Good" if avg >= 0.7 else "Need to improve" if avg >= 0.5 else "Nopes, not alowed"
            lines.append(f"{status}  {metric:<30} {avg:.4f}")
        lines.append("-" * 50)
        lines.append(f"Overall Score: {self.overall_score:.4f}")
        return "\n".join(lines)
