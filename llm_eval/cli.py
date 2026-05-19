"""
CLI interface for llm-eval-framework.
 
Allows running evaluations from the command line without
writing any Python code. Useful for CI/CD pipelines and
quick one-off evaluations.
 
Usage:
    # Run all core metrics
    python -m llm_eval \\
        --dataset examples/rag_pipeline_eval/dataset/test_questions.json \\
        --metrics faithfulness,answer_relevancy,context_precision,context_recall \\
        --output markdown
 
    # Save results to file
    python -m llm_eval \\
        --dataset data/test.json \\
        --metrics faithfulness,answer_relevancy \\
        --output json \\
        --save results/report.json
 
    # With pipeline metadata for tracking
    python -m llm_eval \\
        --dataset data/test.json \\
        --metrics faithfulness \\
        --output markdown \\
        --pipeline-version v1.2 \\
        --model gpt-4
 
Dataset JSON format:
    [
        {
            "question": "What is the return policy?",
            "answer": "Items can be returned within 30 days.",
            "contexts": ["Our policy allows returns within 30 days."],
            "ground_truth": "Returns are accepted within 30 days."
        }
    ]
 
    Note: ground_truth is optional unless using context_recall
    or answer_correctness metrics.
"""
 
import json
import sys
from pathlib import Path
from typing import Optional
 
import click
 
from llm_eval.evaluator import Evaluator
from llm_eval.models import EvalSample
from llm_eval.reporters.json_reporter import JSONReporter
from llm_eval.reporters.markdown_reporter import MarkdownReporter
 
 
# Available metrics mapping
AVAILABLE_METRICS = {
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "ragas_faithfulness",
    "ragas_answer_relevancy",
    "ragas_context_precision",
    "ragas_context_recall",
}
 
 
def load_metrics(metric_names: list) -> list:
    """
    Instantiate metric objects from a list of metric name strings.
 
    Imports are done lazily here to avoid loading all dependencies
    unless the CLI is actually called.
    """
    from llm_eval.metrics import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        RAGASMetric
    )
 
    metric_map = {
        "faithfulness": Faithfulness,
        "answer_relevancy": AnswerRelevancy,
        "context_precision": ContextPrecision,
        "context_recall": ContextRecall,
    }
 
    metrics = []
    for name in metric_names:
        name = name.strip().lower()
 
        if name.startswith("ragas_"):
            ragas_metric_name = name[6:]  # strip "ragas_" prefix
            metrics.append(RAGASMetric(ragas_metric_name=ragas_metric_name))
        elif name in metric_map:
            metrics.append(metric_map[name]())
        else:
            available = ", ".join(sorted(AVAILABLE_METRICS))
            raise click.BadParameter(
                f"Unknown metric: '{name}'. Available: {available}",
                param_hint="--metrics"
            )
 
    return metrics
 
 
def load_dataset(dataset_path: str) -> list:
    """
    Load evaluation dataset from a JSON file.
 
    Expected format: list of dicts with keys:
        question, answer, contexts, ground_truth (optional)
    """
    path = Path(dataset_path)
 
    if not path.exists():
        raise click.BadParameter(
            f"Dataset file not found: {dataset_path}",
            param_hint="--dataset"
        )
 
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
 
    if not isinstance(data, list):
        raise click.BadParameter(
            "Dataset must be a JSON array of sample objects.",
            param_hint="--dataset"
        )
 
    samples = []
    for i, item in enumerate(data):
        try:
            sample = EvalSample(
                question=item["question"],
                answer=item["answer"],
                contexts=item["contexts"],
                ground_truth=item.get("ground_truth")
            )
            samples.append(sample)
        except KeyError as e:
            raise click.BadParameter(
                f"Sample {i} is missing required field: {e}. "
                f"Required fields: question, answer, contexts",
                param_hint="--dataset"
            )
 
    return samples
 
 
@click.command()
@click.option(
    "--dataset", "-d",
    required=True,
    help="Path to evaluation dataset JSON file."
)
@click.option(
    "--metrics", "-m",
    required=True,
    help=(
        "Comma-separated list of metrics to run. "
        "Available: faithfulness, answer_relevancy, "
        "context_precision, context_recall, "
        "ragas_faithfulness, ragas_answer_relevancy, "
        "ragas_context_precision, ragas_context_recall"
    )
)
@click.option(
    "--output", "-o",
    type=click.Choice(["markdown", "json", "summary"]),
    default="summary",
    help="Output format. Default: summary"
)
@click.option(
    "--save", "-s",
    default=None,
    help="File path to save the report. Optional."
)
@click.option(
    "--pipeline-version",
    default=None,
    help="Pipeline version tag for tracking. Optional."
)
@click.option(
    "--model",
    default=None,
    help="Model name for tracking. Optional."
)
@click.option(
    "--quiet", "-q",
    is_flag=True,
    default=False,
    help="Suppress progress output."
)
def main(
    dataset: str,
    metrics: str,
    output: str,
    save: Optional[str],
    pipeline_version: Optional[str],
    model: Optional[str],
    quiet: bool
):
    """
    Run LLM evaluations from the command line.
 
    Examples:\n
        python -m llm_eval --dataset test.json --metrics faithfulness --output markdown\n
        python -m llm_eval --dataset test.json --metrics faithfulness,answer_relevancy --output json --save results.json
    """
    # Parse metric names
    metric_names = [m.strip() for m in metrics.split(",") if m.strip()]
 
    if not quiet:
        click.echo(f"\nllm-eval-framework")
        click.echo(f"Dataset:  {dataset}")
        click.echo(f"Metrics:  {', '.join(metric_names)}")
        click.echo(f"Output:   {output}")
        click.echo("")
 
    # Load dataset
    try:
        samples = load_dataset(dataset)
    except Exception as e:
        click.echo(f"Error loading dataset: {e}", err=True)
        sys.exit(1)
 
    if not quiet:
        click.echo(f"Loaded {len(samples)} samples.")
 
    # Load metrics
    try:
        metric_objects = load_metrics(metric_names)
    except Exception as e:
        click.echo(f"Error loading metrics: {e}", err=True)
        sys.exit(1)
 
    # Build metadata
    metadata = {}
    if pipeline_version:
        metadata["pipeline_version"] = pipeline_version
    if model:
        metadata["model"] = model
 
    # Run evaluation
    evaluator = Evaluator(
        metrics=metric_objects,
        metadata=metadata,
        verbose=not quiet
    )
 
    try:
        report = evaluator.evaluate(samples)
    except Exception as e:
        click.echo(f"Evaluation failed: {e}", err=True)
        sys.exit(1)
 
    # Render output
    if output == "json":
        reporter = JSONReporter(indent=2)
        result_str = reporter.report(report)
        click.echo(result_str)
        if save:
            reporter.save(report, save)
 
    elif output == "markdown":
        reporter = MarkdownReporter(include_per_sample=True)
        result_str = reporter.report(report)
        click.echo(result_str)
        if save:
            reporter.save(report, save)
 
    else:  # summary
        click.echo(report.summary())
        if save:
            # Save as markdown by default for summary
            reporter = MarkdownReporter(include_per_sample=False)
            reporter.save(report, save)
 
    # Exit with non-zero code if overall score below threshold
    # Useful for CI/CD pass/fail gates
    overall = report.overall_score or 0.0
    if overall < 0.7:
        if not quiet:
            click.echo(
                f"\nOverall score {overall:.4f} is below threshold 0.70",
                err=True
            )
        sys.exit(1)
 
    sys.exit(0)
