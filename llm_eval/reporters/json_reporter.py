import json
from pathlib import Path
from typing import Optional
from llm_eval.models import EvalReport
 
 
class JSONReporter:
    """
    Serializes an EvalReport to JSON format.
 
    Designed for machine consumption: CI/CD pipelines,
    programmatic result tracking, and integration with
    monitoring dashboards.
 
    The JSON output is structured so downstream tools can:
        - Parse metric averages for pass/fail gates in CI
        - Store results in a database for trend tracking
        - Feed into monitoring dashboards like Grafana
        - Compare runs across pipeline versions
 
    Usage:
        reporter = JSONReporter()
 
        # Print to console
        print(reporter.report(eval_report))
 
        # Save to file
        reporter.save(eval_report, "results/eval_report.json")
 
        # With pretty printing
        reporter = JSONReporter(indent=2)
    """
 
    def __init__(self, indent: int = 2):
        """
        Args:
            indent: JSON indentation level. Default 2 for readability.
                    Set to None for compact single-line output.
        """
        self.indent = indent
 
    def report(self, eval_report: EvalReport) -> str:
        """
        Serialize EvalReport to a JSON string.
 
        Args:
            eval_report: Completed EvalReport from Evaluator.evaluate()
 
        Returns:
            JSON string representation of the report.
        """
        output = self._build_output(eval_report)
        return json.dumps(output, indent=self.indent, default=str)
 
    def save(
        self,
        eval_report: EvalReport,
        output_path: str
    ) -> None:
        """
        Save EvalReport as a JSON file.
 
        Creates parent directories if they don't exist.
 
        Args:
            eval_report : Completed EvalReport
            output_path : File path to save to (e.g. results/report.json)
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
 
        json_str = self.report(eval_report)
        with open(path, "w", encoding="utf-8") as f:
            f.write(json_str)
 
        print(f"JSON report saved to {output_path}")
 
    def _build_output(self, report: EvalReport) -> dict:
        """
        Build the output dictionary from EvalReport.
 
        Structure:
            run_id, timestamp, summary, metric_averages,
            per_sample_results, metadata
        """
        return {
            "run_id": report.run_id,
            "timestamp": report.timestamp,
            "summary": {
                "num_samples": report.num_samples,
                "metrics_evaluated": report.metric_names,
                "overall_score": report.overall_score,
                "metric_averages": report.metric_averages,
                "passed": {
                    metric: score >= 0.7
                    for metric, score in report.metric_averages.items()
                }
            },
            "per_sample_results": [
                {
                    "question": sr.sample.question,
                    "answer": sr.sample.answer[:200] + "..."
                    if len(sr.sample.answer) > 200
                    else sr.sample.answer,
                    "overall_score": sr.overall_score,
                    "metric_scores": {
                        metric_name: {
                            "score": result.score,
                            "passed": result.passed,
                            "reasoning": result.reasoning
                        }
                        for metric_name, result in sr.results.items()
                    }
                }
                for sr in report.sample_results
            ],
            "metadata": report.metadata
        }
