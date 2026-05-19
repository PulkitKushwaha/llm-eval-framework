from typing import Optional
from pathlib import Path
from llm_eval.models import EvalReport
 
 
class MarkdownReporter:
    """
    Renders an EvalReport as a Markdown table.
 
    Designed for human consumption: GitHub PR comments,
    README badges, and documentation.
 
    The Markdown output is structured so it can be:
        - Posted as a GitHub Actions PR comment automatically
        - Pasted into a repo README as benchmark results
        - Used as a status update in documentation
 
    Usage:
        reporter = MarkdownReporter()
 
        # Print to console
        print(reporter.report(eval_report))
 
        # Save to file
        reporter.save(eval_report, "results/report.md")
 
        # Compact version for PR comments
        reporter = MarkdownReporter(include_per_sample=False)
    """
 
    # Score thresholds for visual indicators
    EXCELLENT_THRESHOLD = 0.85
    GOOD_THRESHOLD = 0.70
    POOR_THRESHOLD = 0.50
 
    def __init__(self, include_per_sample: bool = True):
        """
        Args:
            include_per_sample: If True, includes per-sample breakdown.
                                Set to False for compact PR comment output.
        """
        self.include_per_sample = include_per_sample
 
    def report(self, eval_report: EvalReport) -> str:
        """
        Render EvalReport as a Markdown string.
 
        Args:
            eval_report: Completed EvalReport from Evaluator.evaluate()
 
        Returns:
            Markdown string ready to paste or post.
        """
        sections = [
            self._render_header(eval_report),
            self._render_summary_table(eval_report),
            self._render_metric_breakdown(eval_report),
        ]
 
        if self.include_per_sample and eval_report.sample_results:
            sections.append(self._render_sample_results(eval_report))
 
        sections.append(self._render_footer(eval_report))
 
        return "\n\n".join(sections)
 
    def save(
        self,
        eval_report: EvalReport,
        output_path: str
    ) -> None:
        """
        Save EvalReport as a Markdown file.
 
        Args:
            eval_report : Completed EvalReport
            output_path : File path to save to (e.g. results/report.md)
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
 
        md_str = self.report(eval_report)
        with open(path, "w", encoding="utf-8") as f:
            f.write(md_str)
 
        print(f"Markdown report saved to {output_path}")
 
    def _render_header(self, report: EvalReport) -> str:
        overall = report.overall_score or 0.0
        indicator = self._score_indicator(overall)
        return (
            f"## LLM Evaluation Report {indicator}\n\n"
            f"**Run ID:** `{report.run_id}`  \n"
            f"**Timestamp:** {report.timestamp}  \n"
            f"**Samples evaluated:** {report.num_samples}  \n"
            f"**Overall score:** `{overall:.4f}`"
        )
 
    def _render_summary_table(self, report: EvalReport) -> str:
        rows = ["| Metric | Score | Status | Bar |",
                "|---|---|---|---|"]
 
        for metric, avg in report.metric_averages.items():
            indicator = self._score_indicator(avg)
            status = self._score_label(avg)
            bar = self._score_bar(avg)
            rows.append(f"| {metric} | `{avg:.4f}` | {indicator} {status} | {bar} |")
 
        overall = report.overall_score or 0.0
        indicator = self._score_indicator(overall)
        status = self._score_label(overall)
        bar = self._score_bar(overall)
        rows.append(f"| **Overall** | **`{overall:.4f}`** | {indicator} **{status}** | {bar} |")
 
        return "### Summary\n\n" + "\n".join(rows)
 
    def _render_metric_breakdown(self, report: EvalReport) -> str:
        if not report.metric_averages:
            return ""
 
        lines = ["### Metric Breakdown\n"]
 
        for metric, avg in report.metric_averages.items():
            indicator = self._score_indicator(avg)
            lines.append(f"**{metric}** — `{avg:.4f}` {indicator}")
 
            # Add interpretation
            interpretation = self._interpret_score(metric, avg)
            if interpretation:
                lines.append(f"> {interpretation}")
 
            lines.append("")
 
        return "\n".join(lines)
 
    def _render_sample_results(self, report: EvalReport) -> str:
        rows = ["| # | Question | Overall | " +
                " | ".join(report.metric_names) + " |",
                "|---|---|---|" + "---|" * len(report.metric_names)]
 
        for i, sr in enumerate(report.sample_results, 1):
            question_short = sr.sample.question[:50] + "..." \
                if len(sr.sample.question) > 50 \
                else sr.sample.question
 
            overall = f"`{sr.overall_score:.3f}`" if sr.overall_score else "N/A"
 
            metric_scores = []
            for metric_name in report.metric_names:
                if metric_name in sr.results:
                    score = sr.results[metric_name].score
                    passed = sr.results[metric_name].passed
                    indicator = "✅" if passed else "❌"
                    metric_scores.append(f"{indicator} `{score:.3f}`")
                else:
                    metric_scores.append("N/A")
 
            rows.append(
                f"| {i} | {question_short} | {overall} | "
                + " | ".join(metric_scores) + " |"
            )
 
        return "### Per-Sample Results\n\n" + "\n".join(rows)
 
    def _render_footer(self, report: EvalReport) -> str:
        pipeline_version = report.metadata.get("pipeline_version", "unknown")
        model = report.metadata.get("model", "unknown")
 
        lines = ["---", "**Evaluation details:**"]
 
        if pipeline_version != "unknown":
            lines.append(f"- Pipeline version: `{pipeline_version}`")
        if model != "unknown":
            lines.append(f"- Model: `{model}`")
 
        lines.append(
            f"- Framework: "
            f"[llm-eval-framework]"
            f"(https://github.com/pulkitkushwaha/llm-eval-framework)"
        )
 
        return "\n".join(lines)
 
    def _score_indicator(self, score: float) -> str:
        if score >= self.EXCELLENT_THRESHOLD:
            return "✅"
        elif score >= self.GOOD_THRESHOLD:
            return "🟡"
        elif score >= self.POOR_THRESHOLD:
            return "⚠️"
        else:
            return "❌"
 
    def _score_label(self, score: float) -> str:
        if score >= self.EXCELLENT_THRESHOLD:
            return "Excellent"
        elif score >= self.GOOD_THRESHOLD:
            return "Good"
        elif score >= self.POOR_THRESHOLD:
            return "Moderate"
        else:
            return "Poor"
 
    def _score_bar(self, score: float, width: int = 10) -> str:
        filled = round(score * width)
        empty = width - filled
        return "█" * filled + "░" * empty
 
    def _interpret_score(self, metric: str, score: float) -> str:
        interpretations = {
            "faithfulness": {
                "high": "Answers are well-grounded in retrieved context.",
                "mid": "Some answers contain claims not fully supported by context.",
                "low": "High hallucination risk — answers not grounded in context."
            },
            "answer_relevancy": {
                "high": "Answers directly address the questions asked.",
                "mid": "Some answers are tangential or incomplete.",
                "low": "Answers frequently miss the point of the question."
            },
            "context_precision": {
                "high": "Retriever is returning highly relevant chunks.",
                "mid": "Some irrelevant chunks are being retrieved.",
                "low": "Retriever is returning mostly irrelevant content."
            },
            "context_recall": {
                "high": "Retriever is finding all necessary information.",
                "mid": "Retriever is missing some relevant information.",
                "low": "Retriever is missing most necessary information."
            }
        }
 
        metric_interp = interpretations.get(metric, {})
        if not metric_interp:
            return ""
 
        if score >= self.GOOD_THRESHOLD:
            return metric_interp.get("high", "")
        elif score >= self.POOR_THRESHOLD:
            return metric_interp.get("mid", "")
        else:
            return metric_interp.get("low", "")
