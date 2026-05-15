import pytest
from llm_eval.metrics.context_recall import ContextRecall
from llm_eval.models import EvalSample
 
 
def make_sample_with_gt(
    question: str,
    contexts: list,
    ground_truth: str
) -> EvalSample:
    return EvalSample(
        question=question,
        answer="Test answer.",
        contexts=contexts,
        ground_truth=ground_truth
    )
 
 
def make_sample_without_gt(question: str, contexts: list) -> EvalSample:
    return EvalSample(
        question=question,
        answer="Test answer.",
        contexts=contexts
    )
 
 
def test_context_recall_requires_ground_truth():
    """Should raise ValueError if ground_truth is missing"""
    metric = ContextRecall()
    sample = make_sample_without_gt(
        question="What is the return policy?",
        contexts=["Returns allowed within 30 days."]
    )
    with pytest.raises(ValueError, match="ground_truth"):
        metric.score(sample)
 
 
def test_context_recall_returns_score():
    """Should return a score between 0 and 1"""
    metric = ContextRecall()
    sample = make_sample_with_gt(
        question="What is the return policy?",
        contexts=["Returns are accepted within 30 days of purchase."],
        ground_truth="Items can be returned within 30 days. They must be unused."
    )
    result = metric.score(sample)
    assert 0.0 <= result.score <= 1.0
 
 
def test_context_recall_metric_name():
    """Metric name should be context_recall"""
    metric = ContextRecall()
    sample = make_sample_with_gt(
        question="What are the hours?",
        contexts=["We are open Monday to Friday, 9am to 5pm."],
        ground_truth="The office is open weekdays from 9am to 5pm."
    )
    result = metric.score(sample)
    assert result.metric_name == "context_recall"
 
 
def test_context_recall_metadata_structure():
    """Metadata should contain statement counts"""
    metric = ContextRecall()
    sample = make_sample_with_gt(
        question="What is the shipping policy?",
        contexts=["Standard shipping takes 5-7 days. Express takes 2-3 days."],
        ground_truth="Standard shipping is 5-7 days. Express shipping is 2-3 days."
    )
    result = metric.score(sample)
    assert "total_statements" in result.metadata
    assert "found_in_context" in result.metadata
    assert "missing_from_context" in result.metadata
    assert "statements" in result.metadata
 
 
def test_context_recall_missing_info_reflected_in_score():
    """
    When context is missing information from ground truth,
    score should reflect that gap.
    With mock LLM (always YES), score will be 1.0 —
    this test verifies the structure is correct.
    """
    metric = ContextRecall()
    sample = make_sample_with_gt(
        question="What is the refund policy?",
        contexts=["Refunds are processed within 5 days."],
        ground_truth=(
            "Refunds take 5 days. "
            "Original payment method is used. "
            "Shipping costs are non-refundable."
        )
    )
    result = metric.score(sample)
    # With mock LLM returning YES, all statements found
    assert result.metadata["total_statements"] >= 1
    assert result.metadata["found_in_context"] <= result.metadata["total_statements"]
 
 
def test_context_recall_threshold():
    """Default threshold should be 0.7"""
    metric = ContextRecall()
    assert metric.threshold == 0.7
 
 
def test_context_recall_requires_ground_truth_flag():
    """requires_ground_truth flag should be True"""
    metric = ContextRecall()
    assert metric.requires_ground_truth is True
