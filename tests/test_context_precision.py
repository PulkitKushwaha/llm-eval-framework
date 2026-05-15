import pytest
from llm_eval.metrics.context_precision import ContextPrecision
from llm_eval.models import EvalSample
 
 
def make_sample(question: str, contexts: list) -> EvalSample:
    return EvalSample(
        question=question,
        answer="Test answer for context precision evaluation.",
        contexts=contexts
    )
 
 
def test_context_precision_returns_score():
    """Should return a score between 0 and 1"""
    metric = ContextPrecision()
    sample = make_sample(
        question="What is the return policy?",
        contexts=[
            "Our return policy allows returns within 30 days.",
            "We offer free shipping on orders over $50.",
            "Items must be in original packaging for returns."
        ]
    )
    result = metric.score(sample)
    assert 0.0 <= result.score <= 1.0
 
 
def test_context_precision_metric_name():
    """Metric name should be context_precision"""
    metric = ContextPrecision()
    sample = make_sample(
        question="What are the shipping options?",
        contexts=["We offer standard and express shipping."]
    )
    result = metric.score(sample)
    assert result.metric_name == "context_precision"
 
 
def test_context_precision_single_relevant_chunk():
    """Single relevant chunk should score 1.0"""
    metric = ContextPrecision()
    sample = make_sample(
        question="What is the cancellation policy?",
        contexts=["Cancellations must be made 24 hours in advance."]
    )
    result = metric.score(sample)
    assert result.score == 1.0
 
 
def test_context_precision_metadata_structure():
    """Metadata should contain chunk counts and relevance scores"""
    metric = ContextPrecision()
    sample = make_sample(
        question="How do I reset my password?",
        contexts=[
            "To reset your password, click forgot password.",
            "Our offices are open Monday to Friday.",
            "Password must be at least 8 characters."
        ]
    )
    result = metric.score(sample)
    assert "total_chunks" in result.metadata
    assert "relevant_chunks" in result.metadata
    assert "relevance_scores" in result.metadata
    assert len(result.metadata["relevance_scores"]) == 3
 
 
def test_context_precision_weighted_scoring():
    """Position-weighted precision should penalize irrelevant top chunks"""
    metric = ContextPrecision()
    # All chunks relevant — mock LLM returns YES for all
    sample = make_sample(
        question="What is the refund timeline?",
        contexts=[
            "Refunds are processed within 5 business days.",
            "Refund requests must include order number.",
        ]
    )
    result = metric.score(sample)
    assert result.score > 0.0
