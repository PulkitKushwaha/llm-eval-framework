import pytest
from llm_eval.metrics.answer_relevancy import AnswerRelevancy
from llm_eval.models import EvalSample
 
 
def make_sample(question: str, answer: str) -> EvalSample:
    return EvalSample(
        question=question,
        answer=answer,
        contexts=["Some retrieved context for this test."]
    )
 
 
def test_answer_relevancy_returns_score():
    """Should return a score between 0 and 1"""
    metric = AnswerRelevancy()
    sample = make_sample(
        question="What is the return policy?",
        answer="Items can be returned within 30 days of purchase."
    )
    result = metric.score(sample)
    assert 0.0 <= result.score <= 1.0
 
 
def test_answer_relevancy_metric_name():
    """Metric name should be answer_relevancy"""
    metric = AnswerRelevancy()
    sample = make_sample(
        question="What is the refund process?",
        answer="Refunds are processed within 5 business days."
    )
    result = metric.score(sample)
    assert result.metric_name == "answer_relevancy"
 
 
def test_answer_relevancy_result_structure():
    """MetricResult should have all required fields"""
    metric = AnswerRelevancy()
    sample = make_sample(
        question="How do I contact support?",
        answer="You can contact support via email at support@company.com."
    )
    result = metric.score(sample)
    assert hasattr(result, "score")
    assert hasattr(result, "reasoning")
    assert hasattr(result, "passed")
    assert hasattr(result, "threshold")
 
 
def test_answer_relevancy_metadata_contains_reverse_questions():
    """Metadata should contain the generated reverse questions"""
    metric = AnswerRelevancy()
    sample = make_sample(
        question="What are the shipping options?",
        answer="We offer standard (5-7 days) and express (2-3 days) shipping."
    )
    result = metric.score(sample)
    assert "reverse_questions" in result.metadata
    assert isinstance(result.metadata["reverse_questions"], list)
 
 
def test_answer_relevancy_keyword_fallback():
    """Keyword similarity fallback should work without embedder"""
    metric = AnswerRelevancy(embedder=None)
    sample = make_sample(
        question="What is the cancellation policy?",
        answer="Cancellations must be made 24 hours before the appointment."
    )
    result = metric.score(sample)
    assert result.score >= 0.0
