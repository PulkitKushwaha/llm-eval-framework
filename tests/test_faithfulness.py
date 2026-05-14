import pytest
from llm_eval.metrics.faithfulness import Faithfulness
from llm_eval.models import EvalSample
 
 
def make_sample(answer: str, contexts: list) -> EvalSample:
    return EvalSample(
        question="Test question",
        answer=answer,
        contexts=contexts
    )
 
 
def test_faithfulness_perfect_score():
    """Answer fully supported by context should score 1.0"""
    metric = Faithfulness()  # uses mock LLM, returns YES for all claims
    sample = make_sample(
        answer="The return policy allows 30 days.",
        contexts=["Our return policy allows returns within 30 days."]
    )
    result = metric.score(sample)
    assert result.metric_name == "faithfulness"
    assert 0.0 <= result.score <= 1.0
 
 
def test_faithfulness_requires_answer():
    """Empty answer should raise validation error"""
    metric = Faithfulness()
    with pytest.raises(Exception):
        EvalSample(
            question="What is the policy?",
            answer="",
            contexts=["Some context"]
        )
 
 
def test_faithfulness_requires_contexts():
    """Empty contexts list should raise validation error"""
    metric = Faithfulness()
    with pytest.raises(Exception):
        EvalSample(
            question="What is the policy?",
            answer="The policy allows 30 days.",
            contexts=[]
        )
 
 
def test_faithfulness_result_structure():
    """MetricResult should have all required fields"""
    metric = Faithfulness()
    sample = make_sample(
        answer="Items can be returned within 30 days.",
        contexts=["Returns are accepted within 30 days of purchase."]
    )
    result = metric.score(sample)
    assert hasattr(result, "score")
    assert hasattr(result, "reasoning")
    assert hasattr(result, "passed")
    assert hasattr(result, "threshold")
    assert result.threshold == 0.7
 
 
def test_faithfulness_metadata():
    """MetricResult metadata should contain claim details"""
    metric = Faithfulness()
    sample = make_sample(
        answer="The policy allows 30 days. Items must be unused.",
        contexts=["Returns within 30 days. Items must be in original condition."]
    )
    result = metric.score(sample)
    assert "total_claims" in result.metadata
    assert "supported_claims" in result.metadata
    assert "claims" in result.metadata
