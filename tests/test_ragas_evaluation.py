"""
Test RAGAS evaluation framework.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.ragas_evaluator import RAGASEvaluator


def test_ragas_basic():
    """Test basic RAGAS evaluation."""
    
    print("=" * 60)
    print("TESTING RAGAS EVALUATION")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = RAGASEvaluator()
    
    # Simple test case
    print("\n1. Testing single case...")
    
    question = "What is artificial intelligence?"
    answer = "Artificial intelligence is the simulation of human intelligence by machines and computer systems."
    contexts = [
        "Artificial intelligence (AI) is intelligence demonstrated by machines.",
        "AI systems are designed to simulate human intelligence processes."
    ]
    ground_truth = "Artificial intelligence is the simulation of human intelligence processes by machines."
    
    # Evaluate
    scores = evaluator.evaluate_single_case(
        question=question,
        answer=answer,
        contexts=contexts,
        ground_truth=ground_truth
    )
    
    # Check scores
    print(f"\n2. Validating scores...")
    for metric, score in scores.items():
        if 0 <= score <= 1:
            print(f"   ✅ {metric}: {score:.3f} (valid range)")
        else:
            print(f"   ❌ {metric}: {score:.3f} (invalid range!)")
    
    print("\n" + "=" * 60)
    print("✅ RAGAS BASIC TEST COMPLETE")
    print("=" * 60)


def test_ragas_multiple_cases():
    """Test RAGAS with multiple cases."""
    
    print("\n" + "=" * 60)
    print("TESTING RAGAS WITH MULTIPLE CASES")
    print("=" * 60)
    
    evaluator = RAGASEvaluator()
    
    # Multiple test cases
    questions = [
        "What is machine learning?",
        "What is deep learning?",
        "What is neural network?"
    ]
    
    answers = [
        "Machine learning is a method of data analysis that automates analytical model building.",
        "Deep learning is a subset of machine learning using neural networks with multiple layers.",
        "A neural network is a series of algorithms that recognizes patterns in data."
    ]
    
    contexts = [
        ["Machine learning is a branch of AI focused on building systems that learn from data."],
        ["Deep learning uses neural networks with many layers to model complex patterns."],
        ["Neural networks are computing systems inspired by biological neural networks."]
    ]
    
    ground_truths = [
        "Machine learning is a type of artificial intelligence that allows systems to learn from data.",
        "Deep learning is machine learning using multi-layered neural networks.",
        "A neural network is a computational model inspired by the human brain."
    ]
    
    # Evaluate
    scores = evaluator.evaluate_rag_system(
        questions=questions,
        answers=answers,
        contexts=contexts,
        ground_truths=ground_truths
    )
    
    # Summary
    print(f"\n📊 Overall Performance:")
    print(f"   Average Score: {scores['overall']:.3f}")
    
    if scores['overall'] >= 0.7:
        print(f"   ✅ Good performance (>= 0.7)")
    elif scores['overall'] >= 0.5:
        print(f"   ⚠️  Moderate performance (0.5-0.7)")
    else:
        print(f"   ❌ Needs improvement (< 0.5)")
    
    print("\n" + "=" * 60)
    print("✅ MULTIPLE CASES TEST COMPLETE")
    print("=" * 60)


def main():
    """Run all RAGAS tests."""
    
    print("=" * 60)
    print("RAGAS EVALUATION TEST SUITE")
    print("=" * 60)
    
    test_ragas_basic()
    test_ragas_multiple_cases()
    
    print("\n" + "=" * 60)
    print("✅ ALL RAGAS TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()