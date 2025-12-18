"""
RAGAS-based evaluation for RAG system.
Measures: Answer Relevancy, Faithfulness, Context Precision, Context Recall
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import List, Dict, Any
import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_precision,
    context_recall
)


class RAGASEvaluator:
    """
    Evaluate RAG system using RAGAS framework.
    
    Metrics:
    - Answer Relevancy: How relevant is the answer to the question?
    - Faithfulness: Is the answer grounded in the context?
    - Context Precision: Are retrieved contexts relevant?
    - Context Recall: Did we retrieve all necessary contexts?
    """
    
    def __init__(self):
        """Initialize RAGAS evaluator."""
        self.metrics = [
            answer_relevancy,
            faithfulness,
            context_precision,
            context_recall
        ]
        print("📊 RAGAS Evaluator initialized")
        print(f"   Metrics: {len(self.metrics)}")
    
    def evaluate_rag_system(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate RAG system performance.
        
        Args:
            questions: List of queries
            answers: List of generated answers
            contexts: List of retrieved contexts (list of lists)
            ground_truths: List of expected answers
            
        Returns:
            Dictionary of metric scores
        """
        print(f"\n📊 Evaluating {len(questions)} test cases...")
        
        # Create dataset
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }
        
        dataset = Dataset.from_dict(data)
        
        # Evaluate
        print("   Running RAGAS evaluation...")
        results = evaluate(
            dataset,
            metrics=self.metrics
        )
        
        # Extract scores
        scores = {
            'answer_relevancy': results['answer_relevancy'],
            'faithfulness': results['faithfulness'],
            'context_precision': results['context_precision'],
            'context_recall': results['context_recall'],
            'overall': (
                results['answer_relevancy'] +
                results['faithfulness'] +
                results['context_precision'] +
                results['context_recall']
            ) / 4
        }
        
        print(f"\n   ✅ Evaluation complete!")
        print(f"      Answer Relevancy: {scores['answer_relevancy']:.3f}")
        print(f"      Faithfulness: {scores['faithfulness']:.3f}")
        print(f"      Context Precision: {scores['context_precision']:.3f}")
        print(f"      Context Recall: {scores['context_recall']:.3f}")
        print(f"      Overall Score: {scores['overall']:.3f}")
        
        return scores
    
    def load_test_dataset(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Load test dataset from JSON.
        
        Args:
            filepath: Path to test dataset JSON
            
        Returns:
            List of test cases
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return data['test_cases']
    
    def evaluate_single_case(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str
    ) -> Dict[str, float]:
        """
        Evaluate a single Q&A case.
        
        Args:
            question: Query
            answer: Generated answer
            contexts: Retrieved contexts
            ground_truth: Expected answer
            
        Returns:
            Dictionary of metric scores
        """
        return self.evaluate_rag_system(
            questions=[question],
            answers=[answer],
            contexts=[contexts],
            ground_truths=[ground_truth]
        )