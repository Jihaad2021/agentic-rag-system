"""
Simple evaluation metrics without RAGAS.
Manual implementation of basic quality metrics.
"""

from typing import List, Dict, Any
import numpy as np
from difflib import SequenceMatcher


class SimpleEvaluator:
    """
    Simple evaluation metrics for RAG system.
    
    Metrics:
    - Answer Relevancy: Keyword overlap with question
    - Answer Quality: Length and completeness
    - Context Relevance: Keyword overlap with contexts
    - Similarity: String similarity with ground truth
    """
    
    def __init__(self):
        """Initialize simple evaluator."""
        print("📊 Simple Evaluator initialized")
    
    def evaluate_answer_relevancy(
        self,
        question: str,
        answer: str
    ) -> float:
        """
        Measure answer relevancy to question.
        Based on keyword overlap.
        
        Args:
            question: User query
            answer: Generated answer
            
        Returns:
            Score 0.0-1.0
        """
        # Extract keywords (simple: just words)
        q_words = set(question.lower().split())
        a_words = set(answer.lower().split())
        
        # Remove common words
        stop_words = {'is', 'the', 'a', 'an', 'what', 'how', 'why', 'when', 'where'}
        q_words = q_words - stop_words
        a_words = a_words - stop_words
        
        if not q_words:
            return 0.5
        
        # Calculate overlap
        overlap = len(q_words & a_words)
        score = overlap / len(q_words)
        
        return min(score, 1.0)
    
    def evaluate_faithfulness(
        self,
        answer: str,
        contexts: List[str]
    ) -> float:
        """
        Measure if answer is grounded in contexts.
        Based on word overlap with contexts.
        
        Args:
            answer: Generated answer
            contexts: Retrieved contexts
            
        Returns:
            Score 0.0-1.0
        """
        if not contexts:
            return 0.0
        
        answer_words = set(answer.lower().split())
        context_words = set()
        
        for context in contexts:
            context_words.update(context.lower().split())
        
        if not answer_words:
            return 0.0
        
        # Calculate how many answer words appear in contexts
        overlap = len(answer_words & context_words)
        score = overlap / len(answer_words)
        
        return min(score, 1.0)
    
    def evaluate_similarity(
        self,
        answer: str,
        ground_truth: str
    ) -> float:
        """
        Measure similarity with ground truth.
        Using sequence matcher.
        
        Args:
            answer: Generated answer
            ground_truth: Expected answer
            
        Returns:
            Score 0.0-1.0
        """
        return SequenceMatcher(None, answer.lower(), ground_truth.lower()).ratio()
    
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
            contexts: List of retrieved contexts
            ground_truths: List of expected answers
            
        Returns:
            Dictionary of metric scores
        """
        print(f"\n📊 Evaluating {len(questions)} test cases...")
        
        relevancy_scores = []
        faithfulness_scores = []
        similarity_scores = []
        
        for q, a, c, gt in zip(questions, answers, contexts, ground_truths):
            relevancy_scores.append(self.evaluate_answer_relevancy(q, a))
            faithfulness_scores.append(self.evaluate_faithfulness(a, c))
            similarity_scores.append(self.evaluate_similarity(a, gt))
        
        scores = {
            'answer_relevancy': np.mean(relevancy_scores),
            'faithfulness': np.mean(faithfulness_scores),
            'similarity': np.mean(similarity_scores),
            'overall': np.mean([
                np.mean(relevancy_scores),
                np.mean(faithfulness_scores),
                np.mean(similarity_scores)
            ])
        }
        
        print(f"\n   ✅ Evaluation complete!")
        print(f"      Answer Relevancy: {scores['answer_relevancy']:.3f}")
        print(f"      Faithfulness: {scores['faithfulness']:.3f}")
        print(f"      Similarity: {scores['similarity']:.3f}")
        print(f"      Overall Score: {scores['overall']:.3f}")
        
        return scores