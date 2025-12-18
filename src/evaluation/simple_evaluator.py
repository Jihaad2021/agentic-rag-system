"""
Simple evaluation metrics without RAGAS.
Manual implementation of basic quality metrics.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from difflib import SequenceMatcher


class SimpleEvaluator:
    """Simple evaluation metrics for RAG system."""
    
    def __init__(self):
        print("📊 Simple Evaluator initialized")
    
    def evaluate_answer_relevancy(self, question: str, answer: str) -> float:
        """Measure answer relevancy to question."""
        q_words = set(question.lower().split())
        a_words = set(answer.lower().split())
        
        stop_words = {'is', 'the', 'a', 'an', 'what', 'how', 'why', 'when', 'where', 'who'}
        q_words = q_words - stop_words
        a_words = a_words - stop_words
        
        if not q_words:
            return 0.5
        
        overlap = len(q_words & a_words)
        score = overlap / len(q_words)
        return min(score, 1.0)
    
    def evaluate_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """Measure if answer is grounded in contexts."""
        if not contexts:
            return 0.0
        
        answer_words = set(answer.lower().split())
        context_words = set()
        
        for context in contexts:
            context_words.update(context.lower().split())
        
        if not answer_words:
            return 0.0
        
        overlap = len(answer_words & context_words)
        score = overlap / len(answer_words)
        return min(score, 1.0)
    
    def evaluate_similarity(self, answer: str, ground_truth: str) -> float:
        """Measure similarity with ground truth."""
        return SequenceMatcher(None, answer.lower(), ground_truth.lower()).ratio()
    
    def evaluate_completeness(self, answer: str) -> float:
        """Measure answer completeness based on length."""
        words = len(answer.split())
        # Ideal answer: 20-100 words
        if words < 20:
            return words / 20
        elif words > 100:
            return max(0.5, 1.0 - (words - 100) / 200)
        else:
            return 1.0
    
    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate single Q&A.
        
        Args:
            question: User query
            answer: Generated answer
            contexts: Retrieved contexts
            ground_truth: Expected answer (optional)
            
        Returns:
            Dictionary of scores
        """
        
        scores = {
            'relevancy': self.evaluate_answer_relevancy(question, answer),
            'faithfulness': self.evaluate_faithfulness(answer, contexts),
            'completeness': self.evaluate_completeness(answer)
        }
        
        # Only add similarity if ground_truth provided
        if ground_truth:
            scores['similarity'] = self.evaluate_similarity(answer, ground_truth)
        
        scores['overall'] = np.mean(list(scores.values()))
        
        return scores
    
    def evaluate_rag_system(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate multiple Q&A cases.
        
        Args:
            questions: List of queries
            answers: List of generated answers
            contexts: List of retrieved contexts (list of lists)
            ground_truths: List of expected answers (optional)
            
        Returns:
            Dictionary of average scores
        """
        
        print(f"\n📊 Evaluating {len(questions)} test cases...")
        
        all_scores = []
        for i, (q, a, c) in enumerate(zip(questions, answers, contexts)):
            gt = ground_truths[i] if ground_truths and i < len(ground_truths) else None
            scores = self.evaluate_single(q, a, c, gt)
            all_scores.append(scores)
        
        # Average scores across all test cases
        result = {}
        for key in all_scores[0].keys():
            result[key] = np.mean([s[key] for s in all_scores])
        
        print(f"\n   ✅ Evaluation complete!")
        for metric, score in result.items():
            print(f"      {metric.title()}: {score:.3f}")
        
        return result