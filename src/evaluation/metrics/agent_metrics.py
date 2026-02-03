"""Agentic RAG-specific metrics.

Metrics for evaluating agentic RAG system performance:
- Guardrail precision (false positive rate)
- Query rewrite effectiveness
- Adaptive retrieval success rate
- Decision quality metrics
"""

import logging
from typing import Dict, List, Optional

import numpy as np
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class AgentDecision(BaseModel):
    """Single agent decision for evaluation."""

    query_id: str
    query: str
    
    # Guardrail
    guardrail_decision: str  # "proceed" or "out_of_scope"
    guardrail_correct: bool  # Was the decision correct?
    
    # Retrieval attempts
    retrieval_attempts: int
    final_success: bool  # Did retrieval ultimately succeed?
    
    # Query rewriting
    original_query: str
    rewritten_query: Optional[str] = None
    rewrite_improved: bool = False  # Did rewrite improve results?
    
    # Document grading
    initial_relevant_count: int = 0
    final_relevant_count: int = 0
    
    # Metadata
    reasoning_steps: List[str] = []
    metadata: Dict = {}


class AgentMetrics:
    """Calculate agentic RAG-specific metrics."""

    @staticmethod
    def guardrail_precision(decisions: List[AgentDecision]) -> Dict[str, float]:
        """Calculate guardrail performance metrics.

        Measures how accurately the guardrail identifies out-of-scope queries.

        Args:
            decisions: List of agent decisions with guardrail annotations

        Returns:
            Dictionary with precision, recall, F1, and false positive rate
        """
        if not decisions:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "false_positive_rate": 0.0,
                "accuracy": 0.0,
            }

        # True positives: correctly identified out-of-scope
        tp = sum(
            1
            for d in decisions
            if d.guardrail_decision == "out_of_scope" and d.guardrail_correct
        )

        # False positives: incorrectly flagged as out-of-scope
        fp = sum(
            1
            for d in decisions
            if d.guardrail_decision == "out_of_scope" and not d.guardrail_correct
        )

        # True negatives: correctly identified in-scope
        tn = sum(
            1
            for d in decisions
            if d.guardrail_decision == "proceed" and d.guardrail_correct
        )

        # False negatives: missed out-of-scope queries
        fn = sum(
            1
            for d in decisions
            if d.guardrail_decision == "proceed" and not d.guardrail_correct
        )

        total = tp + fp + tn + fn

        # Precision: Of all flagged queries, how many were truly out-of-scope?
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        # Recall: Of all out-of-scope queries, how many did we catch?
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # F1 score
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # False positive rate: Of all in-scope queries, how many were incorrectly flagged?
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        # Overall accuracy
        accuracy = (tp + tn) / total if total > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positive_rate": fpr,
            "accuracy": accuracy,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
        }

    @staticmethod
    def query_rewrite_effectiveness(decisions: List[AgentDecision]) -> Dict[str, float]:
        """Calculate query rewrite effectiveness.

        Measures how often query rewriting improves retrieval results.

        Args:
            decisions: List of agent decisions with rewrite information

        Returns:
            Dictionary with rewrite metrics
        """
        # Filter to decisions where rewriting occurred
        rewritten = [d for d in decisions if d.rewritten_query is not None]

        if not rewritten:
            return {
                "rewrite_rate": 0.0,
                "success_rate": 0.0,
                "improvement_rate": 0.0,
                "avg_improvement": 0.0,
            }

        # How often was rewriting attempted?
        rewrite_rate = len(rewritten) / len(decisions) if decisions else 0.0

        # How often did rewriting lead to success?
        success_rate = (
            sum(1 for d in rewritten if d.final_success) / len(rewritten)
            if rewritten
            else 0.0
        )

        # How often did rewriting improve results?
        improvement_rate = (
            sum(1 for d in rewritten if d.rewrite_improved) / len(rewritten)
            if rewritten
            else 0.0
        )

        # Average improvement in relevant documents
        improvements = [
            d.final_relevant_count - d.initial_relevant_count
            for d in rewritten
            if d.final_relevant_count > 0 or d.initial_relevant_count > 0
        ]

        avg_improvement = np.mean(improvements) if improvements else 0.0

        return {
            "rewrite_rate": rewrite_rate,
            "success_rate": success_rate,
            "improvement_rate": improvement_rate,
            "avg_improvement": avg_improvement,
            "num_rewrites": len(rewritten),
        }

    @staticmethod
    def adaptive_retrieval_success(decisions: List[AgentDecision]) -> Dict[str, float]:
        """Calculate adaptive retrieval success metrics.

        Measures how well the multi-attempt retrieval strategy works.

        Args:
            decisions: List of agent decisions with retrieval information

        Returns:
            Dictionary with retrieval success metrics
        """
        if not decisions:
            return {
                "overall_success_rate": 0.0,
                "first_attempt_success": 0.0,
                "second_attempt_success": 0.0,
                "avg_attempts": 0.0,
            }

        # Overall success rate
        overall_success = sum(1 for d in decisions if d.final_success) / len(decisions)

        # First attempt success
        first_attempt = [d for d in decisions if d.retrieval_attempts >= 1]
        first_success = (
            sum(1 for d in first_attempt if d.retrieval_attempts == 1 and d.final_success)
            / len(first_attempt)
            if first_attempt
            else 0.0
        )

        # Second attempt success (after first failed)
        second_attempt = [d for d in decisions if d.retrieval_attempts >= 2]
        second_success = (
            sum(1 for d in second_attempt if d.final_success) / len(second_attempt)
            if second_attempt
            else 0.0
        )

        # Average number of attempts
        avg_attempts = np.mean([d.retrieval_attempts for d in decisions])

        return {
            "overall_success_rate": overall_success,
            "first_attempt_success": first_success,
            "second_attempt_success": second_success,
            "avg_attempts": avg_attempts,
            "multi_attempt_needed": len(second_attempt) / len(decisions),
        }

    @staticmethod
    def decision_quality(decisions: List[AgentDecision]) -> Dict[str, float]:
        """Calculate overall decision quality metrics.

        Combines multiple aspects of agent decision-making.

        Args:
            decisions: List of agent decisions

        Returns:
            Dictionary with decision quality metrics
        """
        # Document grading accuracy
        grading_scores = []
        for d in decisions:
            if d.initial_relevant_count > 0 or d.final_relevant_count > 0:
                # Quality: ratio of final relevant to total attempts
                quality = d.final_relevant_count / max(d.retrieval_attempts, 1)
                grading_scores.append(quality)

        avg_grading_quality = np.mean(grading_scores) if grading_scores else 0.0

        # Reasoning transparency (number of reasoning steps)
        avg_reasoning_steps = np.mean(
            [len(d.reasoning_steps) for d in decisions if d.reasoning_steps]
        )

        # End-to-end success (guardrail + retrieval + grading)
        e2e_success = sum(
            1
            for d in decisions
            if d.guardrail_correct and d.final_success
        ) / len(decisions)

        return {
            "avg_grading_quality": avg_grading_quality,
            "avg_reasoning_steps": avg_reasoning_steps,
            "end_to_end_success": e2e_success,
        }

    @staticmethod
    def evaluate_batch(decisions: List[AgentDecision]) -> Dict[str, float]:
        """Evaluate a batch of agent decisions.

        Args:
            decisions: List of agent decisions

        Returns:
            Comprehensive metrics dictionary
        """
        metrics = {}

        # Guardrail metrics
        guardrail = AgentMetrics.guardrail_precision(decisions)
        metrics.update({f"guardrail_{k}": v for k, v in guardrail.items()})

        # Query rewrite metrics
        rewrite = AgentMetrics.query_rewrite_effectiveness(decisions)
        metrics.update({f"rewrite_{k}": v for k, v in rewrite.items()})

        # Adaptive retrieval metrics
        retrieval = AgentMetrics.adaptive_retrieval_success(decisions)
        metrics.update({f"retrieval_{k}": v for k, v in retrieval.items()})

        # Decision quality metrics
        quality = AgentMetrics.decision_quality(decisions)
        metrics.update({f"decision_{k}": v for k, v in quality.items()})

        metrics["num_queries"] = len(decisions)

        logger.info(f"Evaluated {len(decisions)} agent decisions")
        logger.info(f"Agent Metrics: {metrics}")

        return metrics

    @staticmethod
    def generate_summary(metrics: Dict[str, float]) -> str:
        """Generate human-readable summary of agent metrics.

        Args:
            metrics: Dictionary of agent metrics

        Returns:
            Formatted summary string
        """
        summary = "Agentic RAG Performance Summary\n"
        summary += "=" * 50 + "\n\n"

        # Guardrail performance
        summary += "Guardrail Performance:\n"
        summary += f"  - Precision: {metrics.get('guardrail_precision', 0):.2%}\n"
        summary += f"  - False Positive Rate: {metrics.get('guardrail_false_positive_rate', 0):.2%}\n"
        summary += f"  - Accuracy: {metrics.get('guardrail_accuracy', 0):.2%}\n\n"

        # Query rewriting
        summary += "Query Rewrite Effectiveness:\n"
        summary += f"  - Rewrite Rate: {metrics.get('rewrite_rewrite_rate', 0):.2%}\n"
        summary += f"  - Success After Rewrite: {metrics.get('rewrite_success_rate', 0):.2%}\n"
        summary += f"  - Improvement Rate: {metrics.get('rewrite_improvement_rate', 0):.2%}\n\n"

        # Adaptive retrieval
        summary += "Adaptive Retrieval:\n"
        summary += f"  - Overall Success: {metrics.get('retrieval_overall_success_rate', 0):.2%}\n"
        summary += f"  - First Attempt Success: {metrics.get('retrieval_first_attempt_success', 0):.2%}\n"
        summary += f"  - Second Attempt Success: {metrics.get('retrieval_second_attempt_success', 0):.2%}\n"
        summary += f"  - Avg Attempts: {metrics.get('retrieval_avg_attempts', 0):.2f}\n\n"

        # Decision quality
        summary += "Decision Quality:\n"
        summary += f"  - Grading Quality: {metrics.get('decision_avg_grading_quality', 0):.2f}\n"
        summary += f"  - End-to-End Success: {metrics.get('decision_end_to_end_success', 0):.2%}\n\n"

        summary += f"Evaluated: {metrics.get('num_queries', 0)} queries\n"

        return summary
