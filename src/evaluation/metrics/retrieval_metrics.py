"""Retrieval metrics for RAG evaluation.

Implements standard information retrieval metrics:
- Precision@K
- Recall@K  
- nDCG (Normalized Discounted Cumulative Gain)
- MRR (Mean Reciprocal Rank)
"""

import logging
import math
from typing import Dict, List, Set

import numpy as np
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class RetrievalResult(BaseModel):
    """Single retrieval result for evaluation."""

    query_id: str
    retrieved_doc_ids: List[str]  # Ranked list of retrieved document IDs
    relevant_doc_ids: List[str]  # Ground truth relevant document IDs
    scores: List[float] = []  # Optional relevance scores


class RetrievalMetrics:
    """Calculate retrieval performance metrics."""

    @staticmethod
    def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        """Calculate Precision@K.

        Precision@K = (# relevant documents in top-k) / k

        Args:
            retrieved: Ranked list of retrieved document IDs
            relevant: List of ground truth relevant document IDs
            k: Number of top results to consider

        Returns:
            Precision@K score (0.0 to 1.0)
        """
        if k <= 0 or len(retrieved) == 0:
            return 0.0

        retrieved_at_k = set(retrieved[:k])
        relevant_set = set(relevant)

        num_relevant_retrieved = len(retrieved_at_k & relevant_set)
        return num_relevant_retrieved / k

    @staticmethod
    def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        """Calculate Recall@K.

        Recall@K = (# relevant documents in top-k) / (total # relevant documents)

        Args:
            retrieved: Ranked list of retrieved document IDs
            relevant: List of ground truth relevant document IDs
            k: Number of top results to consider

        Returns:
            Recall@K score (0.0 to 1.0)
        """
        if len(relevant) == 0:
            return 0.0

        if k <= 0 or len(retrieved) == 0:
            return 0.0

        retrieved_at_k = set(retrieved[:k])
        relevant_set = set(relevant)

        num_relevant_retrieved = len(retrieved_at_k & relevant_set)
        return num_relevant_retrieved / len(relevant)

    @staticmethod
    def dcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        """Calculate Discounted Cumulative Gain at K.

        DCG@K = Σ (relevance / log2(position + 1))

        Args:
            retrieved: Ranked list of retrieved document IDs
            relevant: List of ground truth relevant document IDs
            k: Number of top results to consider

        Returns:
            DCG@K score
        """
        if k <= 0 or len(retrieved) == 0:
            return 0.0

        relevant_set = set(relevant)
        dcg = 0.0

        for i, doc_id in enumerate(retrieved[:k]):
            if doc_id in relevant_set:
                # Binary relevance: 1 if relevant, 0 otherwise
                relevance = 1.0
                # Position is 1-indexed for log calculation
                position = i + 1
                dcg += relevance / math.log2(position + 1)

        return dcg

    @staticmethod
    def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain at K.

        nDCG@K = DCG@K / IDCG@K
        where IDCG is the ideal (maximum possible) DCG

        Args:
            retrieved: Ranked list of retrieved document IDs
            relevant: List of ground truth relevant document IDs
            k: Number of top results to consider

        Returns:
            nDCG@K score (0.0 to 1.0)
        """
        dcg = RetrievalMetrics.dcg_at_k(retrieved, relevant, k)

        # Calculate ideal DCG (all relevant docs ranked first)
        ideal_retrieved = list(relevant) + [
            doc for doc in retrieved if doc not in relevant
        ]
        idcg = RetrievalMetrics.dcg_at_k(ideal_retrieved, relevant, k)

        if idcg == 0.0:
            return 0.0

        return dcg / idcg

    @staticmethod
    def reciprocal_rank(retrieved: List[str], relevant: List[str]) -> float:
        """Calculate Reciprocal Rank.

        RR = 1 / (position of first relevant document)

        Args:
            retrieved: Ranked list of retrieved document IDs
            relevant: List of ground truth relevant document IDs

        Returns:
            Reciprocal rank score
        """
        if len(retrieved) == 0 or len(relevant) == 0:
            return 0.0

        relevant_set = set(relevant)

        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant_set:
                # Position is 1-indexed
                return 1.0 / (i + 1)

        return 0.0

    @staticmethod
    def mean_reciprocal_rank(results: List[RetrievalResult]) -> float:
        """Calculate Mean Reciprocal Rank across multiple queries.

        MRR = (1/N) * Σ RR_i

        Args:
            results: List of retrieval results for multiple queries

        Returns:
            Mean reciprocal rank score
        """
        if len(results) == 0:
            return 0.0

        rr_scores = [
            RetrievalMetrics.reciprocal_rank(r.retrieved_doc_ids, r.relevant_doc_ids)
            for r in results
        ]

        return np.mean(rr_scores)

    @staticmethod
    def evaluate_batch(
        results: List[RetrievalResult], k_values: List[int] = [5, 10]
    ) -> Dict[str, float]:
        """Evaluate a batch of retrieval results.

        Args:
            results: List of retrieval results
            k_values: List of K values to evaluate

        Returns:
            Dictionary of metric names to scores
        """
        metrics = {}

        for k in k_values:
            # Precision@K
            precision_scores = [
                RetrievalMetrics.precision_at_k(r.retrieved_doc_ids, r.relevant_doc_ids, k)
                for r in results
            ]
            metrics[f"precision@{k}"] = np.mean(precision_scores)

            # Recall@K
            recall_scores = [
                RetrievalMetrics.recall_at_k(r.retrieved_doc_ids, r.relevant_doc_ids, k)
                for r in results
            ]
            metrics[f"recall@{k}"] = np.mean(recall_scores)

            # nDCG@K
            ndcg_scores = [
                RetrievalMetrics.ndcg_at_k(r.retrieved_doc_ids, r.relevant_doc_ids, k)
                for r in results
            ]
            metrics[f"ndcg@{k}"] = np.mean(ndcg_scores)

        # MRR (not K-specific)
        metrics["mrr"] = RetrievalMetrics.mean_reciprocal_rank(results)

        # Add counts
        metrics["num_queries"] = len(results)

        logger.info(f"Evaluated {len(results)} queries")
        logger.info(f"Metrics: {metrics}")

        return metrics

    @staticmethod
    def evaluate_single(
        retrieved: List[str], relevant: List[str], k_values: List[int] = [5, 10]
    ) -> Dict[str, float]:
        """Evaluate a single retrieval result.

        Args:
            retrieved: Ranked list of retrieved document IDs
            relevant: List of ground truth relevant document IDs
            k_values: List of K values to evaluate

        Returns:
            Dictionary of metric names to scores
        """
        result = RetrievalResult(
            query_id="single",
            retrieved_doc_ids=retrieved,
            relevant_doc_ids=relevant,
        )

        metrics = {}

        for k in k_values:
            metrics[f"precision@{k}"] = RetrievalMetrics.precision_at_k(retrieved, relevant, k)
            metrics[f"recall@{k}"] = RetrievalMetrics.recall_at_k(retrieved, relevant, k)
            metrics[f"ndcg@{k}"] = RetrievalMetrics.ndcg_at_k(retrieved, relevant, k)

        metrics["rr"] = RetrievalMetrics.reciprocal_rank(retrieved, relevant)

        return metrics
