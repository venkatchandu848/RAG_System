"""RAGAS framework integration for RAG evaluation.

Integrates with RAGAS (Retrieval-Augmented Generation Assessment) framework
for comprehensive RAG system evaluation.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
from pydantic import BaseModel

from src.evaluation.metrics.generation_metrics import GenerationMetrics
from src.evaluation.metrics.retrieval_metrics import RetrievalMetrics

logger = logging.getLogger(__name__)


class RAGASResult(BaseModel):
    """RAGAS evaluation result."""

    query_id: str
    query: str
    contexts: List[str]
    answer: str
    ground_truth: str
    
    # Scores
    context_precision: float = 0.0
    context_recall: float = 0.0
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    
    # Metadata
    metadata: Dict = {}


class RAGASEvaluator:
    """RAGAS-compatible evaluator for RAG systems.

    Implements key RAGAS metrics:
    - Context Precision: Precision of retrieved contexts
    - Context Recall: Recall of retrieved contexts
    - Faithfulness: Factual consistency of answer with context
    - Answer Relevancy: Relevance of answer to query
    """

    def __init__(
        self,
        retrieval_metrics: Optional[RetrievalMetrics] = None,
        generation_metrics: Optional[GenerationMetrics] = None,
        ollama_client=None,
        embeddings_client=None,
    ):
        """Initialize RAGAS evaluator.

        Args:
            retrieval_metrics: Retrieval metrics calculator
            generation_metrics: Generation metrics calculator
            ollama_client: Ollama client for LLM-based evaluation
            embeddings_client: Embeddings client for semantic similarity
        """
        self.retrieval_metrics = retrieval_metrics or RetrievalMetrics()
        self.generation_metrics = generation_metrics or GenerationMetrics(
            ollama_client=ollama_client,
            embeddings_client=embeddings_client,
        )

        logger.info("Initialized RAGAS evaluator")

    async def context_precision(
        self,
        query: str,
        contexts: List[str],
        ground_truth: str,
    ) -> float:
        """Calculate context precision.

        Measures how many of the retrieved contexts are actually relevant.
        Uses LLM to determine relevance of each context.

        Args:
            query: User query
            contexts: Retrieved context chunks
            ground_truth: Ground truth answer

        Returns:
            Context precision score (0.0 to 1.0)
        """
        if not contexts:
            logger.warning("RAGAS Context Precision: No contexts retrieved, returning 0.0")
            return 0.0

        if not self.generation_metrics.ollama:
            logger.info("RAGAS Context Precision: No Ollama client, falling back to simple keyword matching")
            # Fallback: simple keyword matching
            return self._context_precision_simple(query, contexts, ground_truth)

        relevant_count = 0

        for i, context in enumerate(contexts):
            # Ask LLM if context is relevant to query
            prompt = f"""Query: {query}
Context: {context[:500]}
Is this context relevant to the query? Answer "yes" or "no"."""

            try:
                response = await self.generation_metrics.ollama.generate(
                    prompt=prompt,
                    model="llama3.2:1b",
                    system_prompt="Evaluating context relevance. Answer yes/no.",
                    temperature=0.0,
                )

                # Handle dictionary response from Ollama
                response_text = response.get("response", "") if isinstance(response, dict) else str(response)

                is_relevant = "yes" in response_text.lower()
                if is_relevant:
                    relevant_count += 1
                else:
                    logger.debug(f"Context {i} judged irrelevant by LLM. Resp: {response_text[:30]}...")

            except Exception as e:
                logger.error(f"Error in context precision evaluation (ctx {i}): {e}")
                continue

        precision = relevant_count / len(contexts)
        logger.info(f"RAGAS Context Precision: {precision:.2f} ({relevant_count}/{len(contexts)} relevant)")
        return precision

    def _context_precision_simple(
        self, query: str, contexts: List[str], ground_truth: str
    ) -> float:
        """Simple context precision based on keyword overlap.

        Args:
            query: User query
            contexts: Retrieved contexts
            ground_truth: Ground truth answer

        Returns:
            Context precision score
        """
        query_tokens = set(self._tokenize(query.lower()))
        truth_tokens = set(self._tokenize(ground_truth.lower()))
        relevant_tokens = query_tokens | truth_tokens

        if not relevant_tokens:
            return 0.0

        relevant_count = 0

        for context in contexts:
            context_tokens = set(self._tokenize(context.lower()))
            overlap = len(context_tokens & relevant_tokens)

            # Consider context relevant if it has significant overlap
            if overlap / len(relevant_tokens) > 0.1:
                relevant_count += 1

        return relevant_count / len(contexts) if contexts else 0.0

    async def context_recall(
        self,
        contexts: List[str],
        ground_truth: str,
    ) -> float:
        """Calculate context recall.

        Measures what portion of ground truth information is present in retrieved contexts.

        Args:
            contexts: Retrieved context chunks
            ground_truth: Ground truth answer

        Returns:
            Context recall score (0.0 to 1.0)
        """
        if not ground_truth or not contexts:
            logger.warning("RAGAS Context Recall: Missing ground_truth or contexts, returning 0.0")
            return 0.0

        # Extract key information from ground truth
        truth_statements = self._extract_statements(ground_truth)

        if not truth_statements:
            logger.warning("RAGAS Context Recall: Could not extract statements from ground truth. Returning 1.0 (default)")
            return 1.0  # No statements to verify

        combined_context = " ".join(contexts)
        found_count = 0

        for statement in truth_statements:
            # Check if statement information is in contexts
            if self.generation_metrics.ollama:
                found = await self._verify_statement_in_context(
                    statement, combined_context
                )
            else:
                logger.info("RAGAS Context Recall: No Ollama, using simple token verification")
                found = self._verify_statement_simple(statement, combined_context)

            if found:
                found_count += 1
            else:
                logger.debug(f"Statement not found in context used for Recall: '{statement[:50]}...'")

        recall = found_count / len(truth_statements)
        logger.info(f"RAGAS Context Recall: {recall:.2f} ({found_count}/{len(truth_statements)} statements found)")
        return recall

    async def _verify_statement_in_context(
        self, statement: str, context: str
    ) -> bool:
        """Verify if statement is supported by context using LLM.

        Args:
            statement: Statement to verify
            context: Combined context

        Returns:
            True if statement is supported
        """
        prompt = f"""Context: {context[:1000]}
Statement: {statement}
Is this statement supported by the context? Answer "yes" or "no"."""

        try:
            response = await self.generation_metrics.ollama.generate(
                prompt=prompt,
                model="llama3.2:1b",
                system_prompt="Fact checker. Answer yes/no.",
                temperature=0.0,
            )

            # Handle dictionary response from Ollama
            response_text = response.get("response", "") if isinstance(response, dict) else str(response)

            result = "yes" in response_text.lower()
            if not result:
                logger.debug(f"LLM could not verify statement: {statement[:30]}... Response: {response_text}")
            return result

        except Exception as e:
            logger.error(f"Error in statement verification: {e}")
            return False

    def _verify_statement_simple(self, statement: str, context: str) -> bool:
        """Simple statement verification using token overlap.

        Args:
            statement: Statement to verify
            context: Combined context

        Returns:
            True if statement is likely supported
        """
        statement_tokens = set(self._tokenize(statement.lower()))
        context_tokens = set(self._tokenize(context.lower()))

        if not statement_tokens:
            return False

        overlap = len(statement_tokens & context_tokens)
        # Consider supported if >50% of statement tokens are in context
        return overlap / len(statement_tokens) > 0.5

    async def evaluate_single(
        self,
        query: str,
        contexts: List[str],
        answer: str,
        ground_truth: str,
        query_id: str = "single",
    ) -> RAGASResult:
        """Evaluate a single RAG response.

        Args:
            query: User query
            contexts: Retrieved contexts
            answer: Generated answer
            ground_truth: Ground truth answer
            query_id: Query identifier

        Returns:
            RAGAS evaluation result
        """
        # Calculate metrics
        context_prec = await self.context_precision(query, contexts, ground_truth)
        context_rec = await self.context_recall(contexts, ground_truth)
        faith = await self.generation_metrics.faithfulness(answer, contexts)
        relevancy = await self.generation_metrics.answer_relevancy(query, answer)

        return RAGASResult(
            query_id=query_id,
            query=query,
            contexts=contexts,
            answer=answer,
            ground_truth=ground_truth,
            context_precision=context_prec,
            context_recall=context_rec,
            faithfulness=faith,
            answer_relevancy=relevancy,
        )

    async def evaluate_batch(
        self, results: List[Dict]
    ) -> Dict[str, float]:
        """Evaluate a batch of RAG results.

        Args:
            results: List of dicts with keys: query, contexts, answer, ground_truth

        Returns:
            Dictionary of average RAGAS metrics
        """
        ragas_results = []

        for i, result in enumerate(results):
            ragas_result = await self.evaluate_single(
                query=result["query"],
                contexts=result["contexts"],
                answer=result["answer"],
                ground_truth=result["ground_truth"],
                query_id=result.get("query_id", f"query_{i}"),
            )
            ragas_results.append(ragas_result)

        # Calculate averages
        metrics = {
            "context_precision": np.mean([r.context_precision for r in ragas_results]),
            "context_recall": np.mean([r.context_recall for r in ragas_results]),
            "faithfulness": np.mean([r.faithfulness for r in ragas_results]),
            "answer_relevancy": np.mean([r.answer_relevancy for r in ragas_results]),
        }

        # Calculate composite RAGAS score (harmonic mean of all metrics)
        metric_values = list(metrics.values())
        if all(v > 0 for v in metric_values):
            harmonic_mean = len(metric_values) / sum(1 / v for v in metric_values)
            metrics["ragas_score"] = harmonic_mean
        else:
            metrics["ragas_score"] = 0.0

        metrics["num_queries"] = len(results)

        logger.info(f"Evaluated {len(results)} RAG responses")
        logger.info(f"RAGAS Metrics: {metrics}")

        return metrics

    # Helper methods
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple tokenization.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        import re

        text = re.sub(r"[^\w\s]", " ", text)
        return [t for t in text.split() if len(t) > 2]

    @staticmethod
    def _extract_statements(text: str) -> List[str]:
        """Extract statements from text.

        Args:
            text: Input text

        Returns:
            List of statements
        """
        import re

        sentences = re.split(r"[.!?]+", text)
        statements = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        return statements
