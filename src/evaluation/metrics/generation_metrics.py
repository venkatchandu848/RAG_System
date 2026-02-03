"""Generation quality metrics for RAG evaluation.

Implements metrics for evaluating generated answers:
- Faithfulness (factual consistency with retrieved context)
- Answer Relevancy (relevance to the query)
- BERTScore (semantic similarity)
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional

import numpy as np
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class GenerationResult(BaseModel):
    """Single generation result for evaluation."""

    query_id: str
    query: str
    generated_answer: str
    ground_truth_answer: str
    retrieved_contexts: List[str]


class GenerationMetrics:
    """Calculate generation quality metrics."""

    def __init__(self, ollama_client=None, embeddings_client=None):
        """Initialize generation metrics.

        Args:
            ollama_client: Optional Ollama client for LLM-based metrics
            embeddings_client: Optional embeddings client for semantic similarity
        """
        self.ollama = ollama_client
        self.embeddings = embeddings_client

    async def faithfulness(
        self, answer: str, contexts: List[str], use_llm: bool = True
    ) -> float:
        """Calculate faithfulness score.

        Measures if the generated answer is factually consistent with retrieved context.
        Uses LLM to verify each claim in the answer against the context.

        Args:
            answer: Generated answer
            contexts: Retrieved context chunks
            use_llm: Whether to use LLM for evaluation (fallback to simple overlap)

        Returns:
            Faithfulness score (0.0 to 1.0)
        """
        if not answer or not contexts:
            return 0.0

        if use_llm and self.ollama:
            return await self._faithfulness_llm(answer, contexts)
        else:
            return self._faithfulness_simple(answer, contexts)

    async def _faithfulness_llm(self, answer: str, contexts: List[str]) -> float:
        """LLM-based faithfulness evaluation.

        Args:
            answer: Generated answer
            contexts: Retrieved context chunks

        Returns:
            Faithfulness score
        """
        # Extract claims from answer
        claims = self._extract_claims(answer)

        if not claims:
            return 1.0  # No claims to verify

        combined_context = "\n\n".join(contexts)

        verified_claims = 0

        for claim in claims:
            # SIMPLIFIED PROMPT FOR 1B MODEL
            prompt = f"""Context: {combined_context[:1000]}

Statement: {claim}

Is the Statement supported by the Context?
Answer ONLY "yes" or "no"."""

            # Retry logic for robustness
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = await self.ollama.generate(
                        prompt=prompt,
                        model="llama3.2:1b",
                        system_prompt="You are a checker. Answer yes or no.",
                        temperature=0.0,
                    )

                    # Handle dictionary response from Ollama
                    response_text = response.get("response", "") if isinstance(response, dict) else str(response)

                    if "yes" in response_text.lower():
                        verified_claims += 1
                    
                    # Success, break retry loop
                    break

                except Exception as e:
                    if attempt < max_retries - 1:
                        # Exponential backoff: 2s, 4s, 8s
                        wait_time = 2 * (2 ** attempt)
                        logger.warning(f"Error in faithfulness check (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Failed to verify claim after {max_retries} attempts: {e}")
                        # If we fail, we essentially treat it as unverified (hallucination) which is the default behavior,
                        # but at least we tried multiple times.
                        continue

        return verified_claims / len(claims) if claims else 0.0

    def _faithfulness_simple(self, answer: str, contexts: List[str]) -> float:
        """Simple faithfulness based on token overlap.

        Args:
            answer: Generated answer
            contexts: Retrieved context chunks

        Returns:
            Faithfulness score
        """
        # Tokenize answer and context
        answer_tokens = set(self._tokenize(answer.lower()))
        context_tokens = set()

        for context in contexts:
            context_tokens.update(self._tokenize(context.lower()))

        if not answer_tokens:
            return 0.0

        # Calculate overlap
        overlap = len(answer_tokens & context_tokens)
        return overlap / len(answer_tokens)

    async def answer_relevancy(
        self, query: str, answer: str, use_embeddings: bool = True
    ) -> float:
        """Calculate answer relevancy score.

        Measures how well the answer addresses the query.
        Uses embeddings similarity or LLM evaluation.

        Args:
            query: User query
            answer: Generated answer
            use_embeddings: Whether to use embeddings (fallback to LLM)

        Returns:
            Relevancy score (0.0 to 1.0)
        """
        if not query or not answer:
            return 0.0

        if use_embeddings and self.embeddings:
            return await self._relevancy_embeddings(query, answer)
        elif self.ollama:
            return await self._relevancy_llm(query, answer)
        else:
            return self._relevancy_simple(query, answer)

    async def _relevancy_embeddings(self, query: str, answer: str) -> float:
        """Embedding-based relevancy evaluation.

        Args:
            query: User query
            answer: Generated answer

        Returns:
            Relevancy score
        """
        try:
            # Get embeddings
            query_embedding = await self.embeddings.embed_text(query)
            answer_embedding = await self.embeddings.embed_text(answer)

            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, answer_embedding)

            # Normalize to 0-1 range (cosine is -1 to 1)
            return (similarity + 1) / 2

        except Exception as e:
            logger.error(f"Error in embeddings-based relevancy: {e}")
            return 0.0

    async def _relevancy_llm(self, query: str, answer: str) -> float:
        """LLM-based relevancy evaluation.

        Args:
            query: User query
            answer: Generated answer

        Returns:
            Relevancy score
        """
        # SIMPLIFIED PROMPT FOR 1B MODEL
        prompt = f"""Question: {query}
Answer: {answer}

Is the answer relevant to the question?
Reply with a score from 0 to 10.
Score:"""

        try:
            response = await self.ollama.generate(
                prompt=prompt,
                model="llama3.2:1b",
                system_prompt="You are a scorer. Reply with a number only.",
                temperature=0.0,
            )

            # Extract number from response
            score_match = re.search(r"\d+", response)
            if score_match:
                score = int(score_match.group())
                return min(score / 10.0, 1.0)

        except Exception as e:
            logger.error(f"Error in LLM-based relevancy: {e}")

        return 0.0

    def _relevancy_simple(self, query: str, answer: str) -> float:
        """Simple relevancy based on keyword overlap.

        Args:
            query: User query
            answer: Generated answer

        Returns:
            Relevancy score
        """
        query_tokens = set(self._tokenize(query.lower()))
        answer_tokens = set(self._tokenize(answer.lower()))

        if not query_tokens:
            return 0.0

        overlap = len(query_tokens & answer_tokens)
        return overlap / len(query_tokens)

    async def bert_score(
        self, generated: str, reference: str, use_embeddings: bool = True
    ) -> Dict[str, float]:
        """Calculate BERTScore F1.

        Measures semantic similarity between generated and reference answers.

        Args:
            generated: Generated answer
            reference: Reference (ground truth) answer
            use_embeddings: Whether to use embeddings

        Returns:
            Dictionary with precision, recall, f1 scores
        """
        if not generated or not reference:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        if use_embeddings and self.embeddings:
            return await self._bert_score_embeddings(generated, reference)
        else:
            return self._bert_score_simple(generated, reference)

    async def _bert_score_embeddings(
        self, generated: str, reference: str
    ) -> Dict[str, float]:
        """Embedding-based BERTScore approximation.

        Args:
            generated: Generated answer
            reference: Reference answer

        Returns:
            Precision, recall, F1 scores
        """
        try:
            gen_embedding = await self.embeddings.embed_text(generated)
            ref_embedding = await self.embeddings.embed_text(reference)

            similarity = self._cosine_similarity(gen_embedding, ref_embedding)
            # Normalize to 0-1
            score = (similarity + 1) / 2

            # For simplicity, use same score for P, R, F1
            return {"precision": score, "recall": score, "f1": score}

        except Exception as e:
            logger.error(f"Error in embedding-based BERTScore: {e}")
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def _bert_score_simple(self, generated: str, reference: str) -> Dict[str, float]:
        """Simple token-based BERTScore approximation.

        Args:
            generated: Generated answer
            reference: Reference answer

        Returns:
            Precision, recall, F1 scores
        """
        gen_tokens = set(self._tokenize(generated.lower()))
        ref_tokens = set(self._tokenize(reference.lower()))

        if not gen_tokens or not ref_tokens:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        overlap = len(gen_tokens & ref_tokens)

        precision = overlap / len(gen_tokens)
        recall = overlap / len(ref_tokens)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        return {"precision": precision, "recall": recall, "f1": f1}

    async def evaluate_batch(self, results: List[GenerationResult]) -> Dict[str, float]:
        """Evaluate a batch of generation results.

        Args:
            results: List of generation results

        Returns:
            Dictionary of metric names to scores
        """
        faithfulness_scores = []
        relevancy_scores = []
        bert_scores = []

        for result in results:
            # Faithfulness
            faith_score = await self.faithfulness(
                result.generated_answer, result.retrieved_contexts
            )
            faithfulness_scores.append(faith_score)

            # Relevancy
            rel_score = await self.answer_relevancy(result.query, result.generated_answer)
            relevancy_scores.append(rel_score)

            # BERTScore
            bert = await self.bert_score(
                result.generated_answer, result.ground_truth_answer
            )
            bert_scores.append(bert["f1"])

        metrics = {
            "faithfulness": np.mean(faithfulness_scores),
            "answer_relevancy": np.mean(relevancy_scores),
            "bertscore_f1": np.mean(bert_scores),
            "num_queries": len(results),
        }

        logger.info(f"Evaluated {len(results)} generations")
        logger.info(f"Metrics: {metrics}")

        return metrics

    # Helper methods
    @staticmethod
    def _extract_claims(text: str) -> List[str]:
        """Extract claims (sentences) from text.

        Args:
            text: Input text

        Returns:
            List of claims
        """
        # Simple sentence splitting
        sentences = re.split(r"[.!?]+", text)
        claims = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        return claims

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple tokenization.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        # Remove punctuation and split
        text = re.sub(r"[^\w\s]", " ", text)
        return [t for t in text.split() if len(t) > 2]

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity
        """
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)

        dot_product = np.dot(vec1_np, vec2_np)
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)
