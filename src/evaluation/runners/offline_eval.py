"""Offline batch evaluation runner.

Runs comprehensive evaluation on test dataset and generates reports.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel

from src.evaluation.datasets.synthetic_generator import GroundTruthQuery
from src.evaluation.metrics.agent_metrics import AgentDecision, AgentMetrics
from src.evaluation.metrics.generation_metrics import GenerationMetrics, GenerationResult
from src.evaluation.metrics.ragas_evaluator import RAGASEvaluator
from src.evaluation.metrics.retrieval_metrics import RetrievalMetrics, RetrievalResult

logger = logging.getLogger(__name__)


class EvaluationConfig(BaseModel):
    """Configuration for evaluation run."""

    # Dataset
    test_dataset_path: Path
    output_dir: Path

    # Evaluation settings
    k_values: List[int] = [5, 10]
    use_llm_metrics: bool = True
    use_embeddings_metrics: bool = True

    # System settings
    top_k: int = 5
    use_hybrid: bool = True
    model: str = "llama3.2:1b"

    # Metadata
    run_name: str = "evaluation_run"
    description: str = ""

    model_config = {"arbitrary_types_allowed": True}

    def model_dump(self, **kwargs):
        """Override model_dump to serialize Path objects as strings."""
        data = super().model_dump(**kwargs)
        # Convert Path objects to strings
        if isinstance(data.get('test_dataset_path'), Path):
            data['test_dataset_path'] = str(data['test_dataset_path'])
        if isinstance(data.get('output_dir'), Path):
            data['output_dir'] = str(data['output_dir'])
        return data

class SystemResponse(BaseModel):
    """RAG system response for a query."""

    query_id: str
    query: str

    # Retrieved documents
    retrieved_doc_ids: List[str]
    retrieved_contexts: List[str]

    # Generated answer
    generated_answer: str

    # Agent metadata (if agentic)
    is_agentic: bool = False
    guardrail_decision: Optional[str] = None
    retrieval_attempts: int = 1
    reasoning_steps: List[str] = []
    rewritten_query: Optional[str] = None


class EvaluationResults(BaseModel):
    """Complete evaluation results."""

    run_id: str
    timestamp: str
    config: EvaluationConfig

    # Metrics
    retrieval_metrics: Dict[str, float]
    generation_metrics: Dict[str, float]
    ragas_metrics: Dict[str, float]
    agent_metrics: Optional[Dict[str, float]] = None

    # Summary
    num_queries: int
    success_rate: float

    # Per-query details
    query_results: List[Dict] = []

    model_config = {"arbitrary_types_allowed": True}

    def model_dump(self, **kwargs):
        """Override model_dump to ensure Path objects in nested config are serialized."""
        data = super().model_dump(**kwargs)
        # Recursively convert Path objects to strings
        def convert_paths(obj):
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            return obj
        
        return convert_paths(data)


class OfflineEvaluationRunner:
    """Run offline evaluation on test dataset."""

    def __init__(
        self,
        rag_system,  # Your RAG system instance
        config: EvaluationConfig,
        ollama_client=None,
        embeddings_client=None,
    ):
        """Initialize evaluation runner.

        Args:
            rag_system: RAG system to evaluate
            config: Evaluation configuration
            ollama_client: Ollama client for LLM-based metrics
            embeddings_client: Embeddings client for semantic metrics
        """
        self.rag_system = rag_system
        self.config = config
        self.ollama = ollama_client
        self.embeddings = embeddings_client

        # Initialize metrics calculators
        self.retrieval_metrics = RetrievalMetrics()
        self.generation_metrics = GenerationMetrics(
            ollama_client=ollama_client,
            embeddings_client=embeddings_client,
        )
        self.ragas_evaluator = RAGASEvaluator(
            retrieval_metrics=self.retrieval_metrics,
            generation_metrics=self.generation_metrics,
            ollama_client=ollama_client,
            embeddings_client=embeddings_client,
        )
        self.agent_metrics = AgentMetrics()

        logger.info(f"Initialized OfflineEvaluationRunner: {config.run_name}")

    async def run_system(self, query: GroundTruthQuery) -> SystemResponse:
        """Run RAG system on a single query.

        Args:
            query: Ground truth query

        Returns:
            System response
        """
        # This is a placeholder - you'll need to adapt this to your actual RAG system API
        # Example for standard RAG:
        try:
            result = await self.rag_system.search_and_answer(
                query=query.question,
                top_k=self.config.top_k,
                use_hybrid=self.config.use_hybrid,
                model=self.config.model,
            )

            return SystemResponse(
                query_id=query.query_id,
                query=query.question,
                retrieved_doc_ids=result.get("doc_ids", []),
                retrieved_contexts=result.get("contexts", []),
                generated_answer=result.get("answer", ""),
                is_agentic=False,
            )

        except Exception as e:
            logger.error(f"Error running system for query {query.query_id}: {e}")
            return SystemResponse(
                query_id=query.query_id,
                query=query.question,
                retrieved_doc_ids=[],
                retrieved_contexts=[],
                generated_answer="",
            )

    async def run_agentic_system(self, query: GroundTruthQuery) -> SystemResponse:
        """Run agentic RAG system on a single query.

        Args:
            query: Ground truth query

        Returns:
            System response with agent metadata
        """
        # Placeholder for agentic system
        try:
            result = await self.rag_system.ask(query=query.question)

            # Extract doc_ids from sources if not present (AgenticRAGService returns 'sources')
            doc_ids = result.get("doc_ids", [])
            if not doc_ids and "sources" in result:
                doc_ids = [s.get("arxiv_id") for s in result["sources"] if s.get("arxiv_id")]

            # Extract contexts from sources if not present
            contexts = result.get("contexts", [])
            if not contexts and "sources" in result:
                contexts = [s.get("text", "") for s in result["sources"] if s.get("text")]

            return SystemResponse(
                query_id=query.query_id,
                query=query.question,
                retrieved_doc_ids=doc_ids,
                retrieved_contexts=contexts,
                generated_answer=result.get("answer", ""),
                is_agentic=True,
                guardrail_decision=result.get("guardrail_decision"),
                retrieval_attempts=result.get("retrieval_attempts", 1),
                reasoning_steps=result.get("reasoning_steps", []),
                rewritten_query=result.get("rewritten_query"),
            )

        except Exception as e:
            logger.error(f"Error running agentic system for query {query.query_id}: {e}")
            return SystemResponse(
                query_id=query.query_id,
                query=query.question,
                retrieved_doc_ids=[],
                retrieved_contexts=[],
                generated_answer="",
                is_agentic=True,
            )

    async def evaluate(
        self, ground_truth_queries: List[GroundTruthQuery]
    ) -> EvaluationResults:
        """Run complete evaluation.

        Args:
            ground_truth_queries: List of test queries with ground truth

        Returns:
            Evaluation results
        """
        logger.info(f"Starting evaluation on {len(ground_truth_queries)} queries")

        # Run system on all queries
        system_responses = []
        for query in ground_truth_queries:
            if hasattr(self.rag_system, "ask"):
                # Agentic system
                response = await self.run_agentic_system(query)
            else:
                # Standard system
                response = await self.run_system(query)

            system_responses.append(response)

            # Progress logging
            if len(system_responses) % 10 == 0:
                logger.info(f"Processed {len(system_responses)}/{len(ground_truth_queries)} queries")

        logger.info("System execution complete. Calculating metrics...")

        # Prepare results for evaluation
        retrieval_results = []
        generation_results = []
        ragas_inputs = []
        agent_decisions = []

        for query, response in zip(ground_truth_queries, system_responses):
            # Use relevant_arxiv_ids if available (as our system uses arxiv_ids), else fallback to paper_ids
            relevant_ids = getattr(query, "relevant_arxiv_ids", []) or query.relevant_paper_ids
            
            # Retrieval evaluation
            retrieval_results.append(
                RetrievalResult(
                    query_id=query.query_id,
                    retrieved_doc_ids=response.retrieved_doc_ids,
                    relevant_doc_ids=relevant_ids,
                )
            )

            # Generation evaluation
            generation_results.append(
                GenerationResult(
                    query_id=query.query_id,
                    query=query.question,
                    generated_answer=response.generated_answer,
                    ground_truth_answer=query.ideal_answer,
                    retrieved_contexts=response.retrieved_contexts,
                )
            )

            # RAGAS evaluation
            ragas_inputs.append(
                {
                    "query_id": query.query_id,
                    "query": query.question,
                    "contexts": response.retrieved_contexts,
                    "answer": response.generated_answer,
                    "ground_truth": query.ideal_answer,
                }
            )

            # Agent evaluation (if agentic)
            if response.is_agentic:
                agent_decisions.append(
                    AgentDecision(
                        query_id=query.query_id,
                        query=query.question,
                        guardrail_decision=response.guardrail_decision or "proceed",
                        guardrail_correct=True,  # You'll need ground truth for this
                        retrieval_attempts=response.retrieval_attempts,
                        final_success=len(response.retrieved_doc_ids) > 0,
                        original_query=query.question,
                        rewritten_query=response.rewritten_query,
                        reasoning_steps=response.reasoning_steps,
                    )
                )

        # Calculate all metrics
        logger.info("Calculating retrieval metrics...")
        retrieval_metrics = self.retrieval_metrics.evaluate_batch(
            retrieval_results, k_values=self.config.k_values
        )

        logger.info("Calculating generation metrics...")
        generation_metrics = await self.generation_metrics.evaluate_batch(
            generation_results
        )

        logger.info("Calculating RAGAS metrics...")
        ragas_metrics = await self.ragas_evaluator.evaluate_batch(ragas_inputs)

        # Agent metrics (if applicable)
        agent_metrics = None
        if agent_decisions:
            logger.info("Calculating agent metrics...")
            agent_metrics = self.agent_metrics.evaluate_batch(agent_decisions)

        # Calculate success rate
        success_rate = sum(
            1 for r in system_responses if r.generated_answer
        ) / len(system_responses)

        # Generate per-query results
        query_results = []
        for query, response in zip(ground_truth_queries, system_responses):
            query_results.append(
                {
                    "query_id": query.query_id,
                    "question": query.question,
                    "query_type": query.query_type,
                    "generated_answer": response.generated_answer,
                    "ground_truth": query.ideal_answer,
                    "num_retrieved": len(response.retrieved_doc_ids),
                    "reasoning_steps": response.reasoning_steps if response.is_agentic else [],
                }
            )

        # Create results object
        results = EvaluationResults(
            run_id=f"{self.config.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            config=self.config,
            retrieval_metrics=retrieval_metrics,
            generation_metrics=generation_metrics,
            ragas_metrics=ragas_metrics,
            agent_metrics=agent_metrics,
            num_queries=len(ground_truth_queries),
            success_rate=success_rate,
            query_results=query_results,
        )

        # Save results
        await self.save_results(results)

        logger.info("Evaluation complete!")
        return results

    async def save_results(self, results: EvaluationResults) -> None:
        """Save evaluation results to disk.

        Args:
            results: Evaluation results
        """
        output_dir = self.config.output_dir / results.run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save full results as JSON
        results_path = output_dir / "results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results.model_dump(), f, indent=2, ensure_ascii=False)

        logger.info(f"Saved full results to {results_path}")

        # Save summary report
        summary_path = output_dir / "summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(self.generate_summary(results))

        logger.info(f"Saved summary to {summary_path}")

    def generate_summary(self, results: EvaluationResults) -> str:
        """Generate human-readable summary.

        Args:
            results: Evaluation results

        Returns:
            Formatted summary string
        """
        summary = f"RAG System Evaluation Results - {results.run_id}\n"
        summary += "=" * 70 + "\n\n"

        summary += f"Run: {self.config.run_name}\n"
        summary += f"Timestamp: {results.timestamp}\n"
        summary += f"Queries Evaluated: {results.num_queries}\n"
        summary += f"Success Rate: {results.success_rate:.2%}\n\n"

        # Retrieval metrics
        summary += "Retrieval Performance:\n"
        for metric, value in results.retrieval_metrics.items():
            if metric != "num_queries":
                summary += f"  - {metric}: {value:.3f}\n"
        summary += "\n"

        # Generation metrics
        summary += "Generation Quality:\n"
        for metric, value in results.generation_metrics.items():
            if metric != "num_queries":
                summary += f"  - {metric}: {value:.3f}\n"
        summary += "\n"

        # RAGAS metrics
        summary += "RAGAS Metrics:\n"
        for metric, value in results.ragas_metrics.items():
            if metric != "num_queries":
                summary += f"  - {metric}: {value:.3f}\n"
        summary += "\n"

        # Agent metrics (if available)
        if results.agent_metrics:
            summary += self.agent_metrics.generate_summary(results.agent_metrics)

        return summary


async def run_evaluation_from_config(config_path: Path) -> EvaluationResults:
    """Run evaluation from configuration file.

    Args:
        config_path: Path to JSON configuration file

    Returns:
        Evaluation results
    """
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    config = EvaluationConfig(**config_dict)

    # Load test dataset
    from src.evaluation.datasets.synthetic_generator import SyntheticDataGenerator

    queries = SyntheticDataGenerator.load_dataset(config.test_dataset_path)

    # Initialize your RAG system here
    # rag_system = YourRAGSystem()

    # For now, this is a placeholder
    logger.warning("RAG system not initialized - using placeholder")
    rag_system = None

    runner = OfflineEvaluationRunner(rag_system=rag_system, config=config)

    return await runner.evaluate(queries)
