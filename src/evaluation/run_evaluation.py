"""Run complete RAG evaluation on test dataset.

This script will:
1. Load test dataset (test_set.json)
2. Run your RAG system on all queries
3. Calculate all metrics (retrieval, generation, RAGAS, agentic)
4. Generate reports and visualizations

Prerequisites:
- Test dataset exists (run generate_dataset.py first)
- Services are running (docker compose up -d)
- RAG system is properly integrated

Run:
$ python -m src.evaluation.run_evaluation
"""

import asyncio
import logging
from pathlib import Path

from src.config import get_settings
from src.db.factory import make_database
from src.evaluation.datasets.synthetic_generator import SyntheticDataGenerator
from src.evaluation.reports.dashboard import EvaluationDashboard
from src.evaluation.runners.offline_eval import EvaluationConfig, OfflineEvaluationRunner
from src.services.agents.agentic_rag import AgenticRAGService
from src.services.embeddings.factory import make_embeddings_service
from src.services.langfuse.factory import make_langfuse_tracer
from src.services.ollama.factory import make_ollama_client
from src.services.opensearch.factory import make_opensearch_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def run_evaluation(
    test_dataset_path: Path = Path("src/evaluation/datasets/test_set.json"),
    output_dir: Path = Path("evaluation_results"),
    run_name: str = "evaluation_v1",
    use_agentic: bool = True,
):
    """Run complete RAG evaluation.

    Args:
        test_dataset_path: Path to test dataset JSON
        output_dir: Directory to save results
        run_name: Name for this evaluation run
        use_agentic: Whether to use agentic RAG system
    """
    logger.info("=" * 80)
    logger.info("RAG SYSTEM EVALUATION")
    logger.info("=" * 80)
    logger.info(f"Test dataset: {test_dataset_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Run name: {run_name}")
    logger.info(f"Agentic mode: {use_agentic}")
    logger.info("")

    # Check if test dataset exists
    if not test_dataset_path.exists():
        logger.error(f"❌ Test dataset not found: {test_dataset_path}")
        logger.error("\nGenerate it first:")
        logger.error("  python -m src.evaluation.generate_dataset")
        return

    # Load test dataset
    logger.info("Loading test dataset...")
    try:
        queries = SyntheticDataGenerator.load_dataset(test_dataset_path)
        logger.info(f"✅ Loaded {len(queries)} test queries")
    except Exception as e:
        logger.error(f"❌ Failed to load dataset: {e}")
        return

    # Initialize services
    logger.info("\nInitializing services...")
    try:
        settings = get_settings()
        database = make_database()
        opensearch = make_opensearch_client()
        embeddings = make_embeddings_service()
        ollama = make_ollama_client()
        langfuse = make_langfuse_tracer()
        
        logger.info("✅ Services initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")
        logger.error("Make sure docker compose is running: docker compose up -d")
        return

    # Initialize RAG system
    logger.info("\nInitializing RAG system...")
    try:
        if use_agentic:
            rag_system = AgenticRAGService(
                opensearch_client=opensearch,
                ollama_client=ollama,
                embeddings_client=embeddings,
                langfuse_tracer=langfuse,
            )
            logger.info("✅ Agentic RAG system initialized")
        else:
            # You can add standard RAG system here
            logger.error("❌ Standard RAG system not implemented yet")
            logger.error("Use --agentic flag or implement standard RAG")
            return
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG system: {e}")
        return

    # Configure evaluation
    config = EvaluationConfig(
        test_dataset_path=test_dataset_path,
        output_dir=output_dir,
        k_values=[5, 10],
        use_llm_metrics=True,
        use_embeddings_metrics=True,
        top_k=5,
        use_hybrid=True,
        model="llama3.2:1b",
        run_name=run_name,
        description=f"Evaluation of {'agentic' if use_agentic else 'standard'} RAG system",
    )

    # Initialize evaluation runner
    logger.info("\nInitializing evaluation runner...")
    runner = OfflineEvaluationRunner(
        rag_system=rag_system,
        config=config,
        ollama_client=ollama,
        embeddings_client=embeddings,
    )
    logger.info("✅ Evaluation runner initialized")

    # Run evaluation
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING EVALUATION")
    logger.info("=" * 80)
    logger.info(f"Processing {len(queries)} queries...")
    logger.info("This will take a while depending on your system...")
    logger.info("")

    try:
        results = await runner.evaluate(queries)

        logger.info("\n" + "=" * 80)
        logger.info("✅ EVALUATION COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"Results saved to: {output_dir / results.run_id}")
        logger.info("")

        # Show quick summary
        logger.info("QUICK SUMMARY:")
        logger.info("-" * 40)
        logger.info(f"Queries evaluated: {results.num_queries}")
        logger.info(f"Success rate: {results.success_rate:.1%}")
        logger.info("")
        
        logger.info("Retrieval:")
        logger.info(f"  Precision@5: {results.retrieval_metrics.get('precision@5', 0):.3f}")
        logger.info(f"  Recall@10:   {results.retrieval_metrics.get('recall@10', 0):.3f}")
        logger.info(f"  nDCG@10:     {results.retrieval_metrics.get('ndcg@10', 0):.3f}")
        logger.info(f"  MRR:         {results.retrieval_metrics.get('mrr', 0):.3f}")
        logger.info("")
        
        logger.info("Generation:")
        logger.info(f"  Faithfulness:     {results.generation_metrics.get('faithfulness', 0):.3f}")
        logger.info(f"  Answer Relevancy: {results.generation_metrics.get('answer_relevancy', 0):.3f}")
        logger.info(f"  BERTScore F1:     {results.generation_metrics.get('bertscore_f1', 0):.3f}")
        logger.info("")
        
        logger.info("RAGAS:")
        logger.info(f"  RAGAS Score: {results.ragas_metrics.get('ragas_score', 0):.3f}/1.0")
        logger.info("")

        if results.agent_metrics:
            logger.info("Agentic:")
            logger.info(f"  Guardrail Precision: {results.agent_metrics.get('guardrail_precision', 0):.3f}")
            logger.info(f"  Retrieval Success:   {results.agent_metrics.get('retrieval_overall_success_rate', 0):.3f}")
            logger.info("")

        # Generate reports
        logger.info("\n" + "=" * 80)
        logger.info("GENERATING REPORTS")
        logger.info("=" * 80)

        results_path = output_dir / results.run_id / "results.json"
        reports_dir = output_dir / results.run_id / "reports"

        dashboard = EvaluationDashboard(results_path)
        dashboard.generate_all_reports(reports_dir)

        logger.info(f"\n✅ Reports generated in: {reports_dir}")
        logger.info("\nGenerated files:")
        logger.info(f"  - summary.txt              (text summary)")
        logger.info(f"  - metrics_comparison.png   (metrics visualization)")
        logger.info(f"  - query_type_analysis.png  (performance by type)")
        logger.info(f"  - report.html              (interactive dashboard)")
        logger.info("")
        logger.info(f"Open the HTML report: {reports_dir / 'report.html'}")

    except Exception as e:
        logger.error(f"❌ Error during evaluation: {e}", exc_info=True)
        return


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run RAG system evaluation")
    parser.add_argument(
        "--dataset",
        type=str,
        default="src/evaluation/datasets/test_set.json",
        help="Path to test dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="evaluation_v1",
        help="Name for this evaluation run",
    )
    parser.add_argument(
        "--agentic",
        action="store_true",
        default=True,
        help="Use agentic RAG system (default: True)",
    )

    args = parser.parse_args()

    await run_evaluation(
        test_dataset_path=Path(args.dataset),
        output_dir=Path(args.output),
        run_name=args.name,
        use_agentic=args.agentic,
    )


if __name__ == "__main__":
    asyncio.run(main())
