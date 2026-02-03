"""Quick start script for RAG evaluation.

Run this to test the evaluation system components.
"""

import asyncio
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def quick_test_metrics():
    """Quick test of individual metrics."""
    logger.info("=" * 70)
    logger.info("Testing Evaluation Metrics")
    logger.info("=" * 70)

    # Test 1: Retrieval Metrics
    logger.info("\n1. Testing Retrieval Metrics...")
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from src.evaluation.metrics import RetrievalMetrics

    retrieved_docs = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant_docs = ["doc1", "doc3", "doc7", "doc8"]

    metrics = RetrievalMetrics.evaluate_single(
        retrieved=retrieved_docs,
        relevant=relevant_docs,
        k_values=[5],
    )

    logger.info("   Results:")
    for metric, value in metrics.items():
        logger.info(f"   - {metric}: {value:.3f}")

    # Test 2: Load synthetic data structure
    logger.info("\n2. Testing Synthetic Data Structure...")
    from src.evaluation.datasets.synthetic_generator import GroundTruthQuery

    sample_query = GroundTruthQuery(
        query_id="test_001",
        question="What are the key innovations in transformer architectures?",
        query_type="factual",
        relevant_paper_ids=["123", "456"],
        relevant_arxiv_ids=["2301.00001", "2301.00002"],
        ideal_answer="Transformers introduce self-attention mechanism...",
        context_used="Paper abstract about transformers...",
        difficulty="medium",
        metadata={"test": True},
    )

    logger.info(f"   Sample query created: {sample_query.query_id}")
    logger.info(f"   Question: {sample_query.question}")
    logger.info(f"   Type: {sample_query.query_type}")
    logger.info(f"   Difficulty: {sample_query.difficulty}")

    # Test 3: Agent Metrics
    logger.info("\n3. Testing Agent Metrics...")
    from src.evaluation.metrics import AgentDecision, AgentMetrics

    decisions = [
        AgentDecision(
            query_id="q1",
            query="What is attention?",
            guardrail_decision="proceed",
            guardrail_correct=True,
            retrieval_attempts=1,
            final_success=True,
            original_query="What is attention?",
            initial_relevant_count=3,
            final_relevant_count=3,
        ),
        AgentDecision(
            query_id="q2",
            query="Irrelevant question",
            guardrail_decision="out_of_scope",
            guardrail_correct=True,
            retrieval_attempts=0,
            final_success=False,
            original_query="Irrelevant question",
            initial_relevant_count=0,
            final_relevant_count=0,
        ),
    ]

    agent_metrics = AgentMetrics.evaluate_batch(decisions)
    logger.info("   Agent Metrics:")
    logger.info(f"   - Guardrail Accuracy: {agent_metrics['guardrail_accuracy']:.2%}")
    logger.info(f"   - Overall Success: {agent_metrics['retrieval_overall_success_rate']:.2%}")

    # Test 4: Production Monitoring
    logger.info("\n4. Testing Production Monitoring...")
    from src.evaluation.runners.online_eval import OnlineEvaluationMonitor

    monitor = OnlineEvaluationMonitor(window_size_minutes=60)

    # Simulate some queries
    for i in range(5):
        monitor.record_query(
            query_id=f"query_{i}",
            query=f"Test query {i}",
            latency_ms=100 + i * 50,
            cache_hit=i % 2 == 0,
            num_retrieved=5,
            answer_length=150,
        )

    summary = monitor.get_summary()
    logger.info("   Production Metrics:")
    logger.info(f"   - Queries: {summary['num_queries']}")
    logger.info(f"   - Avg Latency: {summary['avg_latency_ms']:.1f}ms")
    logger.info(f"   - Cache Hit Rate: {summary['cache_hit_rate']:.1%}")

    logger.info("\n✅ All metric tests passed!")


async def show_usage_examples():
    """Show usage examples."""
    logger.info("\n" + "=" * 70)
    logger.info("Usage Examples")
    logger.info("=" * 70)

    print("""
1. GENERATE TEST DATASET:
   --------------------------------------------------
   from src.evaluation.datasets.synthetic_generator import SyntheticDataGenerator
   
   generator = SyntheticDataGenerator(ollama_client)
   queries = await generator.generate_dataset(
       papers=papers,
       target_size=150,
       output_path=Path("src/evaluation/datasets/test_set.json")
   )

2. RUN OFFLINE EVALUATION:
   --------------------------------------------------
   from src.evaluation.runners.offline_eval import OfflineEvaluationRunner
   
   runner = OfflineEvaluationRunner(
       rag_system=your_rag_system,
       config=config,
       ollama_client=ollama,
       embeddings_client=embeddings
   )
   
   results = await runner.evaluate(queries)

3. GENERATE REPORTS:
   --------------------------------------------------
   from src.evaluation.reports.dashboard import EvaluationDashboard
   
   dashboard = EvaluationDashboard(results_path)
   dashboard.generate_all_reports(output_dir)

4. PRODUCTION MONITORING:
   --------------------------------------------------
   from src.evaluation.runners.online_eval import OnlineEvaluationMonitor
   
   monitor = OnlineEvaluationMonitor(window_size_minutes=60)
   
   # In your API
   monitor.record_query(
       query_id=trace_id,
       query=query,
       latency_ms=latency,
       cache_hit=cache_hit,
       num_retrieved=num_docs,
       answer_length=len(answer)
   )
   
   # Get metrics
   summary = monitor.get_summary()
   alerts = monitor.check_alerts()
""")


async def show_next_steps():
    """Show next steps."""
    logger.info("\n" + "=" * 70)
    logger.info("Next Steps")
    logger.info("=" * 70)

    print("""
📋 TO GET STARTED:

1. Install dependencies:
   $ uv sync

2. Ensure services are running:
   $ docker compose up -d

3. Generate test dataset:
   - Uncomment dataset generation in example_evaluation.py
   - Run: uv run python src/evaluation/example_evaluation.py
   - Review: src/evaluation/datasets/test_set.json

4. Integrate with your RAG system:
   - Adapt offline_eval.py run_system() method
   - Map your RAG outputs to SystemResponse format

5. Run evaluation:
   - Test on small batch first (10 queries)
   - Run full evaluation (150+ queries)
   - Generate reports

6. Set up production monitoring:
   - Add OnlineEvaluationMonitor to main.py
   - Record metrics in API endpoints
   - Monitor in real-time

📚 DOCUMENTATION:
   - Full guide: src/evaluation/README.md
   - Examples: src/evaluation/example_evaluation.py
   - Summary: EVALUATION_IMPLEMENTATION_SUMMARY.md

🎯 EVALUATION TARGETS:
   - Precision@5: >0.85
   - nDCG@10: >0.85
   - Faithfulness: >0.90
   - RAGAS Score: >0.85
   - Guardrail Precision: >0.95
   - P95 Latency: <3s
""")


async def main():
    """Run quick start."""
    logger.info("🚀 RAG Evaluation System - Quick Start")
    logger.info("")

    try:
        # Test metrics
        await quick_test_metrics()

        # Show usage
        await show_usage_examples()

        # Show next steps
        await show_next_steps()

        logger.info("\n" + "=" * 70)
        logger.info("✅ Quick Start Complete!")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
