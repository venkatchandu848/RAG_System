"""Generate test dataset for RAG evaluation.

This script will:
1. Fetch papers from your database
2. Use LLM to generate diverse test queries
3. Save to test_set.json with ground truth annotations

Run this when your services are running:
$ docker compose up -d
$ python -m src.evaluation.generate_dataset
"""

import asyncio
import logging
from pathlib import Path

from src.config import get_settings
from src.db.factory import make_database
from src.evaluation.datasets.synthetic_generator import SyntheticDataGenerator
from src.repositories.paper import PaperRepository
from src.services.ollama.factory import make_ollama_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def generate_dataset(
    target_size: int = 150,
    output_path: Path = Path("src/evaluation/datasets/test_set.json"),
    model: str = "llama3.2:3b",
):
    """Generate test dataset from papers in database.

    Args:
        target_size: Number of queries to generate (default: 150)
        output_path: Where to save the dataset
        model: LLM model to use for generation
    """
    logger.info("=" * 80)
    logger.info("TEST DATASET GENERATION")
    logger.info("=" * 80)
    logger.info(f"Target size: {target_size} queries")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Model: {model}")
    logger.info("")

    # Check if dataset already exists
    if output_path.exists():
        logger.warning(f"⚠️  Dataset already exists: {output_path}")
        response = input("Do you want to overwrite it? (yes/no): ")
        if response.lower() not in ["yes", "y"]:
            logger.info("Cancelled. Exiting.")
            return

    # Initialize services
    logger.info("Initializing services...")
    try:
        settings = get_settings()
        database = make_database()
        ollama_client = make_ollama_client()
        logger.info("✅ Services initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")
        logger.error("Make sure docker compose is running: docker compose up -d")
        return

    # Fetch papers from database
    logger.info("\nFetching papers from database...")
    try:
        with database.get_session() as session:
            paper_repo = PaperRepository(session)
            papers = paper_repo.get_all(limit=300)  # Get extra for variety
        
        if not papers:
            logger.error("❌ No papers found in database!")
            logger.error("Run the data ingestion pipeline first to fetch papers.")
            return

        logger.info(f"✅ Found {len(papers)} papers in database")
        
        # Show sample
        logger.info("\nSample papers:")
        for i, paper in enumerate(papers[:3]):
            logger.info(f"  {i+1}. {paper.arxiv_id}: {paper.title[:60]}...")

    except Exception as e:
        logger.error(f"❌ Failed to fetch papers: {e}")
        logger.error("Make sure PostgreSQL is running and contains papers.")
        return

    # Filter papers with content
    papers_with_content = [p for p in papers if p.raw_text and len(p.raw_text) > 100]
    logger.info(f"\nPapers with content: {len(papers_with_content)}")

    if len(papers_with_content) < 50:
        logger.warning(f"⚠️  Only {len(papers_with_content)} papers have content")
        logger.warning("You might want to fetch more papers first for better diversity")
        response = input("Continue anyway? (yes/no): ")
        if response.lower() not in ["yes", "y"]:
            logger.info("Cancelled. Please run data ingestion first.")
            return

    # Initialize generator
    logger.info(f"\nInitializing synthetic data generator with {model}...")
    generator = SyntheticDataGenerator(
        ollama_client=ollama_client,
        model=model,
    )

    # Generate dataset
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING QUERIES")
    logger.info("=" * 80)
    logger.info("This will take several minutes depending on the model and query count...")
    logger.info("")

    try:
        queries = await generator.generate_dataset(
            papers=papers_with_content,
            target_size=target_size,
            output_path=output_path,
        )

        logger.info("\n" + "=" * 80)
        logger.info("✅ GENERATION COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"Total queries generated: {len(queries)}")
        logger.info(f"Saved to: {output_path}")
        logger.info("")

        # Show distribution
        from collections import Counter
        
        type_dist = Counter(q.query_type for q in queries)
        diff_dist = Counter(q.difficulty for q in queries)

        logger.info("Query type distribution:")
        for qtype, count in type_dist.items():
            logger.info(f"  - {qtype}: {count} ({count/len(queries)*100:.1f}%)")

        logger.info("\nDifficulty distribution:")
        for diff, count in diff_dist.items():
            logger.info(f"  - {diff}: {count} ({count/len(queries)*100:.1f}%)")

        # Show sample queries
        logger.info("\n" + "=" * 80)
        logger.info("SAMPLE QUERIES")
        logger.info("=" * 80)
        for i, query in enumerate(queries[:5]):
            logger.info(f"\n{i+1}. [{query.query_type}] [{query.difficulty}]")
            logger.info(f"   Q: {query.question}")
            logger.info(f"   Relevant papers: {len(query.relevant_arxiv_ids)}")
            logger.info(f"   Answer preview: {query.ideal_answer[:100]}...")

        logger.info("\n" + "=" * 80)
        logger.info("NEXT STEPS")
        logger.info("=" * 80)
        logger.info("1. Review the generated queries in test_set.json")
        logger.info("2. Optionally add manual ground truth queries")
        logger.info("3. Run evaluation:")
        logger.info("   python -m src.evaluation.run_evaluation")
        logger.info("")

    except Exception as e:
        logger.error(f"❌ Error during generation: {e}", exc_info=True)
        logger.error("\nTroubleshooting:")
        logger.error("- Check if Ollama is running: curl http://localhost:11434/api/tags")
        logger.error("- Check if model is available: ollama list")
        logger.error(f"- Pull model if needed: ollama pull {model}")
        return


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate test dataset for RAG evaluation"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=150,
        help="Number of queries to generate (default: 150)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.2:3b",
        help="LLM model to use (default: llama3.2:3b)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="src/evaluation/datasets/test_set.json",
        help="Output file path",
    )

    args = parser.parse_args()

    await generate_dataset(
        target_size=args.size,
        output_path=Path(args.output),
        model=args.model,
    )


if __name__ == "__main__":
    asyncio.run(main())
