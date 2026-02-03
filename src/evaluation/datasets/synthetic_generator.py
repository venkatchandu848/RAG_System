"""Synthetic test dataset generator for RAG evaluation.

Generates question-answer pairs from arXiv papers with ground truth annotations.
"""

import json
import logging
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel

from src.models.paper import Paper
from src.services.ollama.client import OllamaClient

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    """Types of queries to generate."""

    FACTUAL = "factual"  # Direct fact extraction
    COMPARATIVE = "comparative"  # Compare multiple papers
    MULTI_HOP = "multi_hop"  # Requires reasoning across sections


class GroundTruthQuery(BaseModel):
    """Ground truth test query with annotations."""

    query_id: str
    question: str
    query_type: QueryType
    relevant_paper_ids: List[str]
    relevant_arxiv_ids: List[str]
    ideal_answer: str
    context_used: str
    difficulty: str  # easy, medium, hard
    metadata: Dict = {}


class SyntheticDataGenerator:
    """Generate synthetic test queries from arXiv papers.

    Uses LLM to create diverse question types with ground truth annotations.
    """

    def __init__(
        self,
        ollama_client: OllamaClient,
        model: str = "llama3.2:3b",
    ):
        """Initialize synthetic data generator.

        Args:
            ollama_client: Ollama client for LLM generation
            model: Model to use for generation
        """
        self.ollama = ollama_client
        self.model = model
        logger.info(f"Initialized SyntheticDataGenerator with model: {model}")

    async def generate_factual_query(
        self, paper: Paper, section_content: str
    ) -> Optional[GroundTruthQuery]:
        """Generate a factual query from paper content.

        Args:
            paper: Source paper
            section_content: Content to generate question from

        Returns:
            Ground truth query or None if generation fails
        """
        prompt = f"""You are creating test questions for a RAG system evaluation.

Given this paper section, create ONE factual question that can be answered directly from the text.

Paper Title: {paper.title}
Paper ID: {paper.arxiv_id}

Section Content:
{section_content[:2000]}

Generate a JSON response with:
{{
    "question": "A specific factual question answerable from this section",
    "ideal_answer": "The correct answer extracted from the text",
    "difficulty": "easy|medium|hard"
}}

Requirements:
- Question must be answerable from the provided content
- Answer must be factually correct
- Be specific and avoid generic questions
- Focus on key technical concepts or findings

Respond with ONLY valid JSON, no other text."""

        try:
            response = await self.ollama.generate(
                prompt=prompt, model=self.model, system_prompt="", temperature=0.5, format="json"
            )

            # Parse JSON response - extract text from response dict
            response_text = response.get("response", "") if isinstance(response, dict) else response
            result = json.loads(response_text.strip())

            # Make fields optional with defaults - be flexible with model output
            question = result.get("question", result.get("Question", ""))
            ideal_answer = result.get("ideal_answer", result.get("answer", result.get("ideal-answer", "")))
            difficulty = result.get("difficulty", "medium")

            if not question or not ideal_answer:
                logger.warning(f"Missing required fields in response: {result}")
                return None

            return GroundTruthQuery(
                query_id=f"{paper.arxiv_id}_factual_{hash(question) % 10000}",
                question=question,
                query_type=QueryType.FACTUAL,
                relevant_paper_ids=[str(paper.id)] if paper.id else [],
                relevant_arxiv_ids=[paper.arxiv_id],
                ideal_answer=ideal_answer,
                context_used=section_content[:1000],
                difficulty=difficulty,
                metadata={"paper_title": paper.title, "method": "llm_generated"},
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Response was: {response_text[:500]}")
            return None
        except KeyError as e:
            logger.error(f"Missing required key in response: {e}")
            logger.error(f"Response keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
            logger.error(f"Full response: {result}")
            return None
        except Exception as e:
            logger.error(f"Error generating factual query: {e}")
            return None

    async def generate_comparative_query(
        self, papers: List[Paper], max_papers: int = 2
    ) -> Optional[GroundTruthQuery]:
        """Generate a comparative query across multiple papers.

        Args:
            papers: List of papers to compare
            max_papers: Maximum papers to include in comparison

        Returns:
            Ground truth query or None if generation fails
        """
        selected_papers = papers[:max_papers]
        if len(selected_papers) < 2:
            return None

        papers_info = "\n\n".join(
            [
                f"Paper {i+1}:\nTitle: {p.title}\nID: {p.arxiv_id}\nAbstract: {p.abstract[:500]}"
                for i, p in enumerate(selected_papers)
            ]
        )

        prompt = f"""You are creating test questions for a RAG system evaluation.

Given these papers, create ONE comparative question that requires analyzing multiple papers.

{papers_info}

Generate a JSON response with:
{{
    "question": "A comparative question requiring insights from multiple papers",
    "ideal_answer": "An answer that synthesizes information from the papers",
    "difficulty": "medium|hard"
}}

Requirements:
- Question must require information from at least 2 papers
- Focus on comparing approaches, methods, or findings
- Answer should synthesize insights across papers

Respond with ONLY valid JSON, no other text."""

        try:
            response = await self.ollama.generate(
                prompt=prompt, model=self.model, system_prompt="", temperature=0.4, format="json"
            )

            # Extract text from response dict
            response_text = response.get("response", "") if isinstance(response, dict) else response
            result = json.loads(response_text.strip())

            # Make fields optional with defaults
            question = result.get("question", result.get("Question", ""))
            ideal_answer = result.get("ideal_answer", result.get("answer", result.get("ideal-answer", "")))
            difficulty = result.get("difficulty", "medium")

            if not question or not ideal_answer:
                logger.warning(f"Missing required fields in comparative response: {result}")
                return None

            return GroundTruthQuery(
                query_id=f"comparative_{hash(question) % 10000}",
                question=question,
                query_type=QueryType.COMPARATIVE,
                relevant_paper_ids=[str(p.id) for p in selected_papers if p.id],
                relevant_arxiv_ids=[p.arxiv_id for p in selected_papers],
                ideal_answer=ideal_answer,
                context_used=papers_info[:1000],
                difficulty=difficulty,
                metadata={
                    "paper_titles": [p.title for p in selected_papers],
                    "method": "llm_generated",
                },
            )

        except Exception as e:
            logger.error(f"Error generating comparative query: {e}")
            return None

    async def generate_multi_hop_query(
        self, paper: Paper, full_content: str
    ) -> Optional[GroundTruthQuery]:
        """Generate a multi-hop reasoning query.

        Args:
            paper: Source paper
            full_content: Full paper content

        Returns:
            Ground truth query or None if generation fails
        """
        prompt = f"""You are creating test questions for a RAG system evaluation.

Given this paper, create ONE multi-hop question that requires reasoning across multiple sections or concepts.

Paper Title: {paper.title}
Paper ID: {paper.arxiv_id}

Content Sample:
{full_content[:3000]}

Generate a JSON response with:
{{
    "question": "A complex question requiring multi-step reasoning",
    "ideal_answer": "An answer that demonstrates the reasoning chain",
    "difficulty": "hard"
}}

Requirements:
- Question must require connecting information from different parts
- Should involve logical reasoning or inference
- Answer should show the reasoning process

Respond with ONLY valid JSON, no other text."""

        try:
            response = await self.ollama.generate(
                prompt=prompt, model=self.model, system_prompt="", temperature=0.5, format="json"
            )

            # Extract text from response dict
            response_text = response.get("response", "") if isinstance(response, dict) else response
            result = json.loads(response_text.strip())

            # Make fields optional with defaults
            question = result.get("question", result.get("Question", ""))
            ideal_answer = result.get("ideal_answer", result.get("answer", result.get("ideal-answer", "")))
            
            if not question or not ideal_answer:
                logger.warning(f"Missing required fields in multi-hop response: {result}")
                return None

            return GroundTruthQuery(
                query_id=f"{paper.arxiv_id}_multihop_{hash(question) % 10000}",
                question=question,
                query_type=QueryType.MULTI_HOP,
                relevant_paper_ids=[str(paper.id)] if paper.id else [],
                relevant_arxiv_ids=[paper.arxiv_id],
                ideal_answer=ideal_answer,
                context_used=full_content[:1000],
                difficulty="hard",
                metadata={"paper_title": paper.title, "method": "llm_generated"},
            )

        except Exception as e:
            logger.error(f"Error generating multi-hop query: {e}")
            return None

    async def generate_dataset(
        self,
        papers: List[Paper],
        target_size: int = 150,
        output_path: Optional[Path] = None,
    ) -> List[GroundTruthQuery]:
        """Generate complete test dataset.

        Args:
            papers: List of papers to generate queries from
            target_size: Target number of queries to generate
            output_path: Optional path to save dataset

        Returns:
            List of generated queries
        """
        logger.info(f"Generating dataset of {target_size} queries from {len(papers)} papers")

        queries = []
        
        # Distribution: 50% factual, 30% comparative, 20% multi-hop
        target_factual = int(target_size * 0.5)
        target_comparative = int(target_size * 0.3)
        target_multihop = int(target_size * 0.2)

        # Generate factual queries
        logger.info(f"Generating {target_factual} factual queries...")
        for paper in papers[:target_factual]:
            if not paper.raw_text:
                continue

            # Use abstract or first section
            content = paper.abstract or paper.raw_text[:2000]
            query = await self.generate_factual_query(paper, content)
            if query:
                queries.append(query)

            if len([q for q in queries if q.query_type == QueryType.FACTUAL]) >= target_factual:
                break

        # Generate comparative queries
        logger.info(f"Generating {target_comparative} comparative queries...")
        for i in range(0, min(len(papers), target_comparative * 2), 2):
            query = await self.generate_comparative_query(papers[i : i + 2])
            if query:
                queries.append(query)

            if len([q for q in queries if q.query_type == QueryType.COMPARATIVE]) >= target_comparative:
                break

        # Generate multi-hop queries
        logger.info(f"Generating {target_multihop} multi-hop queries...")
        for paper in papers[:target_multihop]:
            if not paper.raw_text:
                continue

            query = await self.generate_multi_hop_query(paper, paper.raw_text[:4000])
            if query:
                queries.append(query)

            if len([q for q in queries if q.query_type == QueryType.MULTI_HOP]) >= target_multihop:
                break

        logger.info(f"Generated {len(queries)} total queries")

        # Save dataset if path provided
        if output_path:
            self.save_dataset(queries, output_path)

        return queries

    def save_dataset(self, queries: List[GroundTruthQuery], output_path: Path) -> None:
        """Save dataset to JSON file.

        Args:
            queries: List of queries to save
            output_path: Path to save dataset
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        dataset = {
            "metadata": {
                "total_queries": len(queries),
                "query_types": {
                    "factual": len([q for q in queries if q.query_type == QueryType.FACTUAL]),
                    "comparative": len([q for q in queries if q.query_type == QueryType.COMPARATIVE]),
                    "multi_hop": len([q for q in queries if q.query_type == QueryType.MULTI_HOP]),
                },
                "difficulty": {
                    "easy": len([q for q in queries if q.difficulty == "easy"]),
                    "medium": len([q for q in queries if q.difficulty == "medium"]),
                    "hard": len([q for q in queries if q.difficulty == "hard"]),
                },
            },
            "queries": [q.model_dump() for q in queries],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(queries)} queries to {output_path}")

    @staticmethod
    def load_dataset(input_path: Path) -> List[GroundTruthQuery]:
        """Load dataset from JSON file.

        Args:
            input_path: Path to dataset file

        Returns:
            List of ground truth queries
        """
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        queries = [GroundTruthQuery(**q) for q in data["queries"]]
        logger.info(f"Loaded {len(queries)} queries from {input_path}")

        return queries
