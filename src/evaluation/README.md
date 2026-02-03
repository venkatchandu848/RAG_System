# RAG System Evaluation Module

Comprehensive evaluation framework for the arXiv Paper Curator RAG system.

## 📋 Overview

This module provides end-to-end evaluation capabilities for RAG systems, including:

- **Synthetic Test Data Generation**: LLM-based generation of diverse test queries with ground truth
- **Retrieval Metrics**: Precision@K, Recall@K, nDCG, MRR
- **Generation Metrics**: Faithfulness, Answer Relevancy, BERTScore
- **RAGAS Integration**: Complete RAG evaluation using RAGAS framework
- **Agentic Metrics**: Specialized metrics for agentic RAG systems
- **Production Monitoring**: Real-time performance tracking
- **Visualization**: Interactive dashboards and reports

## 🏗️ Module Structure

```
src/evaluation/
├── metrics/
│   ├── retrieval_metrics.py      # Precision, Recall, nDCG, MRR
│   ├── generation_metrics.py     # Faithfulness, Relevancy, BERTScore
│   ├── ragas_evaluator.py        # RAGAS framework integration
│   └── agent_metrics.py          # Agentic-specific metrics
├── datasets/
│   ├── synthetic_generator.py    # LLM-based test generation
│   ├── test_set.json            # Your curated test dataset
│   └── ground_truth.json        # Labeled data
├── runners/
│   ├── offline_eval.py          # Batch evaluation runner
│   └── online_eval.py           # Production monitoring
├── reports/
│   └── dashboard.py             # Results visualization
└── example_evaluation.py        # Complete usage example
```

## 🚀 Quick Start

### 1. Generate Test Dataset

```python
from src.evaluation.datasets.synthetic_generator import SyntheticDataGenerator
from src.services.ollama.factory import make_ollama_client

# Initialize generator
ollama = make_ollama_client()
generator = SyntheticDataGenerator(ollama, model="llama3.2:3b")

# Generate 150 diverse queries
queries = await generator.generate_dataset(
    papers=papers,  # Your paper collection
    target_size=150,
    output_path=Path("src/evaluation/datasets/test_set.json")
)
```

### 2. Run Offline Evaluation

```python
from src.evaluation.runners.offline_eval import OfflineEvaluationRunner, EvaluationConfig

# Configure evaluation
config = EvaluationConfig(
    test_dataset_path=Path("src/evaluation/datasets/test_set.json"),
    output_dir=Path("evaluation_results"),
    k_values=[5, 10],
    top_k=5,
    use_hybrid=True,
    run_name="evaluation_v1"
)

# Run evaluation
runner = OfflineEvaluationRunner(
    rag_system=your_rag_system,
    config=config,
    ollama_client=ollama,
    embeddings_client=embeddings
)

results = await runner.evaluate(queries)
```

### 3. Generate Reports

```python
from src.evaluation.reports.dashboard import EvaluationDashboard

# Load results and generate all reports
dashboard = EvaluationDashboard(Path("evaluation_results/run_123/results.json"))
dashboard.generate_all_reports(Path("evaluation_results/run_123/reports"))
```

## 📊 Metrics Reference

### Retrieval Metrics

| Metric | Description | Range | Good Score |
|--------|-------------|-------|------------|
| **Precision@K** | Fraction of top-K results that are relevant | 0-1 | >0.7 |
| **Recall@K** | Fraction of relevant docs in top-K | 0-1 | >0.8 |
| **nDCG@K** | Normalized Discounted Cumulative Gain | 0-1 | >0.8 |
| **MRR** | Mean Reciprocal Rank | 0-1 | >0.75 |

### Generation Metrics

| Metric | Description | Range | Good Score |
|--------|-------------|-------|------------|
| **Faithfulness** | Factual consistency with context | 0-1 | >0.85 |
| **Answer Relevancy** | Relevance to query | 0-1 | >0.8 |
| **BERTScore F1** | Semantic similarity to reference | 0-1 | >0.8 |

### RAGAS Metrics

| Metric | Description | Range | Good Score |
|--------|-------------|-------|------------|
| **Context Precision** | Relevance of retrieved contexts | 0-1 | >0.8 |
| **Context Recall** | Coverage of ground truth in contexts | 0-1 | >0.85 |
| **RAGAS Score** | Harmonic mean of all metrics | 0-1 | >0.85 |

### Agentic Metrics

| Metric | Description | Range | Good Score |
|--------|-------------|-------|------------|
| **Guardrail Precision** | Accuracy of scope detection | 0-1 | >0.9 |
| **False Positive Rate** | Rate of incorrect rejections | 0-1 | <0.1 |
| **Query Rewrite Success** | Improvement after rewriting | 0-1 | >0.7 |
| **Adaptive Retrieval Success** | Success after multiple attempts | 0-1 | >0.9 |

### Individual Metrics

```python
# Retrieval metrics
from src.evaluation.metrics import RetrievalMetrics

metrics = RetrievalMetrics.evaluate_single(
    retrieved=["doc1", "doc2", "doc3"],
    relevant=["doc1", "doc3"],
    k_values=[5, 10]
)
# Returns: {"precision@5": 0.667, "recall@5": 1.0, ...}
```

```python
# Generation metrics
from src.evaluation.metrics import GenerationMetrics

gen_metrics = GenerationMetrics(ollama_client, embeddings_client)

faithfulness = await gen_metrics.faithfulness(
    answer="Transformers use attention mechanism...",
    contexts=["Context about transformers..."]
)

relevancy = await gen_metrics.answer_relevancy(
    query="How do transformers work?",
    answer="Transformers use attention mechanism..."
)
```

### Production Monitoring

```python
from src.evaluation.runners.online_eval import OnlineEvaluationMonitor

# Initialize monitor
monitor = OnlineEvaluationMonitor(
    window_size_minutes=60,
    langfuse_tracer=langfuse
)

# Record queries
monitor.record_query(
    query_id="query_123",
    query="What is attention?",
    latency_ms=250.5,
    cache_hit=True,
    num_retrieved=5,
    answer_length=150
)

# Record user feedback
monitor.record_feedback(
    query_id="query_123",
    score=0.9,
    feedback="Great answer!"
)

# Get real-time metrics
summary = monitor.get_summary()
# {"avg_latency_ms": 245.3, "cache_hit_rate": 0.67, ...}

# Check for alerts
alerts = monitor.check_alerts()
# ["HIGH LATENCY: P95 latency is 5200ms"]

# Generate report
report = monitor.generate_report()
```

## 🧪 Testing Your Evaluation

Run the example script:

```bash
# Generate test data and demo metrics
uv run python src/evaluation/example_evaluation.py
```

Or run individual components:

```python
# Test dataset generation
python -c "
import asyncio
from pathlib import Path
from src.evaluation.datasets.synthetic_generator import SyntheticDataGenerator
from src.services.ollama.factory import make_ollama_client

async def test():
    ollama = make_ollama_client()
    generator = SyntheticDataGenerator(ollama)
    # Use a small sample
    queries = await generator.generate_dataset(
        papers=papers[:10],
        target_size=10,
        output_path=Path('test_queries.json')
    )
    print(f'Generated {len(queries)} queries')

asyncio.run(test())
"
```

## 📝 Ground Truth Format

Test dataset structure:

```json
{
  "metadata": {
    "total_queries": 150,
    "query_types": {
      "factual": 75,
      "comparative": 45,
      "multi_hop": 30
    }
  },
  "queries": [
    {
      "query_id": "2301.00001_factual_1234",
      "question": "What is the main contribution of this paper?",
      "query_type": "factual",
      "relevant_paper_ids": ["123"],
      "relevant_arxiv_ids": ["2301.00001"],
      "ideal_answer": "The paper introduces...",
      "context_used": "Abstract text...",
      "difficulty": "medium",
      "metadata": {}
    }
  ]
}
```

## 🎯 Integration with RAG System

To integrate with your RAG system, implement the required interface in `offline_eval.py`:

```python
async def run_system(self, query: GroundTruthQuery) -> SystemResponse:
    # Call your RAG system
    result = await self.rag_system.search_and_answer(
        query=query.question,
        top_k=self.config.top_k,
        use_hybrid=self.config.use_hybrid,
    )
    
    return SystemResponse(
        query_id=query.query_id,
        query=query.question,
        retrieved_doc_ids=result["doc_ids"],
        retrieved_contexts=result["contexts"],
        generated_answer=result["answer"],
    )
```

## 📊 Frameworks Integration

### RAGAS

The module provides RAGAS-compatible metrics:
- Context Precision
- Context Recall
- Faithfulness
- Answer Relevancy

### DeepEval

Compatible with DeepEval's evaluation format. Metrics can be exported to DeepEval format.

### Langfuse

Direct integration with Langfuse for:
- Trace-level evaluation
- Feedback collection
- Production monitoring

## 🔍 Best Practices

1. **Test Dataset Quality**
   - Aim for 150-200 diverse queries
   - Balance query types (50% factual, 30% comparative, 20% multi-hop)
   - Include edge cases and difficult queries

2. **Evaluation Frequency**
   - Run offline evaluation before each release
   - Monitor production metrics continuously
   - Re-evaluate when making significant changes

3. **Metric Selection**
   - Use retrieval metrics to optimize search
   - Use generation metrics to optimize LLM prompts
   - Use RAGAS for end-to-end quality
   - Use agent metrics for decision-making quality

4. **Thresholds**
   - Set minimum acceptable scores based on your use case
   - Track metric trends over time
   - Alert on significant degradation

## 📚 References

- [RAGAS Framework](https://github.com/explodinggradients/ragas)
- [Information Retrieval Metrics](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval))
- [BERTScore](https://github.com/Tiiiger/bert_score)
- [Langfuse Documentation](https://langfuse.com/docs)

## 🤝 Contributing

To add new metrics:

1. Create metric class in appropriate file
2. Add tests in `tests/evaluation/`
3. Update dashboard visualization
4. Document in this README

## 📄 License

Same as parent project (arXiv Paper Curator)
