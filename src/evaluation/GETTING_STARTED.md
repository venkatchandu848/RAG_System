# 🎯 RAG Evaluation - Getting Started

## Current Status

✅ **Evaluation Framework**: Complete (all metrics, runners, reports)  
❌ **Test Dataset**: Not generated yet (you need to create this)

## Quick Start: Generate Test Dataset & Run Evaluation

### Step 1: Generate Test Dataset (100-200 queries)

**Prerequisites:**
- Services running: `docker compose up -d`
- Papers in database (run data ingestion if needed)
- Ollama model available: `ollama pull llama3.2:3b`

**Generate the dataset:**
```bash
# Generate 150 test queries (takes 10-30 minutes)
python -m src.evaluation.generate_dataset

# Or customize:
python -m src.evaluation.generate_dataset --size 200 --model llama3.2:3b
```

This will create: `src/evaluation/datasets/test_set.json`

**What it generates:**
- 50% Factual queries (direct fact extraction)
- 30% Comparative queries (comparing papers)
- 20% Multi-hop queries (reasoning across sections)
- Ground truth: relevant paper IDs, ideal answers
- Difficulty levels: easy, medium, hard

### Step 2: Review the Dataset

```bash
# Check the generated file
cat src/evaluation/datasets/test_set.json | head -n 50

# Or open in VS Code
code src/evaluation/datasets/test_set.json
```

Verify:
- Questions are sensible and diverse
- Relevant papers are correctly identified
- Ideal answers are reasonable

### Step 3: Run Evaluation

**Run on all queries:**
```bash
# Full evaluation (takes 30-60 minutes for 150 queries)
python -m src.evaluation.run_evaluation
```

**Or start with a small batch:**
```bash
# Test on first 10 queries only
python -c "
import json
from pathlib import Path

# Load full dataset
with open('src/evaluation/datasets/test_set.json') as f:
    data = json.load(f)

# Save small batch
small_batch = {
    'metadata': {**data['metadata'], 'total_queries': 10},
    'queries': data['queries'][:10]
}

Path('src/evaluation/datasets/test_set_small.json').write_text(
    json.dumps(small_batch, indent=2)
)
print('Created test_set_small.json with 10 queries')
"

# Run evaluation on small batch
python -m src.evaluation.run_evaluation --dataset src/evaluation/datasets/test_set_small.json --name test_run
```

### Step 4: Review Results

Results will be in: `evaluation_results/<run_id>/`

**Files generated:**
- `results.json` - Full detailed results
- `summary.txt` - Text summary
- `reports/metrics_comparison.png` - Visual metrics
- `reports/report.html` - Interactive dashboard

**View results:**
```bash
# Read summary
cat evaluation_results/evaluation_v1_*/summary.txt

# Open HTML dashboard
start evaluation_results/evaluation_v1_*/reports/report.html  # Windows
open evaluation_results/evaluation_v1_*/reports/report.html   # Mac
```

## Expected Results Format

```
RAG SYSTEM EVALUATION RESULTS
================================================================================

RETRIEVAL PERFORMANCE:
  Precision@5:  0.XXX  ← How accurate are top 5 results?
  Recall@10:    0.XXX  ← How many relevant docs in top 10?
  nDCG@10:      0.XXX  ← Quality of ranking?
  MRR:          0.XXX  ← How quickly do we find relevant docs?

GENERATION QUALITY:
  Faithfulness:     0.XXX  ← Is answer factually consistent?
  Answer Relevancy: 0.XXX  ← Does answer address the query?
  BERTScore F1:     0.XXX  ← Semantic similarity to ideal answer

RAGAS SCORE: 0.XXX/1.0  ← Overall end-to-end quality

AGENTIC PERFORMANCE:
  Guardrail Precision:  0.XXX  ← How well does it detect out-of-scope?
  Retrieval Success:    0.XXX  ← Success rate after adaptive retrieval
  Avg Attempts:         X.XX   ← How many retrieval attempts needed?
```

## Troubleshooting

### "No papers found in database"
```bash
# Check if papers exist
docker compose exec postgres psql -U postgres -d rag_db -c "SELECT COUNT(*) FROM papers;"

# If empty, run data ingestion
docker compose exec airflow airflow dags trigger arxiv_paper_ingestion
```

### "Ollama model not found"
```bash
# Pull the model
ollama pull llama3.2:3b

# Check available models
ollama list
```

### "Services not initialized"
```bash
# Start all services
docker compose up -d

# Check services are running
docker compose ps

# Check health
curl http://localhost:8000/health
curl http://localhost:11434/api/tags
curl http://localhost:9200/_cluster/health
```

### "Generation is too slow"
```bash
# Use smaller model
python -m src.evaluation.generate_dataset --model llama3.2:1b --size 50

# Or reduce dataset size
python -m src.evaluation.generate_dataset --size 50
```

## Advanced Usage

### Custom Evaluation Configuration

Edit `run_evaluation.py` to customize:
- `k_values`: [5, 10, 20] - Different K values for metrics
- `top_k`: 5 - Number of documents to retrieve
- `use_hybrid`: True - Use hybrid vs BM25 search
- `model`: "llama3.2:1b" - LLM model for generation

### Production Monitoring

Add to your API in `src/main.py`:

```python
from src.evaluation.runners.online_eval import OnlineEvaluationMonitor

# Initialize in lifespan
monitor = OnlineEvaluationMonitor(window_size_minutes=60)
app.state.eval_monitor = monitor

# Record in endpoints
@router.post("/api/v1/ask")
async def ask(request: AskRequest, monitor: OnlineEvaluationMonitor = Depends(get_monitor)):
    start_time = time.time()
    result = await rag_system.ask(request.query)
    
    monitor.record_query(
        query_id=result["trace_id"],
        query=request.query,
        latency_ms=(time.time() - start_time) * 1000,
        cache_hit=result.get("cache_hit", False),
        num_retrieved=len(result.get("sources", [])),
        answer_length=len(result["answer"])
    )
    
    return result
```

## Next Steps

1. ✅ Generate test dataset
2. ✅ Run evaluation on small batch (10 queries)
3. ✅ Review results and fix issues
4. ✅ Run full evaluation (150-200 queries)
5. ✅ Set up production monitoring
6. ✅ Iterate and improve based on metrics

## Documentation

- **Full Guide**: [src/evaluation/README.md](README.md)
- **Implementation Details**: [../EVALUATION_IMPLEMENTATION_SUMMARY.md](../EVALUATION_IMPLEMENTATION_SUMMARY.md)
- **Metric Definitions**: See README.md

---

**Questions?** Check the full documentation or run:
```bash
python -m src.evaluation.quickstart  # Test individual components
```
