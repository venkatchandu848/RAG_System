"""Results visualization and reporting dashboard.

Generates visualizations and interactive reports for evaluation results.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


class EvaluationDashboard:
    """Generate visualizations and reports for evaluation results."""

    def __init__(self, results_path: Path):
        """Initialize dashboard.

        Args:
            results_path: Path to evaluation results JSON file
        """
        self.results_path = results_path

        with open(results_path, "r") as f:
            self.results = json.load(f)

        logger.info(f"Loaded evaluation results from {results_path}")

    def generate_summary_report(self, output_path: Optional[Path] = None) -> str:
        """Generate comprehensive summary report.

        Args:
            output_path: Optional path to save report

        Returns:
            Formatted summary string
        """
        summary = "=" * 80 + "\n"
        summary += "RAG SYSTEM EVALUATION RESULTS - arXiv Paper Curator\n"
        summary += "=" * 80 + "\n\n"

        # Overview
        summary += f"Run ID: {self.results['run_id']}\n"
        summary += f"Timestamp: {self.results['timestamp']}\n"
        summary += f"Queries Evaluated: {self.results['num_queries']}\n"
        summary += f"Success Rate: {self.results['success_rate']:.1%}\n\n"

        # Retrieval Performance
        summary += "RETRIEVAL PERFORMANCE:\n"
        summary += "-" * 40 + "\n"
        ret = self.results["retrieval_metrics"]
        summary += f"  Precision@5:  {ret.get('precision@5', 0):.3f}\n"
        summary += f"  Precision@10: {ret.get('precision@10', 0):.3f}\n"
        summary += f"  Recall@5:     {ret.get('recall@5', 0):.3f}\n"
        summary += f"  Recall@10:    {ret.get('recall@10', 0):.3f}\n"
        summary += f"  nDCG@10:      {ret.get('ndcg@10', 0):.3f}\n"
        summary += f"  MRR:          {ret.get('mrr', 0):.3f}\n\n"

        # Generation Quality
        summary += "GENERATION QUALITY:\n"
        summary += "-" * 40 + "\n"
        gen = self.results["generation_metrics"]
        summary += f"  Faithfulness:     {gen.get('faithfulness', 0):.3f}\n"
        summary += f"  Answer Relevancy: {gen.get('answer_relevancy', 0):.3f}\n"
        summary += f"  BERTScore F1:     {gen.get('bertscore_f1', 0):.3f}\n\n"

        # RAGAS Score
        summary += "END-TO-END RAGAS METRICS:\n"
        summary += "-" * 40 + "\n"
        ragas = self.results["ragas_metrics"]
        summary += f"  Context Precision: {ragas.get('context_precision', 0):.3f}\n"
        summary += f"  Context Recall:    {ragas.get('context_recall', 0):.3f}\n"
        summary += f"  Faithfulness:      {ragas.get('faithfulness', 0):.3f}\n"
        summary += f"  Answer Relevancy:  {ragas.get('answer_relevancy', 0):.3f}\n"
        summary += f"  RAGAS Score:       {ragas.get('ragas_score', 0):.3f}/1.0\n\n"

        # Agentic Performance (if available)
        if self.results.get("agent_metrics"):
            summary += "AGENTIC PERFORMANCE:\n"
            summary += "-" * 40 + "\n"
            agent = self.results["agent_metrics"]
            summary += f"  Guardrail Precision:     {agent.get('guardrail_precision', 0):.3f}\n"
            summary += f"  Guardrail FP Rate:       {agent.get('guardrail_false_positive_rate', 0):.3f}\n"
            summary += f"  Query Rewrite Success:   {agent.get('rewrite_success_rate', 0):.3f}\n"
            summary += f"  Retrieval Success:       {agent.get('retrieval_overall_success_rate', 0):.3f}\n"
            summary += f"  Avg Attempts:            {agent.get('retrieval_avg_attempts', 0):.2f}\n\n"

        summary += "=" * 80 + "\n"

        # Save if output path provided
        if output_path:
            with open(output_path, "w") as f:
                f.write(summary)
            logger.info(f"Saved summary report to {output_path}")

        return summary

    def plot_metrics_comparison(self, output_path: Optional[Path] = None) -> None:
        """Plot comparison of all metrics.

        Args:
            output_path: Optional path to save plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available - skipping plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("RAG System Evaluation Metrics", fontsize=16, fontweight="bold")

        # Retrieval metrics
        ax1 = axes[0, 0]
        ret = self.results["retrieval_metrics"]
        metrics = ["P@5", "R@5", "P@10", "R@10", "nDCG@10", "MRR"]
        values = [
            ret.get("precision@5", 0),
            ret.get("recall@5", 0),
            ret.get("precision@10", 0),
            ret.get("recall@10", 0),
            ret.get("ndcg@10", 0),
            ret.get("mrr", 0),
        ]
        ax1.bar(metrics, values, color="steelblue")
        ax1.set_ylim(0, 1)
        ax1.set_title("Retrieval Metrics")
        ax1.set_ylabel("Score")
        ax1.grid(axis="y", alpha=0.3)

        # Generation metrics
        ax2 = axes[0, 1]
        gen = self.results["generation_metrics"]
        metrics = ["Faithfulness", "Relevancy", "BERTScore"]
        values = [
            gen.get("faithfulness", 0),
            gen.get("answer_relevancy", 0),
            gen.get("bertscore_f1", 0),
        ]
        ax2.bar(metrics, values, color="forestgreen")
        ax2.set_ylim(0, 1)
        ax2.set_title("Generation Quality")
        ax2.set_ylabel("Score")
        ax2.grid(axis="y", alpha=0.3)

        # RAGAS metrics
        ax3 = axes[1, 0]
        ragas = self.results["ragas_metrics"]
        metrics = ["Context\nPrecision", "Context\nRecall", "Faithfulness", "Relevancy"]
        values = [
            ragas.get("context_precision", 0),
            ragas.get("context_recall", 0),
            ragas.get("faithfulness", 0),
            ragas.get("answer_relevancy", 0),
        ]
        ax3.bar(metrics, values, color="coral")
        ax3.set_ylim(0, 1)
        ax3.set_title("RAGAS Metrics")
        ax3.set_ylabel("Score")
        ax3.grid(axis="y", alpha=0.3)

        # Overall score
        ax4 = axes[1, 1]
        ragas_score = ragas.get("ragas_score", 0)
        ax4.barh(["RAGAS\nScore"], [ragas_score], color="darkviolet")
        ax4.set_xlim(0, 1)
        ax4.set_title("Overall Score")
        ax4.set_xlabel("Score")
        ax4.grid(axis="x", alpha=0.3)

        # Add score annotation
        ax4.text(ragas_score / 2, 0, f"{ragas_score:.3f}", 
                ha="center", va="center", fontsize=20, fontweight="bold", color="white")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved metrics plot to {output_path}")

        plt.close()

    def plot_query_type_analysis(self, output_path: Optional[Path] = None) -> None:
        """Plot performance by query type.

        Args:
            output_path: Optional path to save plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available - skipping plot")
            return

        # Analyze query results by type
        query_results = self.results.get("query_results", [])

        if not query_results:
            logger.warning("No query results available for analysis")
            return

        # Group by query type
        from collections import defaultdict

        type_stats = defaultdict(lambda: {"count": 0, "success": 0})

        for result in query_results:
            qtype = result.get("query_type", "unknown")
            type_stats[qtype]["count"] += 1
            if result.get("generated_answer"):
                type_stats[qtype]["success"] += 1

        # Calculate success rates
        types = list(type_stats.keys())
        counts = [type_stats[t]["count"] for t in types]
        success_rates = [
            type_stats[t]["success"] / type_stats[t]["count"] for t in types
        ]

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Performance by Query Type", fontsize=14, fontweight="bold")

        # Query distribution
        ax1.pie(counts, labels=types, autopct="%1.1f%%", startangle=90)
        ax1.set_title("Query Distribution")

        # Success rates
        ax2.bar(types, success_rates, color="teal")
        ax2.set_ylim(0, 1)
        ax2.set_title("Success Rate by Type")
        ax2.set_ylabel("Success Rate")
        ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved query type analysis to {output_path}")

        plt.close()

    def export_to_html(self, output_path: Path) -> None:
        """Export results to interactive HTML report.

        Args:
            output_path: Path to save HTML file
        """
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Evaluation Results - {self.results['run_id']}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1, h2, h3 {{
            color: #333;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-value {{
            font-size: 36px;
            font-weight: bold;
            color: #667eea;
        }}
        .metric-label {{
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }}
        .section {{
            background: white;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: 600;
        }}
        .score-bar {{
            height: 20px;
            background-color: #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
        }}
        .score-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.3s ease;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>RAG System Evaluation Results</h1>
        <p><strong>Run ID:</strong> {self.results['run_id']}</p>
        <p><strong>Timestamp:</strong> {self.results['timestamp']}</p>
        <p><strong>Queries Evaluated:</strong> {self.results['num_queries']}</p>
    </div>

    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-value">{self.results['retrieval_metrics'].get('precision@5', 0):.3f}</div>
            <div class="metric-label">Precision@5</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{self.results['retrieval_metrics'].get('ndcg@10', 0):.3f}</div>
            <div class="metric-label">nDCG@10</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{self.results['generation_metrics'].get('faithfulness', 0):.3f}</div>
            <div class="metric-label">Faithfulness</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{self.results['ragas_metrics'].get('ragas_score', 0):.3f}</div>
            <div class="metric-label">RAGAS Score</div>
        </div>
    </div>

    <div class="section">
        <h2>Retrieval Metrics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Score</th>
                <th>Visualization</th>
            </tr>
"""

        # Add retrieval metrics rows
        for metric, value in self.results["retrieval_metrics"].items():
            if metric != "num_queries":
                html += f"""
            <tr>
                <td>{metric}</td>
                <td>{value:.3f}</td>
                <td>
                    <div class="score-bar">
                        <div class="score-fill" style="width: {value * 100}%"></div>
                    </div>
                </td>
            </tr>
"""

        html += """
        </table>
    </div>

    <div class="section">
        <h2>Generation Quality</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Score</th>
                <th>Visualization</th>
            </tr>
"""

        # Add generation metrics rows
        for metric, value in self.results["generation_metrics"].items():
            if metric != "num_queries":
                html += f"""
            <tr>
                <td>{metric}</td>
                <td>{value:.3f}</td>
                <td>
                    <div class="score-bar">
                        <div class="score-fill" style="width: {value * 100}%"></div>
                    </div>
                </td>
            </tr>
"""

        html += """
        </table>
    </div>

</body>
</html>
"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info(f"Saved HTML report to {output_path}")

    def generate_all_reports(self, output_dir: Path) -> None:
        """Generate all reports and visualizations.

        Args:
            output_dir: Directory to save all outputs
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Generating all evaluation reports...")

        # Summary report
        summary = self.generate_summary_report(output_dir / "summary.txt")
        print(summary)

        # Plots
        self.plot_metrics_comparison(output_dir / "metrics_comparison.png")
        self.plot_query_type_analysis(output_dir / "query_type_analysis.png")

        # HTML report
        self.export_to_html(output_dir / "report.html")

        logger.info(f"All reports generated in {output_dir}")
