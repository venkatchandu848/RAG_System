"""Online evaluation for production monitoring.

Tracks RAG system performance in production with real-time metrics.
"""

import asyncio
import logging
from collections import deque
from datetime import datetime, timedelta
from typing import Deque, Dict, List, Optional

from pydantic import BaseModel

from src.evaluation.metrics.agent_metrics import AgentDecision, AgentMetrics
from src.evaluation.metrics.generation_metrics import GenerationMetrics
from src.evaluation.metrics.retrieval_metrics import RetrievalMetrics

logger = logging.getLogger(__name__)


class ProductionQuery(BaseModel):
    """Single production query for monitoring."""

    query_id: str
    query: str
    timestamp: datetime

    # Performance
    latency_ms: float
    cache_hit: bool

    # Results
    num_retrieved: int
    answer_length: int

    # User feedback (optional)
    user_score: Optional[float] = None
    user_feedback: Optional[str] = None


class PerformanceWindow(BaseModel):
    """Performance metrics for a time window."""

    window_start: datetime
    window_end: datetime
    num_queries: int

    # Latency
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float

    # Cache performance
    cache_hit_rate: float

    # Throughput
    queries_per_second: float

    # Quality (if feedback available)
    avg_user_score: Optional[float] = None
    satisfaction_rate: Optional[float] = None


class OnlineEvaluationMonitor:
    """Monitor RAG system performance in production.

    Tracks real-time metrics with sliding time windows.
    """

    def __init__(
        self,
        window_size_minutes: int = 60,
        langfuse_tracer=None,
    ):
        """Initialize online monitoring.

        Args:
            window_size_minutes: Size of sliding window for metrics
            langfuse_tracer: Optional Langfuse tracer for persistence
        """
        self.window_size = timedelta(minutes=window_size_minutes)
        self.langfuse = langfuse_tracer

        # Sliding window of queries
        self.recent_queries: Deque[ProductionQuery] = deque()

        # Metrics calculators
        self.retrieval_metrics = RetrievalMetrics()
        self.generation_metrics = GenerationMetrics()
        self.agent_metrics = AgentMetrics()

        logger.info(
            f"Initialized OnlineEvaluationMonitor with {window_size_minutes}min window"
        )

    def record_query(
        self,
        query_id: str,
        query: str,
        latency_ms: float,
        cache_hit: bool,
        num_retrieved: int,
        answer_length: int,
    ) -> None:
        """Record a production query.

        Args:
            query_id: Unique query identifier
            query: Query text
            latency_ms: Query latency in milliseconds
            cache_hit: Whether cache was hit
            num_retrieved: Number of documents retrieved
            answer_length: Length of generated answer
        """
        production_query = ProductionQuery(
            query_id=query_id,
            query=query,
            timestamp=datetime.now(),
            latency_ms=latency_ms,
            cache_hit=cache_hit,
            num_retrieved=num_retrieved,
            answer_length=answer_length,
        )

        self.recent_queries.append(production_query)

        # Clean old queries outside window
        self._clean_old_queries()

        # Log to Langfuse if available
        if self.langfuse:
            try:
                self.langfuse.trace(
                    trace_id=query_id,
                    metadata={
                        "latency_ms": latency_ms,
                        "cache_hit": cache_hit,
                        "num_retrieved": num_retrieved,
                    },
                )
            except Exception as e:
                logger.error(f"Error logging to Langfuse: {e}")

    def record_feedback(
        self,
        query_id: str,
        score: float,
        feedback: Optional[str] = None,
    ) -> None:
        """Record user feedback for a query.

        Args:
            query_id: Query identifier
            score: User score (0-1 scale)
            feedback: Optional text feedback
        """
        # Find query in recent queries
        for query in self.recent_queries:
            if query.query_id == query_id:
                query.user_score = score
                query.user_feedback = feedback
                break

        # Log to Langfuse if available
        if self.langfuse:
            try:
                self.langfuse.submit_feedback(
                    trace_id=query_id,
                    score=score,
                    comment=feedback,
                )
            except Exception as e:
                logger.error(f"Error logging feedback to Langfuse: {e}")

    def get_current_window(self) -> PerformanceWindow:
        """Get metrics for current time window.

        Returns:
            Performance window metrics
        """
        if not self.recent_queries:
            return PerformanceWindow(
                window_start=datetime.now() - self.window_size,
                window_end=datetime.now(),
                num_queries=0,
                avg_latency_ms=0.0,
                p50_latency_ms=0.0,
                p95_latency_ms=0.0,
                p99_latency_ms=0.0,
                cache_hit_rate=0.0,
                queries_per_second=0.0,
            )

        queries = list(self.recent_queries)
        latencies = [q.latency_ms for q in queries]
        cache_hits = sum(1 for q in queries if q.cache_hit)

        # Calculate percentiles
        import numpy as np

        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)

        # Time span
        window_start = min(q.timestamp for q in queries)
        window_end = max(q.timestamp for q in queries)
        time_span_seconds = (window_end - window_start).total_seconds()

        # Throughput
        qps = len(queries) / time_span_seconds if time_span_seconds > 0 else 0

        # User feedback metrics (if available)
        scores = [q.user_score for q in queries if q.user_score is not None]
        avg_user_score = np.mean(scores) if scores else None
        satisfaction_rate = (
            sum(1 for s in scores if s >= 0.7) / len(scores) if scores else None
        )

        return PerformanceWindow(
            window_start=window_start,
            window_end=window_end,
            num_queries=len(queries),
            avg_latency_ms=np.mean(latencies),
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            cache_hit_rate=cache_hits / len(queries),
            queries_per_second=qps,
            avg_user_score=avg_user_score,
            satisfaction_rate=satisfaction_rate,
        )

    def get_summary(self) -> Dict[str, float]:
        """Get summary of current performance.

        Returns:
            Dictionary of key performance metrics
        """
        window = self.get_current_window()

        summary = {
            "num_queries": window.num_queries,
            "avg_latency_ms": window.avg_latency_ms,
            "p95_latency_ms": window.p95_latency_ms,
            "cache_hit_rate": window.cache_hit_rate,
            "queries_per_second": window.queries_per_second,
        }

        if window.avg_user_score is not None:
            summary["avg_user_score"] = window.avg_user_score
            summary["satisfaction_rate"] = window.satisfaction_rate or 0.0

        return summary

    def check_alerts(self) -> List[str]:
        """Check for performance issues and return alerts.

        Returns:
            List of alert messages
        """
        window = self.get_current_window()
        alerts = []

        # Latency alerts
        if window.p95_latency_ms > 5000:  # >5 seconds
            alerts.append(
                f"HIGH LATENCY: P95 latency is {window.p95_latency_ms:.0f}ms"
            )

        # Cache hit rate alerts
        if window.cache_hit_rate < 0.3 and window.num_queries > 10:
            alerts.append(
                f"LOW CACHE HIT RATE: {window.cache_hit_rate:.1%} (expected >30%)"
            )

        # User satisfaction alerts
        if window.satisfaction_rate is not None and window.satisfaction_rate < 0.7:
            alerts.append(
                f"LOW SATISFACTION: {window.satisfaction_rate:.1%} users satisfied"
            )

        return alerts

    def _clean_old_queries(self) -> None:
        """Remove queries outside the time window."""
        cutoff_time = datetime.now() - self.window_size

        while self.recent_queries and self.recent_queries[0].timestamp < cutoff_time:
            self.recent_queries.popleft()

    def generate_report(self) -> str:
        """Generate human-readable performance report.

        Returns:
            Formatted report string
        """
        window = self.get_current_window()
        summary = self.get_summary()
        alerts = self.check_alerts()

        report = "Production Performance Report\n"
        report += "=" * 50 + "\n\n"

        report += f"Time Window: {window.window_start.strftime('%Y-%m-%d %H:%M')} to "
        report += f"{window.window_end.strftime('%Y-%m-%d %H:%M')}\n"
        report += f"Queries: {window.num_queries}\n\n"

        report += "Performance Metrics:\n"
        report += f"  - Avg Latency: {window.avg_latency_ms:.0f}ms\n"
        report += f"  - P50 Latency: {window.p50_latency_ms:.0f}ms\n"
        report += f"  - P95 Latency: {window.p95_latency_ms:.0f}ms\n"
        report += f"  - P99 Latency: {window.p99_latency_ms:.0f}ms\n\n"

        report += "Cache Performance:\n"
        report += f"  - Hit Rate: {window.cache_hit_rate:.1%}\n"
        report += f"  - Avg Speedup: {self._estimate_cache_speedup(window)}x\n\n"

        report += "Throughput:\n"
        report += f"  - Queries/Second: {window.queries_per_second:.2f}\n\n"

        if window.avg_user_score is not None:
            report += "User Satisfaction:\n"
            report += f"  - Avg Score: {window.avg_user_score:.2f}/1.0\n"
            report += f"  - Satisfaction Rate: {window.satisfaction_rate:.1%}\n\n"

        if alerts:
            report += "🚨 ALERTS:\n"
            for alert in alerts:
                report += f"  - {alert}\n"
        else:
            report += "✅ No alerts - system performing normally\n"

        return report

    def _estimate_cache_speedup(self, window: PerformanceWindow) -> float:
        """Estimate average speedup from caching.

        Args:
            window: Performance window

        Returns:
            Estimated speedup factor
        """
        queries = list(self.recent_queries)

        if not queries:
            return 1.0

        cached = [q.latency_ms for q in queries if q.cache_hit]
        uncached = [q.latency_ms for q in queries if not q.cache_hit]

        if not cached or not uncached:
            return 1.0

        import numpy as np

        avg_cached = np.mean(cached)
        avg_uncached = np.mean(uncached)

        return avg_uncached / avg_cached if avg_cached > 0 else 1.0
