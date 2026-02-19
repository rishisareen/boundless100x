"""Boundless100x Service Layer — orchestrates the full analysis pipeline.

Pipeline: Data Fetch → Compute Engine → Scoring → Peer Discovery → Peer Compute
       → Comparison → LLM Analysis (3-pass)
Future: → Report Generation
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from boundless100x.data_fetcher.suite import DataFetcherSuite
from boundless100x.data_fetcher.peer_discovery import PeerDiscovery, PeerResult
from boundless100x.compute_engine.engine import ComputeEngine
from boundless100x.compute_engine.scorer import SQGLPScorer
from boundless100x.compute_engine.peer_comparison import build_comparison_table
from boundless100x.compute_engine.metrics.base import MetricResult
from boundless100x.compute_engine.metrics.builtin.growth import compute_lever_decomposition_table
from boundless100x.llm_layer.orchestrator import LLMOrchestrator

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


@dataclass
class AnalysisResult:
    """Complete analysis output for a single company."""

    ticker: str
    data: dict = field(default_factory=dict)
    metrics: dict[str, MetricResult] = field(default_factory=dict)
    scores: dict = field(default_factory=dict)
    growth_decomposition: dict | None = None
    peers: PeerResult | None = None
    peer_metrics: dict[str, dict[str, MetricResult]] = field(default_factory=dict)
    comparison: dict = field(default_factory=dict)
    llm_analysis: dict | None = None
    errors: list[str] = field(default_factory=list)


class Boundless100xService:
    """Main service — runs the full SQGLP analysis pipeline.

    Usage:
        svc = Boundless100xService()
        result = svc.analyze("ASTRAL")
        print(result.scores["composite"])  # 7.2
        print(result.peers.direct_competitors)  # ["SUPREMEIND", ...]
    """

    def __init__(self, config_path: str | None = None, config: dict | None = None):
        if config is not None:
            self.config = config
        else:
            path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
            with open(path) as f:
                self.config = yaml.safe_load(f)

        self.suite = DataFetcherSuite(self.config)
        self.peer_discovery = PeerDiscovery(self.config)
        self.engine = ComputeEngine()
        self.scorer = SQGLPScorer(self.engine.metrics, self.engine.element_weights)

        # LLM orchestrator (lazy init — only when API key is available)
        self._llm = None
        if self.config.get("llm", {}).get("enabled", True):
            try:
                self._llm = LLMOrchestrator(self.config)
            except ValueError as e:
                logger.warning(f"LLM not available: {e}")

    def analyze(
        self,
        ticker: str,
        bse_code: str | None = None,
        skip_peers: bool = False,
        use_llm: bool = True,
        deep: bool = False,
        max_peer_compute: int | None = None,
        annual_report_text: str | None = None,
    ) -> AnalysisResult:
        """Run full analysis pipeline for a company.

        Args:
            ticker: NSE symbol (e.g., "ASTRAL").
            bse_code: BSE scrip code (optional).
            skip_peers: If True, skip peer discovery and comparison.
            deep: If True, use Opus model for Pass 1 & 2 (deeper analysis).
            use_llm: If True, run 3-pass LLM analysis after compute.
            max_peer_compute: Max peers to run compute engine on (default: from config).
            annual_report_text: Pre-extracted annual report text for Pass 1.

        Returns:
            AnalysisResult with all computed data.
        """
        result = AnalysisResult(ticker=ticker)
        max_peer_compute = max_peer_compute or self.config.get(
            "peer_discovery", {}
        ).get("max_peers", 5)

        # Stage 1: Data Fetch
        logger.info(f"[Stage 1] Fetching data for {ticker}")
        try:
            result.data = self.suite.fetch_all(ticker, bse_code=bse_code)
        except Exception as e:
            result.errors.append(f"Data fetch failed: {e}")
            logger.error(f"Data fetch failed for {ticker}: {e}")
            return result

        # Stage 2: Compute Engine (target)
        logger.info(f"[Stage 2] Running compute engine for {ticker}")
        try:
            result.metrics = self.engine.run_all(result.data)
            ok_count = sum(1 for m in result.metrics.values() if m.ok)
            logger.info(
                f"Computed {ok_count}/{len(result.metrics)} metrics for {ticker}"
            )
        except Exception as e:
            result.errors.append(f"Compute engine failed: {e}")
            logger.error(f"Compute engine failed for {ticker}: {e}")
            return result

        # Stage 3: SQGLP Scoring
        logger.info(f"[Stage 3] Scoring {ticker}")
        try:
            result.scores = self.scorer.score(result.metrics)
            logger.info(
                f"SQGLP composite: {result.scores.get('composite', 'N/A')}/10"
            )
        except Exception as e:
            result.errors.append(f"Scoring failed: {e}")
            logger.error(f"Scoring failed for {ticker}: {e}")

        # Stage 3.5: Growth Decomposition (v4)
        try:
            financials = result.data.get("financials")
            if financials is not None and not financials.empty:
                result.growth_decomposition = compute_lever_decomposition_table(result.data)
                logger.info("Growth decomposition computed")
        except Exception as e:
            logger.warning(f"Growth decomposition failed: {e}")

        if skip_peers:
            return result

        # Stage 4: Peer Discovery
        logger.info(f"[Stage 4] Discovering peers for {ticker}")
        try:
            result.peers = self.peer_discovery.discover(
                ticker, target_data=result.data, use_llm=use_llm
            )
            logger.info(
                f"Found {len(result.peers.direct_competitors)} direct competitors"
            )
        except Exception as e:
            result.errors.append(f"Peer discovery failed: {e}")
            logger.error(f"Peer discovery failed for {ticker}: {e}")
            return result

        # Stage 5: Compute Engine (peers)
        peers_to_compute = result.peers.direct_competitors[:max_peer_compute]
        logger.info(
            f"[Stage 5] Computing metrics for {len(peers_to_compute)} peers"
        )

        for peer_ticker in peers_to_compute:
            try:
                peer_data = self.suite.fetch_all(peer_ticker)
                peer_results = self.engine.run_all(peer_data)
                result.peer_metrics[peer_ticker] = peer_results
                ok_count = sum(1 for m in peer_results.values() if m.ok)
                logger.info(
                    f"  {peer_ticker}: {ok_count}/{len(peer_results)} metrics"
                )
            except Exception as e:
                logger.warning(f"Peer compute failed for {peer_ticker}: {e}")
                result.errors.append(f"Peer {peer_ticker} failed: {e}")

        # Stage 6: Peer Comparison Table
        if result.peer_metrics:
            logger.info("[Stage 6] Building peer comparison table")
            try:
                result.comparison = build_comparison_table(
                    self.engine, ticker, result.metrics, result.peer_metrics
                )
                logger.info(
                    f"Comparison table: {len(result.comparison.get('metrics', []))} metrics, "
                    f"{len(result.comparison.get('companies', {}))} companies"
                )
            except Exception as e:
                result.errors.append(f"Comparison table failed: {e}")
                logger.error(f"Comparison table failed: {e}")

        # Stage 7: LLM Analysis (3-pass)
        if use_llm and self._llm:
            if deep:
                self._llm.use_deep_models()
            logger.info("[Stage 7] Running LLM analysis (3-pass)")
            try:
                metadata = result.data.get("metadata", {})
                company_name = metadata.get("name", ticker)
                sector = metadata.get("sector", "Unknown")
                market_cap = metadata.get("Market Cap")

                peer_metadata = (
                    result.peers.discovery_metadata if result.peers else {}
                )

                # Resolve annual report text: user-provided overrides auto-extracted
                ar_text = annual_report_text or result.data.get("annual_report_text")

                result.llm_analysis = self._llm.run_analysis(
                    ticker=ticker,
                    company_name=company_name,
                    sector=sector,
                    market_cap=float(market_cap) if market_cap else None,
                    metrics=result.metrics,
                    scores=result.scores,
                    comparison=result.comparison,
                    peer_metadata=peer_metadata,
                    annual_report_text=ar_text,
                    growth_decomposition=result.growth_decomposition,
                )

                usage = result.llm_analysis.get("usage", {})
                logger.info(
                    f"LLM analysis complete: {usage.get('total_tokens', 0)} tokens, "
                    f"~${usage.get('estimated_cost_usd', 0):.4f}"
                )
            except Exception as e:
                result.errors.append(f"LLM analysis failed: {e}")
                logger.error(f"LLM analysis failed: {e}")

        return result

    def analyze_quick(self, ticker: str) -> AnalysisResult:
        """Quick analysis without peers or LLM — for screening."""
        return self.analyze(ticker, skip_peers=True, use_llm=False)

    def get_element_summary(self, result: AnalysisResult) -> dict:
        """Get a readable summary of SQGLP element scores."""
        elements = result.scores.get("elements", {})
        weight_map = self.engine.element_weights

        summary = {}
        for el, score in elements.items():
            weight = weight_map.get(el, 0)
            summary[el] = {
                "score": round(score, 1) if score is not None else None,
                "weight": f"{weight*100:.0f}%",
                "weighted": round(score * weight, 2) if score is not None else None,
            }

        summary["composite"] = result.scores.get("composite")
        return summary

    def get_peer_ranking(self, result: AnalysisResult) -> dict:
        """Get target's rank among peers for each metric."""
        if not result.comparison:
            return {}

        ranks = result.comparison.get("ranks", {}).get(result.ticker, {})
        total_companies = len(result.comparison.get("companies", {}))
        best = result.comparison.get("best_in_class", {})

        return {
            "ranks": ranks,
            "total_companies": total_companies,
            "best_in_class": best,
            "top_quartile_metrics": [
                m for m, r in ranks.items() if r == 1
            ],
        }
