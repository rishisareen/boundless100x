"""Boundless100x Service Layer — orchestrates the full analysis pipeline.

Pipeline: Data Fetch → Compute Engine → Scoring → LLM Analysis (2-pass) → Report Generation
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from boundless100x.data_fetcher.suite import DataFetcherSuite
from boundless100x.compute_engine.engine import ComputeEngine
from boundless100x.compute_engine.scorer import SQGLPScorer
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
        use_llm: bool = True,
        deep: bool = False,
        annual_report_text: str | None = None,
    ) -> AnalysisResult:
        """Run full analysis pipeline for a company.

        Args:
            ticker: NSE symbol (e.g., "ASTRAL").
            bse_code: BSE scrip code (optional).
            deep: If True, use Opus model for Pass 1 & 2 (deeper analysis).
            use_llm: If True, run 2-pass LLM analysis after compute.
            annual_report_text: Pre-extracted annual report text for Pass 1.

        Returns:
            AnalysisResult with all computed data.
        """
        result = AnalysisResult(ticker=ticker)

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

        # Stage 4: LLM Analysis (2-pass)
        if use_llm and self._llm:
            if deep:
                self._llm.use_deep_models()
            logger.info("[Stage 4] Running LLM analysis (2-pass)")
            try:
                metadata = result.data.get("metadata", {})
                company_name = metadata.get("name", ticker)
                sector = metadata.get("sector", "Unknown")
                market_cap = metadata.get("Market Cap")

                # Resolve annual report text: user-provided overrides auto-extracted
                ar_text = annual_report_text or result.data.get("annual_report_text")

                result.llm_analysis = self._llm.run_analysis(
                    ticker=ticker,
                    company_name=company_name,
                    sector=sector,
                    market_cap=float(market_cap) if market_cap else None,
                    metrics=result.metrics,
                    scores=result.scores,
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
        """Quick analysis without LLM — for screening."""
        return self.analyze(ticker, use_llm=False)

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

