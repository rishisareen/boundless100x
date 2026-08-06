"""Boundless100x Service Layer — orchestrates the full analysis pipeline.

Pipeline: Data Fetch → Compute Engine → Scoring → LLM Analysis (2-pass) → Report Generation
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from boundless100x.action_policy import resolve_for_result
from boundless100x.data_fetcher.suite import DataFetcherSuite
from boundless100x.compute_engine.eligibility import (
    EligibilityEvaluator,
    effective_gates,
)
from boundless100x.compute_engine.engine import ComputeEngine
from boundless100x.compute_engine.scorer import SQGLPScorer
from boundless100x.compute_engine.metrics.base import MetricResult
from boundless100x.compute_engine.metrics.builtin.growth import compute_lever_decomposition_table
from boundless100x.llm_layer.checklist import build_sector_context
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
    eligibility: dict | None = None
    llm_analysis: dict | None = None
    # The action a report may display, resolved in code from the LLM's
    # suggestion plus the deterministic eligibility verdict and score
    # coverage. See action_policy.resolve_final_action.
    final_action: dict | None = None
    errors: list[str] = field(default_factory=list)


class Boundless100xService:
    """Main service — runs the full SQGLP analysis pipeline.

    Usage:
        svc = Boundless100xService()
        result = svc.analyze("ASTRAL")
        print(result.scores["composite"])  # 7.2
        print(result.scores["elements"]["growth"])  # 6.4
    """

    def __init__(self, config_path: str | None = None, config: dict | None = None):
        if config is not None:
            self.config = config
        else:
            path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
            with open(path) as f:
                self.config = yaml.safe_load(f)

        self.suite = DataFetcherSuite(self.config)
        self.engine = ComputeEngine(macro=self.config.get("macro", {}))
        # Resolved through the same helper the registry hash uses, so the
        # regime recorded in score history is always the regime enforced here.
        self.eligibility = EligibilityEvaluator(effective_gates(self.engine.gates))
        self.scorer = SQGLPScorer(
            self.engine.metrics,
            self.engine.element_weights,
            history_waiver_mcap=self.engine.master.get("history_waiver_mcap"),
        )

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

        # Financials and price are load-bearing for every downstream stage —
        # scoring, eligibility, and the LLM synthesis all silently degrade to
        # near-empty input rather than erroring, which can still produce a
        # complete-looking report and recommendation from no real data. Stop
        # here rather than let that happen.
        CORE_SOURCES = ("financials", "price")
        source_status = result.data.get("source_status", {})
        missing_core = [s for s in CORE_SOURCES if not source_status.get(s, "").startswith("ok")]
        if missing_core:
            reasons = "; ".join(f"{s}: {source_status.get(s, 'no status recorded')}" for s in missing_core)
            result.errors.append(
                f"Fatal: core data missing ({reasons}) — stopping before scoring, "
                "no recommendation can be produced from this."
            )
            logger.error(f"{ticker}: fatal — core data missing: {reasons}")
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

        # Stage 3.6: 100x eligibility gates (conjunctive, separate from composite)
        try:
            result.eligibility = self.eligibility.evaluate(result.metrics)
            logger.info(f"100x eligibility: {result.eligibility['verdict']}")
        except Exception as e:
            result.errors.append(f"Eligibility evaluation failed: {e}")
            logger.error(f"Eligibility failed for {ticker}: {e}")

        # Stage 3.5: Growth Decomposition (v4)
        try:
            financials = result.data.get("financials")
            if financials is not None and not financials.empty:
                result.growth_decomposition = compute_lever_decomposition_table(
                    result.data, macro=self.engine.macro
                )
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
                    sector_context=build_sector_context(metadata),
                    growth_decomposition=result.growth_decomposition,
                    # Explanatory context only — the guard is Stage 4.5 below.
                    eligibility=result.eligibility,
                )

                usage = result.llm_analysis.get("usage", {})
                logger.info(
                    f"LLM analysis complete: {usage.get('total_tokens', 0)} tokens, "
                    f"~${usage.get('estimated_cost_usd', 0):.4f}"
                )
            except Exception as e:
                result.errors.append(f"LLM analysis failed: {e}")
                logger.error(f"LLM analysis failed: {e}")

        # Stage 4.5: Resolve the displayable action in deterministic code.
        # Pass 2 is given the verdict above, but prompt compliance cannot be
        # the guard that stops a `strong_buy` appearing beside a failed gate.
        result.final_action = self.resolve_action(result)

        return result

    @staticmethod
    def resolve_action(result: AnalysisResult) -> dict | None:
        """The action a report may display, or None when there is no LLM view.

        Derived from the metrics every time; see action_policy for why a
        stored `final_action` is never an input here.
        """
        return resolve_for_result(result)

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

