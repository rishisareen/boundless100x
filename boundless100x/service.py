"""Boundless100x Service Layer — orchestrates the full analysis pipeline.

Pipeline: Data Fetch → Compute Engine → Scoring → LLM Analysis (2-pass) → Report Generation
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from boundless100x.action_policy import resolve_for_result
from boundless100x.data_fetcher.suite import DataFetcherSuite
from boundless100x.data_fetcher.fetch_announcements import build_announcements_context
from boundless100x.compute_engine.eligibility import (
    EligibilityEvaluator,
    effective_gates,
)
from boundless100x.compute_engine.engine import ComputeEngine
from boundless100x.compute_engine.scorer import SQGLPScorer
from boundless100x.compute_engine.sector import SectorApplicability
from boundless100x.compute_engine.metrics.base import MetricResult
from boundless100x.compute_engine.metrics.builtin.growth import compute_lever_decomposition_table
from boundless100x.llm_layer import forward_growth
from boundless100x.llm_layer.checklist import build_sector_context
from boundless100x.llm_layer.orchestrator import LLMOrchestrator
from boundless100x.llm_layer.transport import COST_BASIS_ESTIMATED
from boundless100x import score_history, trajectory

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"

# One per ticker, beside the annual report PDFs the extraction read.
FORWARD_GROWTH_SIDECAR = "forward_growth.extraction.json"


class RegistryValidationError(ValueError):
    """The metric registry did not validate, so nothing can be computed.

    A `ValueError` subclass, because that is what `ComputeEngine` already
    raises and an existing caller catching one must keep catching this. A
    *distinct* type, because the CLI has to tell "a YAML file on disk is
    malformed and here is what to do about it" apart from every other
    `ValueError` a command can meet, and print the one rather than a traceback.

    It carries no verdict of its own — validation stays where it is, and a
    registry that fails it still refuses to run.
    """


def load_config(config_path: str | None = None) -> dict:
    """The pipeline config, from one place.

    The CLI needs it before it builds a service (to resolve the snapshot root),
    and the service needs it to build anything at all. Two loaders would be two
    answers to "what is configured".
    """
    with open(Path(config_path) if config_path else DEFAULT_CONFIG_PATH) as f:
        return yaml.safe_load(f)


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
    # Score trajectory read off the append-only history (Stage 4.7). Carries
    # an explicit insufficient-history outcome rather than a zero, because on
    # the day this landed that was the answer for every ticker.
    momentum: dict | None = None
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
        self.config = config if config is not None else load_config(config_path)

        self.suite = DataFetcherSuite(self.config)
        # The engine is the one collaborator that may not degrade: a registry
        # that fails validation has to stop the run, because a metric missing
        # its `presentation:` block would otherwise reach a reader as a bare
        # number. What it must not do is *end* the run in a traceback.
        # `compute_engine/metrics/custom/` is the documented extension point
        # — "adding a metric = 1 YAML entry + 1 function" — so a drop-in that
        # forgets a required key is an ordinary hand-edit, and every CLI
        # command building a service died on it with a stack trace.
        #
        # Shape follows `LLMOrchestrator` below: catch the `ValueError` at
        # construction and hand the caller something it can act on. The only
        # difference is that this one is fatal rather than degrading.
        try:
            self.engine = ComputeEngine(macro=self.config.get("macro", {}))
        except ValueError as e:
            raise RegistryValidationError(
                f"{e} — each was logged as a REGISTRY ERROR naming the file, "
                f"the metric and the key at fault. Every metric definition "
                f"needs a presentation block (unit, direction, bands); fix or "
                f"remove the definition — most often a drop-in under "
                f"compute_engine/metrics/custom/ — and run again."
            ) from e
        # Where scored runs are recorded. None means the module default; a
        # caller scoring into a scratch store (tests, the future simulator)
        # points this elsewhere so the real history stays organic.
        self.history_path = self.config.get("output", {}).get("score_history_path")
        # Resolved through the same helper the registry hash uses, so the
        # regime recorded in score history is always the regime enforced here.
        self.eligibility = EligibilityEvaluator(effective_gates(self.engine.gates))
        # The applicability table now withdraws metrics from a composite, so a
        # typo in it is a scoring error rather than a display one and stops the
        # run the same way a bad registry does. The report layer keeps its own
        # independent instance behind `_reading_or_none` and its own degrade
        # path: a table that fails here never reaches that layer at all.
        try:
            applicability = SectorApplicability(self.engine.metrics.keys())
        except ValueError as e:
            raise RegistryValidationError(
                f"{e} — each was logged as a SECTOR APPLICABILITY ERROR naming "
                f"the sector, the metric and what was wrong. This table decides "
                f"which metrics are excluded from a company's score, so a "
                f"malformed entry would silently change every composite it "
                f"touches; fix compute_engine/sector_applicability.yaml and run "
                f"again."
            ) from e

        self.scorer = SQGLPScorer(
            self.engine.metrics,
            self.engine.element_weights,
            history_waiver_mcap=self.engine.master.get("history_waiver_mcap"),
            applicability=applicability,
        )

        # LLM orchestrator — constructed only when the configured provider's
        # precondition holds: an API key for `anthropic`, a `claude` on PATH for
        # `claude_cli`. Both raise ValueError, so an unusable provider degrades
        # to compute-only the same way a missing key always has.
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
        include_momentum: bool = True,
    ) -> AnalysisResult:
        """Run full analysis pipeline for a company.

        Args:
            ticker: NSE symbol (e.g., "ASTRAL").
            bse_code: BSE scrip code (optional).
            deep: If True, use Opus model for Pass 1 & 2 (deeper analysis).
            use_llm: If True, run 2-pass LLM analysis after compute.
            annual_report_text: Pre-extracted annual report text for Pass 1.
            include_momentum: If True, read score trajectory at Stage 4.7.
                Only the report renders it, so callers that never build one
                skip a full re-read of the score-history log — `watchlist
                advance` does exactly that, once per tracked ticker.

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

        # Deep mode is resolved here rather than at Stage 4 because Stage 1.5
        # below is also an LLM call, and the model it used is recorded in the
        # extraction cache's version block. Resolved per call in both
        # directions: the orchestrator is long-lived and the service is
        # documented as reusable, so without the reset one deep run would
        # silently make every later shallow one deep as well.
        if use_llm and self._llm:
            if deep:
                self._llm.use_deep_models()
            else:
                self._llm.use_configured_models()

        # Stage 1.5: Forward-growth extraction (Phase 2)
        try:
            self._forward_growth_stage(ticker, result.data, use_llm)
        except Exception as e:
            result.errors.append(f"Forward-growth extraction failed: {e}")
            logger.error(f"Forward-growth extraction failed for {ticker}: {e}")

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
            # The sector decides which metrics are meaningless here. A ticker
            # fetched before the breadcrumb fix carries no sector, which reads
            # indeterminate and scores everything — the old regime, and the
            # right default: an unknown sector must not be able to withdraw a
            # metric any more than it may excuse one.
            sector = (result.data.get("metadata") or {}).get("sector")
            result.scores = self.scorer.score(result.metrics, sector=sector)
            excluded = result.scores.get("not_applicable") or {}
            if excluded:
                logger.info(
                    f"{ticker}: {len(excluded)} metric(s) not scored — they "
                    f"measure nothing for {sector}: {', '.join(sorted(excluded))}"
                )
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
                    # Rendered here rather than in the orchestrator so the
                    # seam stays one-directional: `llm_layer` is handed text,
                    # never a DataFrame it would have to know the shape of.
                    announcements_context=build_announcements_context(
                        result.data.get("announcements")
                    ),
                )

                usage = result.llm_analysis.get("usage", {})
                logger.info(
                    f"LLM analysis complete: {usage.get('total_tokens', 0)} tokens, "
                    f"{usage.get('cost_basis', COST_BASIS_ESTIMATED)} "
                    f"${usage.get('estimated_cost_usd', 0):.4f} "
                    f"({usage.get('provider', 'anthropic')})"
                )
            except Exception as e:
                result.errors.append(f"LLM analysis failed: {e}")
                logger.error(f"LLM analysis failed: {e}")

        # Stage 4.5: Resolve the displayable action in deterministic code.
        # Pass 2 is given the verdict above, but prompt compliance cannot be
        # the guard that stops a `strong_buy` appearing beside a failed gate.
        result.final_action = self.resolve_action(result)

        # Stage 4.6: Record the run in append-only score history. Momentum is
        # computed from these rows in Phase 2, and a run not written when it
        # happened cannot be recovered later — so this runs for every scored
        # analysis, LLM or not. A failure here must never cost the caller the
        # analysis it just paid for.
        try:
            score_history.append_run(
                result,
                self.engine.registry_hash,
                path=self.history_path,
                forward_signal_hash=self.engine.forward_signal_hash,
            )
        except Exception as e:
            result.errors.append(f"Score history write failed: {e}")
            logger.error(f"Score history write failed for {ticker}: {e}")

        # Stage 4.7: Read the trajectory back, including the row just written,
        # so the report shows momentum through today rather than up to
        # yesterday. Routed through the service (not the report generator) so
        # the per-caller history redirect the test-isolation fixture depends on
        # still applies. A failure here costs a signal, never the analysis.
        if not include_momentum:
            return result

        try:
            result.momentum = trajectory.compute_momentum(
                ticker, path=self.history_path
            )
            logger.info(f"Score momentum: {result.momentum['status']}")
        except Exception as e:
            result.errors.append(f"Score momentum read failed: {e}")
            logger.error(f"Score momentum read failed for {ticker}: {e}")

        return result

    # ── Stage 1.5: forward-growth extraction ──

    def _forward_growth_sidecar(self, ticker: str, data: dict) -> Path:
        """Where a ticker's validated extraction is cached.

        Beside the annual report PDFs it was read from, keyed by BSE code for
        the same reason those are — that is the directory the reports live in.
        """
        meta = data.get("metadata", {}) or {}
        key = meta.get("bse_code") or ticker
        return (
            Path(self.suite.raw_data_dir) / str(key) / "annual_reports"
            / FORWARD_GROWTH_SIDECAR
        )

    def _forward_growth_stage(self, ticker: str, data: dict, use_llm: bool) -> None:
        """Put validated forward-growth extraction into `data["forward_growth"]`.

        **Hydration and extraction are separately gated, and that split is the
        load-bearing part of this stage.** Reading a valid cache runs on every
        run, including `use_llm=False`; only *creating or refreshing* one calls
        the model. Gating the whole stage on `use_llm` would mean the cache is
        never read on the very paths it exists to serve — `watchlist advance`
        re-scores with `use_llm=False`, so its sub-metrics would read
        indeterminate forever no matter how many extractions had been paid for.

        The three outcomes are deliberately distinguishable:

          * key absent — we could not look (no sections, or no valid cache and
            no LLM available). A sub-metric reads indeterminate.
          * `{}` — we looked and there was nothing readable: every section was
            `fallback` or failed the content gate. Determinate, and free.
          * populated — extraction ran or its cache was hydrated.

        Called at Stage 1.5 rather than inside `DataFetcherSuite.fetch_all`,
        which takes no `use_llm` argument: wiring it there would fire a paid
        call on every `--no-llm` run, including `screen`'s per-candidate
        `analyze_quick` and every `watchlist advance`.
        """
        sections = data.get("annual_report_sections") or {}
        if not sections:
            return

        # Shared with the sweep's dry run, so what is priced is what is sent.
        plan = forward_growth.plan_submission(self.config, sections, llm=self._llm)
        gated, submission = plan["gated"], plan["submission"]

        if not submission:
            # Nothing survived provenance and the content gate. That is a
            # determinate empty answer, not an unknown one, and it costs
            # nothing to reach — so no call is made.
            data["forward_growth"] = {}
            logger.info(
                f"[Stage 1.5] {ticker}: no usable annual-report sections "
                f"(after the content gate) — no extraction call"
            )
            return

        model = plan["model"]
        sidecar = self._forward_growth_sidecar(ticker, data)

        cached = forward_growth.read_sidecar(sidecar, submission, model)
        if cached is not None:
            data["forward_growth"] = cached
            logger.info(f"[Stage 1.5] {ticker}: forward-growth read from cache")
            return

        if not (use_llm and self._llm):
            logger.info(
                f"[Stage 1.5] {ticker}: no valid extraction cache and no LLM this "
                f"run — forward-growth sub-metrics will read indeterminate"
            )
            return

        logger.info(
            f"[Stage 1.5] {ticker}: extracting forward growth from "
            f"{len(submission)} report year(s)"
        )
        company_name = (data.get("metadata", {}) or {}).get("name", ticker)
        raw = self._llm.run_forward_growth_extraction(ticker, company_name, submission)
        validated = forward_growth.validate_extraction(raw, submission, gated)

        if validated.get("call_failed"):
            # An outage is not a finding. Caching it would serve "nothing was
            # extracted" as a determinate answer on every later run, since
            # nothing re-extracts until the text, schema, prompt or model
            # changes — so one rate-limit would disable this ticker for good.
            raise RuntimeError(
                f"extraction call failed for {ticker}; not cached so the next "
                f"run retries"
            )

        data["forward_growth"] = validated["years"]
        # Keep the discard reasons. They are the only record of *why* a
        # ticker's forward-growth coverage came up empty, and a log line does
        # not survive the run — without this, an empty result is
        # indistinguishable from a report that genuinely said nothing.
        data["forward_growth_discarded"] = validated["discarded"]
        forward_growth.write_sidecar(
            sidecar, validated["years"], submission, model,
            discarded=validated["discarded"],
        )

    @staticmethod
    def resolve_action(result: AnalysisResult) -> dict | None:
        """The action a report may display, or None when there is no LLM view.

        Derived from the metrics every time; see action_policy for why a
        stored `final_action` is never an input here.
        """
        return resolve_for_result(result)

    def analyze_quick(self, ticker: str) -> AnalysisResult:
        """Quick analysis without LLM — for screening.

        Skips the momentum read for the same reason `advance_ticker` does:
        screening runs this once per candidate over the whole universe and
        never reads `result.momentum`, so asking for it would re-parse the
        entire score-history log per candidate for nothing.
        """
        return self.analyze(ticker, use_llm=False, include_momentum=False)

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

