"""LLM Orchestrator — 3-pass Claude API analysis pipeline.

Pass 1: Qualitative analysis (management, moat, risks) — Sonnet
Pass 2: Investment thesis synthesis — Sonnet
Pass 3: Comparative peer judgment — Haiku
"""

import json
import logging
import os
import time
from pathlib import Path

import anthropic

from boundless100x.compute_engine.metrics.base import MetricResult
from boundless100x.llm_layer.checklist import (
    build_flags_context,
    build_key_metrics_context,
    build_peer_comparison_text,
    build_promoter_context,
    build_qg_quadrant_context,
    build_quality_metrics_context,
    build_scores_summary,
)

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"


class LLMOrchestrator:
    """3-pass LLM analysis pipeline using Claude API."""

    def __init__(self, config: dict):
        llm_config = config.get("llm", {})
        self.enabled = llm_config.get("enabled", True)
        self.pass1_model = llm_config.get("pass1_model", "claude-sonnet-4-5-20250929")
        self.pass2_model = llm_config.get("pass2_model", "claude-sonnet-4-5-20250929")
        self.pass3_model = llm_config.get("pass3_model", "claude-haiku-4-5-20251001")
        self.max_tokens = llm_config.get("max_tokens", 2000)
        self.skip_pass1_if_no_ar = llm_config.get("skip_pass1_if_no_ar", True)

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable not set. "
                "Set it or disable LLM with llm.enabled: false in config.yaml"
            )
        self.client = anthropic.Anthropic(api_key=api_key)

        self._usage_log: list[dict] = []

    def use_deep_models(self) -> None:
        """Override Pass 1 & 2 to use Opus for deeper analysis."""
        self.pass1_model = "claude-opus-4-5-20251101"
        self.pass2_model = "claude-opus-4-5-20251101"
        self.max_tokens = 4000  # Opus benefits from more output room
        logger.info(
            f"Deep mode: Pass 1 & 2 → {self.pass1_model}, "
            f"max_tokens → {self.max_tokens}"
        )

    def run_analysis(
        self,
        ticker: str,
        company_name: str,
        sector: str,
        market_cap: float | None,
        metrics: dict[str, MetricResult],
        scores: dict,
        comparison: dict | None = None,
        peer_metadata: dict | None = None,
        annual_report_text: str | None = None,
        sector_context: str = "",
    ) -> dict:
        """Run the full 3-pass LLM analysis.

        Returns dict with keys: pass1, pass2, pass3, usage.
        """
        if not self.enabled:
            return {"skipped": True, "reason": "LLM disabled in config"}

        results = {}

        # Pass 1: Qualitative
        if annual_report_text or not self.skip_pass1_if_no_ar:
            logger.info("[LLM Pass 1] Qualitative analysis")
            results["pass1"] = self._run_pass1(
                ticker=ticker,
                company_name=company_name,
                sector=sector,
                market_cap=market_cap,
                metrics=metrics,
                scores=scores,
                annual_report_text=annual_report_text or "No annual report available.",
                sector_context=sector_context,
            )
        else:
            logger.info("[LLM Pass 1] Skipped (no annual report)")
            results["pass1"] = {
                "skipped": True,
                "reason": "No annual report available",
            }

        # Pass 2: Synthesis (always runs)
        logger.info("[LLM Pass 2] Investment thesis synthesis")
        results["pass2"] = self._run_pass2(
            ticker=ticker,
            company_name=company_name,
            sector=sector,
            metrics=metrics,
            scores=scores,
            pass1_output=results["pass1"],
        )

        # Pass 3: Comparative (only if peer comparison data exists)
        if comparison and comparison.get("companies"):
            logger.info("[LLM Pass 3] Comparative judgment")
            results["pass3"] = self._run_pass3(
                ticker=ticker,
                sector=sector,
                comparison=comparison,
                peer_metadata=peer_metadata or {},
                pass2_thesis=results["pass2"].get("thesis", ""),
            )
        else:
            logger.info("[LLM Pass 3] Skipped (no peer comparison data)")
            results["pass3"] = {
                "skipped": True,
                "reason": "No peer comparison data",
            }

        # Summarize usage
        results["usage"] = self._summarize_usage()

        return results

    # ── Pass 1: Qualitative ──

    def _run_pass1(
        self,
        ticker: str,
        company_name: str,
        sector: str,
        market_cap: float | None,
        metrics: dict[str, MetricResult],
        scores: dict,
        annual_report_text: str,
        sector_context: str,
    ) -> dict:
        template = self._load_template("pass1_qualitative.txt")

        prompt = template.format(
            ticker=ticker,
            company_name=company_name,
            sector=sector,
            market_cap=f"{market_cap:,.0f}" if market_cap else "N/A",
            quality_metrics=build_quality_metrics_context(metrics, scores),
            flags=build_flags_context(metrics),
            promoter_data=build_promoter_context(metrics),
            sector_context=sector_context or "No sector context available.",
            annual_report_text=annual_report_text[:3000],  # Truncate to stay in budget
        )

        return self._call_api(self.pass1_model, prompt, "pass1")

    # ── Pass 2: Synthesis ──

    def _run_pass2(
        self,
        ticker: str,
        company_name: str,
        sector: str,
        metrics: dict[str, MetricResult],
        scores: dict,
        pass1_output: dict,
    ) -> dict:
        template = self._load_template("pass2_synthesis.txt")

        # Format Pass 1 output for context
        if pass1_output.get("skipped"):
            pass1_text = "Qualitative analysis was skipped (no annual report)."
        else:
            pass1_text = json.dumps(pass1_output, indent=2, default=str)

        prompt = template.format(
            ticker=ticker,
            company_name=company_name,
            sector=sector,
            scores_summary=build_scores_summary(scores),
            key_metrics=build_key_metrics_context(metrics, scores),
            flags=build_flags_context(metrics),
            qg_quadrant=build_qg_quadrant_context(metrics),
            pass1_output=pass1_text[:2000],  # Truncate
        )

        return self._call_api(self.pass2_model, prompt, "pass2")

    # ── Pass 3: Comparative ──

    def _run_pass3(
        self,
        ticker: str,
        sector: str,
        comparison: dict,
        peer_metadata: dict,
        pass2_thesis: str,
    ) -> dict:
        template = self._load_template("pass3_comparative.txt")

        prompt = template.format(
            ticker=ticker,
            sector=sector,
            peer_comparison_table=build_peer_comparison_text(comparison),
            pass2_thesis=pass2_thesis[:500],  # Brief thesis context
            candidates_evaluated=peer_metadata.get("candidates_evaluated", "N/A"),
            size_filtered=peer_metadata.get("size_filtered_to", "N/A"),
            similarity_scores=json.dumps(
                peer_metadata.get("similarity_scores", {}), indent=2
            ),
            peer_quality_context=peer_metadata.get(
                "peer_quality_context",
                "Peer quality not assessed (LLM validation not run).",
            ),
        )

        return self._call_api(self.pass3_model, prompt, "pass3")

    # ── API Call ──

    def _call_api(self, model: str, prompt: str, pass_name: str) -> dict:
        """Call Claude API and parse JSON response."""
        start_time = time.time()

        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )

            elapsed = time.time() - start_time
            output_text = response.content[0].text

            # Log usage
            usage = {
                "pass": pass_name,
                "model": model,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "elapsed_seconds": round(elapsed, 1),
            }
            self._usage_log.append(usage)

            logger.info(
                f"  {pass_name}: {usage['input_tokens']}in + "
                f"{usage['output_tokens']}out tokens, {elapsed:.1f}s"
            )

            # Parse JSON from response
            return self._parse_json_response(output_text)

        except anthropic.APIError as e:
            logger.error(f"API error in {pass_name}: {e}")
            return {"error": str(e), "pass": pass_name}
        except Exception as e:
            logger.error(f"Error in {pass_name}: {e}")
            return {"error": str(e), "pass": pass_name}

    def _parse_json_response(self, text: str) -> dict:
        """Extract JSON from LLM response, handling markdown code blocks."""
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from code block
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            try:
                return json.loads(text[start:end].strip())
            except json.JSONDecodeError:
                pass

        # Try extracting from generic code block
        if "```" in text:
            start = text.index("```") + 3
            # Skip optional language identifier on same line
            newline = text.index("\n", start)
            start = newline + 1
            end = text.index("```", start)
            try:
                return json.loads(text[start:end].strip())
            except json.JSONDecodeError:
                pass

        # Try finding JSON object in text
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end != -1:
            try:
                return json.loads(text[brace_start : brace_end + 1])
            except json.JSONDecodeError:
                pass

        logger.warning("Could not parse JSON from LLM response")
        return {"raw_response": text, "parse_error": True}

    def _load_template(self, filename: str) -> str:
        """Load a prompt template file."""
        path = PROMPTS_DIR / filename
        with open(path) as f:
            return f.read()

    def _summarize_usage(self) -> dict:
        """Summarize total token usage and estimated cost."""
        total_input = sum(u["input_tokens"] for u in self._usage_log)
        total_output = sum(u["output_tokens"] for u in self._usage_log)
        total_time = sum(u["elapsed_seconds"] for u in self._usage_log)

        # Cost estimate per model (input/output per MTok)
        cost = 0.0
        for u in self._usage_log:
            if "opus" in u["model"]:
                cost += u["input_tokens"] * 15 / 1_000_000
                cost += u["output_tokens"] * 75 / 1_000_000
            elif "sonnet" in u["model"]:
                cost += u["input_tokens"] * 3 / 1_000_000
                cost += u["output_tokens"] * 15 / 1_000_000
            elif "haiku" in u["model"]:
                cost += u["input_tokens"] * 0.80 / 1_000_000
                cost += u["output_tokens"] * 4 / 1_000_000

        return {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "total_seconds": round(total_time, 1),
            "estimated_cost_usd": round(cost, 4),
            "passes": self._usage_log,
        }
