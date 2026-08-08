"""LLM Orchestrator — Claude API analysis pipeline.

Pass 1: Qualitative analysis (management, moat, risks) — Sonnet
Pass 2: Investment thesis synthesis — Sonnet

Plus a separate forward-growth extraction call (Phase 2), which is not a
"pass": it runs upstream of the compute engine at Stage 1.5 and reaches metrics
only as data. See `llm_layer/forward_growth.py` for why that seam matters.
"""

import json
import logging
from datetime import date
import time
from pathlib import Path

from boundless100x.compute_engine.metrics.base import MetricResult
from boundless100x.lifecycle.checkpoints import vocabulary_prompt_block
from boundless100x.llm_layer import forward_growth
from boundless100x.llm_layer.checklist import (
    build_eligibility_context,
    build_flags_context,
    build_growth_decomposition_context,
    build_key_metrics_context,
    build_promoter_context,
    build_qg_quadrant_context,
    build_quality_metrics_context,
    build_scores_summary,
)
from boundless100x.llm_layer.transport import TransportError, build_transport

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"

DEFAULT_MODEL = "claude-sonnet-5"
DEEP_MODEL = "claude-opus-5"

# Thinking is on by default on the Claude 5 family and is billed against the
# same ceiling as the response text, so a budget sized for the JSON alone
# truncates mid-object. The cap costs nothing unless it is reached.
DEFAULT_MAX_TOKENS = 16000

# USD per million tokens, (input, output), matched on the family name in the
# model id. Kept here as the one price table in the repo: the extraction sweep
# prices a run *before* spending on it, and an estimate computed from a second
# copy of these numbers would drift out of agreement with the bill.
#
# Every model any config may name must appear here. `estimate_cost` returns
# 0.0 for an unrecognised id, and the sweep's ceiling meters on that same
# number — so a missing family prints a ceiling and enforces nothing.
# Sonnet is listed at list price while a lower introductory rate runs; an
# estimate above the bill is the safe direction for a spend ceiling.
MODEL_PRICING = {
    "fable": (10.0, 50.0),
    "opus": (5.0, 25.0),
    "sonnet": (3.0, 15.0),
    "haiku": (1.0, 5.0),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """What a call of this shape costs, in USD. Unknown models price at zero.

    Zero rather than a guess: a made-up price on an unrecognised model would
    read as a real estimate, and the sweep's cost ceiling would enforce a
    number nobody chose.
    """
    for family, (per_input, per_output) in MODEL_PRICING.items():
        if family in (model or ""):
            return (
                input_tokens * per_input / 1_000_000
                + output_tokens * per_output / 1_000_000
            )
    return 0.0


def forward_growth_model(config: dict) -> str:
    """The extraction model a run will use, resolvable without an API key.

    The sidecar's version block records the model that produced it, so a
    hydration-only run (no key, or `--no-llm`) has to be able to name the same
    model an extracting run would have used — otherwise every cached
    extraction would read as stale on exactly the paths the cache exists for.
    """
    llm_config = config.get("llm", {})
    return llm_config.get("forward_growth_model") or llm_config.get(
        "pass1_model", DEFAULT_MODEL
    )


def forward_growth_char_budget(config: dict) -> int:
    """How much of each gated section reaches the extractor.

    Resolved here for the same reason the model is: the service reads it on
    the hydration path, where no orchestrator may exist, and two copies of the
    lookup could drift into submitting a different payload than the one the
    cache was keyed on.
    """
    return config.get("llm", {}).get(
        "forward_growth_char_budget", forward_growth.DEFAULT_CHAR_BUDGET
    )


class LLMOrchestrator:
    """The LLM analysis pipeline, over whichever transport `llm.provider` names.

    Which provider ran is visible only in the usage metadata. Everything else
    here — prompts, parsing, budgets, the deep-mode toggle — is written once and
    behaves identically on both. See `llm_layer/transport.py` for why.
    """

    def __init__(self, config: dict):
        # Kept so a deep-mode override can be undone from the same source the
        # defaults came from, rather than from remembered values.
        self._config = config
        llm_config = config.get("llm", {})
        self.enabled = llm_config.get("enabled", True)
        self.pass1_model = llm_config.get("pass1_model", DEFAULT_MODEL)
        self.pass2_model = llm_config.get("pass2_model", DEFAULT_MODEL)
        # What `--deep` swaps to. Config-driven for the same reason the other
        # models are: moving a generation should not need a code change.
        self.deep_model = llm_config.get("deep_model", DEEP_MODEL)
        # The extraction call is deliberately its own model setting: it is a
        # structured-extraction task rather than a judgement one, so it need
        # not track whatever Pass 1 and 2 are set to.
        self.forward_growth_model = forward_growth_model(config)
        self.max_tokens = llm_config.get("max_tokens", DEFAULT_MAX_TOKENS)
        self.skip_pass1_if_no_ar = llm_config.get("skip_pass1_if_no_ar", True)
        # Per submitted section, per report year. Sits alongside
        # pass1_ar_char_budget rather than sharing it: Pass 1 reads a combined
        # single string for background, while extraction reads each gated
        # section separately and needs enough of each to find a target in.
        self.forward_growth_char_budget = forward_growth_char_budget(config)
        # How much annual-report text Pass 1 may read. Config-driven because
        # the fetcher now caps each extracted section separately: a literal
        # here would silently overrule those caps, and raising a section cap
        # to get more MD&A into the prompt would have no effect.
        self.pass1_ar_char_budget = llm_config.get("pass1_ar_char_budget", 3000)

        # How a call leaves this machine — the API, or headless Claude Code.
        # Model selection stays here rather than moving into the transport, so
        # `--deep` keeps meaning "swap to Opus" identically on both providers.
        # Raises ValueError on a missing precondition (no API key, no `claude`
        # on PATH) or an unknown provider, which is the exception the service
        # already catches to continue compute-only.
        self.transport = build_transport(config)

        self._usage_log: list[dict] = []

    def use_deep_models(self) -> None:
        """Override every call to use Opus for deeper analysis.

        `max_tokens` is deliberately left alone. It used to be set to 4000
        here, which was *below* the configured 4096 — deep mode bought a
        better model and then gave it less room to answer in. The configured
        ceiling already accommodates the deeper model's thinking.
        """
        self.pass1_model = self.deep_model
        self.pass2_model = self.deep_model
        self.forward_growth_model = self.deep_model
        logger.info(f"Deep mode: all passes → {self.deep_model}")

    def use_configured_models(self) -> None:
        """Restore the configured models, undoing any deep-mode override.

        `use_deep_models` mutates a long-lived instance, and the service is
        documented as reusable — so without an undo, one `analyze(deep=True)`
        silently made every later `deep=False` call on that instance run on
        Opus too, at several times the cost and with nothing to say the flag
        had been ignored.
        """
        llm_config = self._config.get("llm", {})
        self.pass1_model = llm_config.get("pass1_model", DEFAULT_MODEL)
        self.pass2_model = llm_config.get("pass2_model", DEFAULT_MODEL)
        self.forward_growth_model = forward_growth_model(self._config)
        self.max_tokens = llm_config.get("max_tokens", DEFAULT_MAX_TOKENS)

    # ── Forward-growth extraction (Stage 1.5) ──

    def usage_summary(self) -> dict:
        """Cumulative tokens and cost across this orchestrator's calls.

        Public because the sweep meters a running total against a cost ceiling
        between tickers, and a ceiling read off a private attribute would break
        silently the first time the accounting changed shape.
        """
        return self._summarize_usage()

    def run_forward_growth_extraction(
        self, ticker: str, company_name: str, submission: dict
    ) -> dict:
        """One extraction call over the already-gated sections.

        Returns the raw parsed response — unvalidated on purpose. Validation
        belongs at the boundary in `forward_growth.validate_extraction`, where
        it can be tested against recorded malformed responses without an API in
        the loop, which is where the failure modes actually live.
        """
        if not self.enabled:
            return {}

        prompt = forward_growth.build_prompt(ticker, company_name, submission)
        return self._call_api(self.forward_growth_model, prompt, "forward_growth")

    def run_analysis(
        self,
        ticker: str,
        company_name: str,
        sector: str,
        market_cap: float | None,
        metrics: dict[str, MetricResult],
        scores: dict,
        annual_report_text: str | None = None,
        sector_context: str = "",
        growth_decomposition: dict | None = None,
        eligibility: dict | None = None,
    ) -> dict:
        """Run the full 2-pass LLM analysis.

        `eligibility` is explanatory context for Pass 2 so its thesis reads
        coherently against the 100x verdict. It is not a control: the action
        the report displays is capped in deterministic code afterwards (see
        action_policy), never by trusting the model to comply.

        Returns dict with keys: pass1, pass2, usage.
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
            growth_decomposition=growth_decomposition,
            eligibility=eligibility,
        )

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
            annual_report_text=annual_report_text[: self.pass1_ar_char_budget],
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
        growth_decomposition: dict | None = None,
        eligibility: dict | None = None,
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
            growth_quality_report=build_growth_decomposition_context(growth_decomposition),
            eligibility_context=build_eligibility_context(eligibility),
            # The closed list of checkpoint ids. Asked for an id without a
            # menu, a model invents plausible ones — sending the vocabulary is
            # what makes structured monitorables possible at all.
            checkpoint_vocabulary=vocabulary_prompt_block(),
            # The model cannot know the run date, and asked for one without it
            # it answers from its training cutoff — the first real run dated
            # every monitorable eleven months in the past.
            today=date.today().isoformat(),
        )

        return self._call_api(self.pass2_model, prompt, "pass2")

    # ── Model Call ──

    def _call_api(self, model: str, prompt: str, pass_name: str) -> dict:
        """Ask the configured transport, log usage, parse the JSON response.

        Kept under its original name and signature: the transport is a
        *transport*, and every caller here already knows what this does. What
        changed is only where the bytes go.
        """
        start_time = time.time()

        try:
            response = self.transport.complete(model, prompt, self.max_tokens)

            elapsed = time.time() - start_time

            usage = {
                "pass": pass_name,
                "model": model,
                # Which pool paid. Without this, a usage block from the CLI path
                # and one from the API path are indistinguishable while meaning
                # different things — see `tokens_basis` and the cache counts.
                "provider": self.transport.name,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "tokens_basis": response.tokens_basis,
                # Actual metered cost when the transport knows one; None leaves
                # `_summarize_usage` to price it from MODEL_PRICING.
                "cost_usd": response.cost_usd,
                "elapsed_seconds": round(elapsed, 1),
            }
            # CLI path only, and absent rather than zero on the API path, which
            # has nothing to say about them. The envelope's `input_tokens`
            # excludes cache reads, so these are what stop a 37K-token call
            # reporting `2` and reading as a phantom efficiency.
            if response.cache_read_input_tokens is not None:
                usage["cache_read_input_tokens"] = response.cache_read_input_tokens
            if response.cache_creation_input_tokens is not None:
                usage["cache_creation_input_tokens"] = (
                    response.cache_creation_input_tokens
                )
            self._usage_log.append(usage)

            logger.info(
                f"  {pass_name}: {usage['input_tokens']}in + "
                f"{usage['output_tokens']}out tokens, {elapsed:.1f}s"
            )

            # Parse JSON from response
            return self._parse_json_response(response.text)

        except TransportError as e:
            logger.error(f"Transport error in {pass_name}: {e}")
            return {"error": str(e), "pass": pass_name}
        except Exception as e:
            logger.error(f"Error in {pass_name}: {e}")
            return {"error": str(e), "pass": pass_name}

    def _parse_json_response(self, text: str) -> dict:
        """Extract JSON from LLM response, handling markdown code blocks and truncation."""
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from ```json code block
        json_marker = text.find("```json")
        if json_marker != -1:
            start = json_marker + 7
            end = text.find("```", start)
            snippet = text[start:end].strip() if end != -1 else text[start:].strip()
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                # May be truncated — try brace matching below
                pass

        # Try extracting from generic ``` code block
        if "```" in text:
            start = text.find("```") + 3
            newline = text.find("\n", start)
            if newline != -1:
                start = newline + 1
                end = text.find("```", start)
                snippet = text[start:end].strip() if end != -1 else text[start:].strip()
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError:
                    pass

        # Try finding JSON object in text (handles truncated responses)
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            try:
                return json.loads(text[brace_start : brace_end + 1])
            except json.JSONDecodeError:
                pass

        # Last resort: try to repair truncated JSON by closing open braces/brackets
        if brace_start != -1:
            fragment = text[brace_start:]
            repaired = self._repair_truncated_json(fragment)
            if repaired:
                return repaired

        logger.warning("Could not parse JSON from LLM response")
        return {"raw_response": text, "parse_error": True}

    @staticmethod
    def _repair_truncated_json(fragment: str) -> dict | None:
        """Attempt to repair truncated JSON by closing open structures."""
        # Strip trailing incomplete string/value
        # Walk backward to find a clean cut point after a complete value
        cleaned = fragment.rstrip()

        # Remove trailing comma if present
        if cleaned.endswith(","):
            cleaned = cleaned[:-1]

        # Count open/close braces and brackets
        open_braces = cleaned.count("{") - cleaned.count("}")
        open_brackets = cleaned.count("[") - cleaned.count("]")

        # Check if we're inside a string (odd number of unescaped quotes)
        in_string = False
        for i, ch in enumerate(cleaned):
            if ch == '"' and (i == 0 or cleaned[i - 1] != "\\"):
                in_string = not in_string

        if in_string:
            cleaned += '"'

        # Close structures
        cleaned += "]" * max(0, open_brackets)
        cleaned += "}" * max(0, open_braces)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None

    def _load_template(self, filename: str) -> str:
        """Load a prompt template file."""
        path = PROMPTS_DIR / filename
        with open(path) as f:
            return f.read()

    def _summarize_usage(self) -> dict:
        """Total tokens and cost, saying plainly which kind of cost it is.

        An entry that carries an actual metered cost is used as-is; one that
        does not is priced from `MODEL_PRICING`. `cost_basis` states which
        happened — the same estimate-versus-recorded honesty `friction.basis`
        uses, and necessary here because `estimated_cost_usd` now sometimes
        holds actuals.

        The key **keeps its name** despite that: the extraction sweep's
        `--ceiling` meters on it, and renaming it would break the one contract
        that stops a sweep running unbounded. Note the ceiling therefore means
        something different per provider — on the CLI path it bounds real
        dollars including per-call harness overhead, so a ceiling calibrated
        against API pricing trips far sooner.
        """
        total_input = sum(u["input_tokens"] for u in self._usage_log)
        total_output = sum(u["output_tokens"] for u in self._usage_log)
        total_time = sum(u["elapsed_seconds"] for u in self._usage_log)

        actuals = [u for u in self._usage_log if u.get("cost_usd") is not None]
        cost = sum(
            u["cost_usd"]
            if u.get("cost_usd") is not None
            else estimate_cost(u["model"], u["input_tokens"], u["output_tokens"])
            for u in self._usage_log
        )

        if actuals and len(actuals) == len(self._usage_log):
            cost_basis = "actual"
        elif actuals:
            cost_basis = "mixed"
        else:
            cost_basis = "estimated"

        return {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "total_seconds": round(total_time, 1),
            "estimated_cost_usd": round(cost, 4),
            "cost_basis": cost_basis,
            "provider": self.transport.name,
            "passes": self._usage_log,
        }
