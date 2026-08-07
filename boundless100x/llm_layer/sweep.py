"""Run forward-growth extraction across chosen tickers, priced before it runs.

Every live extraction to date hit one ticker. Ten more have a usable MD&A
section and have never been tried, and 2 of those first 4 calls surfaced
pipeline defects rather than schema gaps — a truncating char budget and a
grounding check broken by PDF line-wrapping — while every offline test passed.
A new failure mode per new filing layout is the expected case, not the
surprising one, which is why this offers a dry run and a pilot batch before it
offers the corpus.

**It never defaults to everything** (KTD6). A refetch invalidates existing
extraction sidecars, because each year's `source_digest` covers the exact text
submitted and adding annual-report years changes it — so a re-spend happens on
the next LLM run whether or not a sweep is asked for. A command that swept the
whole corpus on a mistyped flag would turn that into a real bill. The ticker
list is explicit or the all-tickers flag is explicit; there is no third way.

**The live path goes through `service._forward_growth_stage`**, so gating,
validation, grounding and sidecar versioning stay in one place. A sweep with
its own copy of that sequence would drift from the one production uses, and
the sweep is precisely the tool used to judge whether the production path
works.

The dry run is the part worth testing hardest: it is what stands between a
mistyped flag and a corpus-wide spend. It prices the *exact* prompt the live
call would send, assembled by the same method, rather than a reconstruction of
it — an estimate that missed the template and vocabulary overhead would be
wrong by the one component every ticker pays.
"""

import json
import logging
import re
from pathlib import Path

from boundless100x.llm_layer import forward_growth
from boundless100x.llm_layer.orchestrator import (
    estimate_cost,
    forward_growth_char_budget,
    forward_growth_model,
)

logger = logging.getLogger(__name__)

# Rule-of-thumb characters per token for English prose. Deliberately a round
# number and deliberately named: the estimate is an estimate, and pretending
# otherwise by tuning this to three decimal places would only make it look
# like a quote.
CHARS_PER_TOKEN = 4

# What one extraction response tends to cost in output tokens. Measured, not
# guessed: two pilot batches over the same three tickers returned 4,212 and
# 4,066 output tokens across three calls each, so ~1,350 apiece. An earlier
# guess of 900 put the estimate ~20% under the bill every time — close enough
# to look right and wrong in the direction that matters.
# `estimated_cost_usd_max` carries the worst case (`max_tokens`) beside it,
# because a point estimate with no ceiling invites exactly the surprise this
# module exists to prevent.
ASSUMED_OUTPUT_TOKENS = 1350

# Why an entry was dropped, grouped. A systematic cause — every sentence
# failing grounding, every figure in the wrong unit — is a pipeline defect or
# a schema gap; the same count scattered across twenty distinct reasons reads
# as noise. Grouping is what makes the difference visible.
_DISCARD_BUCKETS = (
    (re.compile(r"source_sentence does not appear"),
     "quoted sentence is not in the submitted text"),
    (re.compile(r"does not appear in its own source_sentence as an? \w+ figure"),
     "figure is not denominated as the entry claims"),
    (re.compile(r"(?:target_period|commissioning_year) does not appear"),
     "period is not in the quoted sentence"),
    (re.compile(r"too far apart to be one claim"),
     "value and period come from different clauses"),
    (re.compile(r"missing required field"), "missing a required field"),
    (re.compile(r"unit .* is outside the closed set"), "unit outside the vocabulary"),
    (re.compile(r"subject .* is outside the closed set"),
     "subject outside the vocabulary"),
    (re.compile(r"metric .* is outside the closed set"),
     "guidance metric outside the vocabulary"),
    (re.compile(r"outside the declared set"), "fields outside the declared set"),
    (re.compile(r"is not a number"), "figure is not a number"),
    (re.compile(r"was never submitted"),
     "attributed to a section or year never sent"),
)


def _bucket(reason: str) -> str:
    for pattern, label in _DISCARD_BUCKETS:
        if pattern.search(reason or ""):
            return label
    return (reason or "unknown")[:60]


def group_discards(discarded: list) -> dict:
    """Discard reasons as `{bucket: count}`, most frequent first."""
    counts: dict[str, int] = {}
    for entry in discarded or []:
        reason = entry.get("reason") if isinstance(entry, dict) else str(entry)
        label = _bucket(reason)
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def load_context(raw_data_dir, ticker: str) -> dict | None:
    """A ticker's metadata and section sidecars, read off the corpus.

    Offline by construction. The sweep's whole subject is annual-report prose
    already on disk, and refetching to reach it would put a network failure
    between the operator and a decision about spending money.
    """
    directory = Path(raw_data_dir) / ticker
    meta_path = directory / "metadata.json"
    if not meta_path.exists():
        return None
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    code = str(metadata.get("bse_code") or "")
    reports = Path(raw_data_dir) / code / "annual_reports" if code else None
    sections: dict[str, dict] = {}
    if reports and reports.is_dir():
        for sidecar in sorted(reports.glob("*_annual_report.sections.json")):
            try:
                sections[sidecar.name.split("_")[0]] = json.loads(
                    sidecar.read_text(encoding="utf-8")
                )
            except (json.JSONDecodeError, OSError):
                logger.warning(f"{ticker}: unreadable sections sidecar {sidecar.name}")

    return {
        "metadata": metadata,
        "annual_report_sections": sections,
        "source_status": {"financials": "ok", "price": "ok"},
    }


def plan_ticker(service, ticker: str) -> dict:
    """What extracting this ticker would submit and cost — with no API call."""
    context = load_context(service.suite.raw_data_dir, ticker)
    if context is None:
        return {"ticker": ticker, "skipped": "no metadata.json in the corpus"}

    sections = context["annual_report_sections"]
    if not sections:
        return {"ticker": ticker, "skipped": "no annual-report sections on disk"}

    gate_config = service.config.get("annual_reports", {}).get("content_gate", {})
    gated, reasons = forward_growth.gate_sections_with_reasons(
        sections,
        scan_chars=gate_config.get("scan_chars", forward_growth.DEFAULT_SCAN_CHARS),
        min_markers=gate_config.get("min_markers"),
        enabled=gate_config.get("enabled", True),
    )
    budget = forward_growth_char_budget(service.config)
    submission = forward_growth.build_submission(sections, gated, char_budget=budget)

    provenance = {
        year: dict(sections_gated) for year, sections_gated in sorted(gated.items())
    }
    if not submission:
        return {
            "ticker": ticker,
            "skipped": (
                "every extractable section is fallback or suspect — nothing may "
                "be submitted, so nothing would be spent"
            ),
            "provenance": provenance,
            "gate_reasons": {y: r for y, r in reasons.items() if r},
        }

    model = forward_growth_model(service.config)
    company = context["metadata"].get("name", ticker)
    prompt = (
        service._llm.build_forward_growth_prompt(ticker, company, submission)
        if service._llm
        else _offline_prompt(ticker, company, submission)
    )
    input_tokens = len(prompt) // CHARS_PER_TOKEN
    max_output = int(service.config.get("llm", {}).get("max_tokens", 4096))

    return {
        "ticker": ticker,
        "skipped": None,
        "provenance": provenance,
        "gate_reasons": {y: r for y, r in reasons.items() if r},
        "years": sorted(submission),
        "sections": sorted({name for year in submission for name in submission[year]}),
        "submission_chars": sum(
            len(text) for year in submission for text in submission[year].values()
        ),
        "prompt_chars": len(prompt),
        "model": model,
        "estimated_input_tokens": input_tokens,
        "estimated_cost_usd": round(
            estimate_cost(model, input_tokens, ASSUMED_OUTPUT_TOKENS), 4
        ),
        "estimated_cost_usd_max": round(
            estimate_cost(model, input_tokens, max_output), 4
        ),
    }


def _offline_prompt(ticker: str, company: str, submission: dict) -> str:
    """The same prompt assembly, when no API key makes an orchestrator.

    A dry run has to work without credentials — pricing a spend is exactly the
    thing one does before deciding to configure and make it.
    """
    template = forward_growth.prompt_template()
    return template.format(
        ticker=ticker,
        company_name=company,
        vocabulary=forward_growth.vocabulary_prompt_block(),
        report_text=forward_growth.render_report_text(submission),
    )


def extractable_tickers(service) -> list[str]:
    """Every corpus ticker with at least one gated-`found` extractable section."""
    root = Path(service.suite.raw_data_dir)
    if not root.is_dir():
        return []
    candidates = sorted(
        child.name for child in root.iterdir()
        if child.is_dir() and not child.name.isdigit()
        and (child / "metadata.json").exists()
    )
    return [t for t in candidates if plan_ticker(service, t).get("skipped") is None]


def sweep(
    service,
    tickers: list[str] | None = None,
    all_tickers: bool = False,
    dry_run: bool = False,
    cost_ceiling_usd: float | None = None,
    limit: int | None = None,
) -> dict:
    """Price, and optionally run, extraction across a chosen set of tickers."""
    if not tickers and not all_tickers:
        raise ValueError(
            "name the tickers to sweep, or pass all_tickers explicitly. There is "
            "no default: a refetch invalidates every extraction sidecar, so a "
            "sweep that guessed at its scope would spend real money on a typo."
        )

    chosen = (
        [t.upper() for t in tickers] if tickers else extractable_tickers(service)
    )
    plans = [plan_ticker(service, ticker) for ticker in chosen]
    runnable = [p for p in plans if p.get("skipped") is None]
    if limit is not None:
        deferred = [p["ticker"] for p in runnable[limit:]]
        runnable = runnable[:limit]
    else:
        deferred = []

    report = {
        "dry_run": dry_run,
        "plans": plans,
        "skipped": [
            {"ticker": p["ticker"], "reason": p["skipped"]}
            for p in plans if p.get("skipped")
        ],
        "deferred": deferred,
        "estimate": {
            "tickers": len(runnable),
            "input_tokens": sum(p["estimated_input_tokens"] for p in runnable),
            "usd": round(sum(p["estimated_cost_usd"] for p in runnable), 4),
            "usd_max": round(sum(p["estimated_cost_usd_max"] for p in runnable), 4),
        },
        "results": [],
        "not_reached": [],
        "actual": {"usd": 0.0, "input_tokens": 0, "output_tokens": 0},
    }

    if dry_run:
        return report

    if service._llm is None:
        raise RuntimeError(
            "no LLM is configured — set ANTHROPIC_API_KEY, or use dry_run to "
            "price the sweep without one"
        )

    spent_before = service._llm.usage_summary()
    for index, plan in enumerate(runnable):
        ticker = plan["ticker"]
        spend = service._llm.usage_summary()["estimated_cost_usd"] - spent_before[
            "estimated_cost_usd"
        ]
        if cost_ceiling_usd is not None and spend >= cost_ceiling_usd:
            report["not_reached"] = [p["ticker"] for p in runnable[index:]]
            logger.warning(
                f"Cost ceiling ${cost_ceiling_usd} reached after ${spend:.4f} — "
                f"{len(report['not_reached'])} ticker(s) not reached"
            )
            break

        report["results"].append(_extract_one(service, ticker))

    after = service._llm.usage_summary()
    report["actual"] = {
        "usd": round(after["estimated_cost_usd"] - spent_before["estimated_cost_usd"], 4),
        "input_tokens": after["total_input_tokens"] - spent_before["total_input_tokens"],
        "output_tokens": (
            after["total_output_tokens"] - spent_before["total_output_tokens"]
        ),
    }
    report["discard_summary"] = group_discards(
        [d for r in report["results"] for d in r.get("discarded", [])]
    )
    return report


def _extract_one(service, ticker: str) -> dict:
    """One ticker through the production stage, isolated from the rest."""
    before = service._llm.usage_summary()["estimated_cost_usd"]
    context = load_context(service.suite.raw_data_dir, ticker)
    try:
        service._forward_growth_stage(ticker, context, use_llm=True)
    except Exception as e:
        # One filing layout the pass has never met must not end the sweep; the
        # whole point of running it is to find out which ones those are.
        logger.error(f"{ticker}: extraction failed: {e}")
        return {"ticker": ticker, "status": "failed", "detail": str(e),
                "kept": 0, "discarded": []}

    years = context.get("forward_growth") or {}
    discarded = context.get("forward_growth_discarded")
    if discarded is None:
        discarded = forward_growth.read_sidecar_discards(
            service._forward_growth_sidecar(ticker, context)
        )
    kept = {
        kind: sum(len(payload.get(kind) or []) for payload in years.values())
        for kind in ("guidance", "capex", "tam")
    }
    return {
        "ticker": ticker,
        "status": "ok",
        "years": sorted(years),
        "kept": sum(kept.values()),
        "kept_by_kind": kept,
        "discarded": discarded,
        "discard_summary": group_discards(discarded),
        "cost_usd": round(
            service._llm.usage_summary()["estimated_cost_usd"] - before, 4
        ),
    }
