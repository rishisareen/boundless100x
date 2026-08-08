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

from boundless100x.data_fetcher import refetch
from boundless100x.data_fetcher.download_annual_reports import load_cached_sections
from boundless100x.llm_layer import forward_growth
from boundless100x.llm_layer.orchestrator import MODEL_PRICING, estimate_cost
from boundless100x.llm_layer.transport import CHARS_PER_TOKEN, COST_BASIS_ESTIMATED

logger = logging.getLogger(__name__)

# `CHARS_PER_TOKEN` lived here and now lives in `transport.py`, imported above.
# It was measured here first — 4 was the Sonnet 4.6 figure, the Claude 5 family
# tokenizes ~30% higher, so 4 under-priced every estimate by about a third — and
# then the CLI transport's own fallback estimate was written at 4 anyway, which
# is exactly what a second copy of a measured constant does. One definition, in
# the module both readers can import from without a cycle.

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
    (re.compile(r"source_sentence is not prose"),
     "quotation is a flattened chart or table, not prose"),
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

    code = metadata.get("bse_code")
    return {
        "metadata": metadata,
        "annual_report_sections": (
            load_cached_sections(raw_data_dir, code) if code else {}
        ),
    }


def plan_ticker(service, ticker: str) -> dict:
    """What extracting this ticker would submit and cost — with no API call."""
    context = load_context(service.suite.raw_data_dir, ticker)
    if context is None:
        return {"ticker": ticker, "skipped": "no metadata.json in the corpus"}

    sections = context["annual_report_sections"]
    if not sections:
        return {"ticker": ticker, "skipped": "no annual-report sections on disk"}

    # The same planner Stage 1.5 uses, so the estimate prices what is sent.
    plan = forward_growth.plan_submission(
        service.config, sections, llm=service._llm
    )
    gated, reasons, submission = plan["gated"], plan["reasons"], plan["submission"]

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

    model = plan["model"]
    company = context["metadata"].get("name", ticker)
    prompt = forward_growth.build_prompt(ticker, company, submission)
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


def corpus_plans(service) -> list[dict]:
    """A plan for every cached ticker — the extractable ones and the skipped.

    Returns the plans rather than a filtered ticker list, because the skip
    *reasons* are the expensive half and were previously computed and thrown
    away: under `--all` the report named 15 tickers to sweep and 0 skipped,
    while 7 had been excluded moments earlier with reasons in hand. A sweep
    that silently drops what it did not look at is the one thing this command
    must not do.

    Ticker enumeration is `refetch.enumerate_tickers` rather than a second
    copy of the rule — one definition of "which directory is a real ticker".
    """
    tickers, _ = refetch.enumerate_tickers(service.suite.raw_data_dir)
    return [plan_ticker(service, ticker) for ticker in tickers]


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

    plans = (
        [plan_ticker(service, t.upper()) for t in tickers] if tickers
        else corpus_plans(service)
    )
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
        "actual": {
            "usd": 0.0, "input_tokens": 0, "output_tokens": 0,
            "provider": None, "cost_basis": COST_BASIS_ESTIMATED,
        },
    }

    if dry_run:
        return report

    if service._llm is None:
        raise RuntimeError(
            "no LLM is configured — set ANTHROPIC_API_KEY, or use dry_run to "
            "price the sweep without one"
        )

    # `estimate_cost` returns 0.0 for a model id it does not recognise, which is
    # right for *reporting* — a made-up price would read as a real one. But the
    # ceiling meters on that same number, so an unrecognised model would print a
    # ceiling and enforce nothing. Say so rather than run unbounded in silence.
    model = runnable[0]["model"] if runnable else None
    if cost_ceiling_usd is not None and model and not any(
        family in model for family in MODEL_PRICING
    ):
        logger.warning(
            f"No price is known for {model!r}, so every call meters as $0 and "
            f"the ${cost_ceiling_usd} ceiling cannot bind. Add it to "
            f"MODEL_PRICING, or watch the token counts instead."
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
        # Which pool spent it, and whether `usd` is metered or priced from the
        # table. The ceiling meters on the same number under both providers, so
        # what it *means* has to travel with it: on the claude_cli path it is a
        # real bill including a fixed per-call harness overhead, which makes a
        # ceiling calibrated against API pricing trip far sooner.
        "provider": after.get("provider"),
        "cost_basis": after.get("cost_basis", COST_BASIS_ESTIMATED),
    }
    # The cache counts have to travel this far or they reach no surface at all.
    # `input_tokens` above is the envelope's, which **excludes** everything
    # served from or written to the cache — so a sweep that moved half a million
    # cached tokens prints a four-figure input total beside a real dollar figure,
    # and reads as impossibly efficient next to the API path's honest count.
    # Absent rather than zero when the provider had nothing to say about caching:
    # the API path does not cache-report, and a 0 there would claim every prompt
    # was written fresh.
    for key in (
        "total_cache_read_input_tokens",
        "total_cache_creation_input_tokens",
        "total_cached_input_tokens",
    ):
        if key in after:
            report["actual"][key[len("total_") :]] = after[key] - spent_before.get(
                key, 0
            )
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
                "kept": 0, "discarded": [],
                # A failure is not a refund. On the claude_cli path the harness
                # prefix is billed before the model reads a word of ours, so a
                # ticker that failed can still have spent — reported here for
                # the same reason the successful branch reports it, and read off
                # the same running total the `--ceiling` meters on.
                "cost_usd": round(
                    service._llm.usage_summary()["estimated_cost_usd"] - before, 4
                )}

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
