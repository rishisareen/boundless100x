"""Forward-growth sub-metrics: what management said, and what the quarters say.

The engine measures backward and at a point in time. Growth carries 25% of the
composite and is entirely CAGRs and streaks, so a company whose growth is
decelerating *right now* scores identically to one accelerating. These
metrics are the forward half — two read the extraction pass's output, one is
fully offline from the quarterly series. (`capex_pipeline` was a third reader
until 2026-08-07, retired against a stated threshold after extracting zero
entries across 54 report-years and 15 swept tickers.)

All three carry **zero weight in an element deliberately absent from
`element_weights`** (KTD1). §12 asks for forward signal; §13 forbids changing
SQGLP scoring; those reconcile only at zero weight. The unweighted element is
belt and braces on top: zero weight makes them non-scoring, and an element the
scorer never iterates makes them structurally incapable of it.

**Nothing here reaches the LLM layer** (KTD2). The extraction output arrives as
`data["forward_growth"]`, exactly as `annual_report_sections` already does. The
backtest re-runs every registered metric inside a per-ticker loop; a metric
that could call an API would issue one call per ticker per backtest, and would
read *today's* annual-report text against *truncated* financials — the precise
look-ahead leak the backtest exists to prevent.

**No sub-metric delegates its indeterminate to the engine.** The engine's input
check only treats a value as missing when it has a truthy `.empty` attribute,
i.e. a DataFrame; a present-but-empty dict passes straight through. So each
metric checks its own year-keyed entry and errors when it is absent, empty, or
sourced from a section whose provenance was not `found`.
"""

import logging
import math
import re

import numpy as np
import pandas as pd

from boundless100x import forward_growth_schema as schema
from boundless100x.compute_engine.metrics.base import MetricResult
from boundless100x.compute_engine.metrics.builtin._helpers import quarter_index
from boundless100x.compute_engine.metrics.builtin.profitability import _get_annual_rows

logger = logging.getLogger(__name__)

# Periods back for a year-over-year comparison, matching
# `checkpoints._QUARTERS_PER_YEAR`. The same rule for the same reason: comparing
# against the previous quarter reads seasonality as a trend.
_QUARTERS_PER_YEAR = 4

# Two YoY figures are the minimum for a second difference, and each needs four
# quarters of run-up. Screener renders ~11-13, so both this and the eight-period
# preference below are reachable.
_MIN_QUARTERS = _QUARTERS_PER_YEAR + 2


def _year_of(period) -> int | None:
    """The four-digit year named by a period label, or None.

    Handles `FY2026`, `FY26`, `2026`, `Mar 2026`. A label naming no year at all
    ("the medium term") returns None and its promise is discarded rather than
    guessed — an unresolvable period is not a checkable promise.
    """
    digits = re.findall(r"\d{2,4}", str(period or ""))
    if not digits:
        return None
    token = digits[-1]
    if len(token) == 4:
        return int(token)
    if len(token) == 2:
        return 2000 + int(token)
    return None


def _usable_sections(payload: dict, required: tuple) -> set:
    """Which of a sub-metric's required sections were usable for this year.

    Provenance is tagged per *section* while a report year usually carries a
    mix (KTD4), so a year is not usable or unusable as a whole — only its
    individual sections are. `suspect` counts exactly as `fallback` (KTD9):
    both mean the slot does not hold the section it names.
    """
    sections = (payload or {}).get("sections") or {}
    return {name for name in required if sections.get(name) == schema.FOUND}


def _entries_by_year(data: dict, metric_id: str, kind: str) -> tuple[dict, str]:
    """Entries of one kind, per report year, from usable sections only.

    Returns `({year: [entry, ...]}, "")` or `({}, reason)`. The reason
    distinguishes the three ways this can come up empty — never read, read but
    unusable, read and empty — because "indeterminate" without a reason is the
    outcome this whole layer exists to avoid.
    """
    return _entries_and_unusable(data, metric_id, kind)[:2]


def _entries_and_unusable(
    data: dict, metric_id: str, kind: str
) -> tuple[dict, str, list]:
    """As `_entries_by_year`, plus the report years that could not be read.

    A caller that answers from the *newest* year needs to know when a newer one
    exists and was unreadable — otherwise it silently answers from a superseded
    filing (see `compute_tam_runway`).
    """
    if "forward_growth" not in data:
        return {}, (
            "no forward-growth extraction available — run an analysis with the "
            "LLM enabled at least once to populate it"
        ), []

    by_year = data.get("forward_growth") or {}
    if not isinstance(by_year, dict) or not by_year:
        return {}, (
            "forward-growth extraction is empty — no annual-report years read"
        ), []

    required = schema.REQUIRED_SECTIONS[metric_id]

    found: dict[str, list] = {}
    unusable: list[str] = []
    for year, payload in by_year.items():
        usable = _usable_sections(payload, required)
        if not usable:
            sections = (payload or {}).get("sections") or {}
            rendered = ", ".join(
                "{}: {}".format(name, sections.get(name, "absent")) for name in required
            )
            unusable.append(f"{year} ({rendered})")
            continue
        entries = [
            entry
            for entry in ((payload or {}).get(kind) or [])
            if isinstance(entry, dict) and entry.get("section") in usable
        ]
        if entries:
            found[year] = entries

    if found:
        return found, "", unusable

    if unusable:
        return {}, (
            f"no usable {'/'.join(required)} section for {metric_id}: "
            f"{'; '.join(sorted(unusable))}"
        ), unusable
    return {}, (
        f"the {'/'.join(required)} section was read but carried no {kind} statements"
    ), unusable


def _set_aside_reason(subject: str, units: list) -> str:
    """Why a figure this system holds cannot be compared with rupee figures.

    One sentence for all three sub-metrics. They differ only in the noun, and
    three hand-written variants had already drifted in wording while saying
    the same thing.
    """
    return (
        f"{subject} is stated in {'/'.join(units)} rather than INR crore — this "
        f"pipeline holds no exchange rate, so the figure is stored but cannot "
        f"be compared with rupee figures"
    )


# ── promises_kept_ratio ────────────────────────────────────────────────────


def _settling_frames(data: dict) -> dict:
    """The annual frames a promise can be settled against, keyed by name."""
    frames = {}
    for name in ("financials", "cashflow", "ratios"):
        frame = data.get(name)
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            frames[name] = _get_annual_rows(frame, len(frame))
    return frames


def _column_value(frame, labels, column: str, year: int) -> float | None:
    """The numeric value of a column in the row whose label names `year`."""
    match = frame[labels.str.contains(rf"\b{year}\b", regex=True, na=False)]
    if match.empty:
        return None
    value = pd.to_numeric(match[column], errors="coerce").dropna()
    return float(value.iloc[-1]) if not value.empty else None


def _delivered(frames: dict, metric: str, year: int) -> tuple[float | None, str]:
    """What the accounts actually show for a guided quantity in a fiscal year.

    Returns `(value, "frame.column")`, or `(None, "")` when no settling row
    exists. Screener labels an annual column by its period end (`Mar 2026`), so
    the row for FY N is the one whose label names year N.

    **A growth-rate promise is settled from two rows, and both must be named.**
    "Revenue growth of 12% in FY2026" is delivered by the change from FY2025 to
    FY2026, so the prior year is looked up by its own label rather than taken as
    whichever row happens to sit above. A company whose FY2025 column is absent
    from the frame is unsettleable — settling it against FY2024 instead would
    silently compare a two-year change against a one-year promise and read a
    kept promise as spectacularly beaten.
    """
    spec = schema.GUIDANCE_METRICS.get(metric)
    if spec is None:
        return None, ""

    frame = frames.get(spec["frame"])
    if frame is None or spec["column"] not in frame.columns:
        return None, ""

    if "year" not in frame.columns:
        return None, ""
    labels = frame["year"].astype(str)

    current = _column_value(frame, labels, spec["column"], year)
    if current is None:
        return None, ""

    if spec.get("growth"):
        prior = _column_value(frame, labels, spec["column"], year - 1)
        if prior is None or prior <= 0:
            # A percent change off a loss is not the growth anyone guided: with
            # `abs(prior)`, a PAT going -100 -> -50 reads as +50% and would
            # settle a 20%-growth promise as kept. Unsettleable is the honest
            # answer for a non-positive base.
            return None, ""
        return (
            (current - prior) / abs(prior) * 100.0,
            f"{spec['frame']}.{spec['column']} YoY",
        )

    delivered = abs(current) if spec.get("absolute") else current
    return delivered, f"{spec['frame']}.{spec['column']}"


def compute_promises_kept(data: dict, params: dict) -> MetricResult:
    """The share of management's own due targets that the accounts later met.

    Settled semantics, because each of these changes what the number means:

    * **A promise carries a number.** Guidance must name a quantity from the
      closed set and a target value a later financials row can settle.
      Directional prose ("we expect strong growth") is recorded by the
      extractor but is not a promise and never enters the denominator —
      counting unfalsifiable statements would let vague management score
      perfectly.
    * **Kept means delivered at or above `tolerance` of the guided value**
      (0.95 by default). Indian annual-report guidance is frequently a range or
      a rounded target, so exact matching would read rounding as broken
      credibility. For a range the **lower bound** is the promise.
    * **The denominator is promises that came due.** A promise whose target
      period has not yet arrived is pending and enters neither side — the same
      due-versus-not distinction Phase 1's checkpoints already draw, and for
      the same reason: zero kept out of zero due is silence, not a record.
    * **A period that cannot be resolved to a column is discarded**, not
      guessed at.
    * **Only the company's own growth is a promise** (KTD8). Market, industry
      and economy forecasts outnumber company-subject statements about four to
      one in the same MD&A sections, and a percentage cannot be told apart by
      type, grounding, or unit — so the entry's declared `subject` decides.
      Counting market forecasts would make this a measure of macroeconomic
      luck rather than management credibility, which is worse than the blank
      it would replace.

    Requires guidance from at least two report years (A5). One year of targets
    is a snapshot; credibility is a pattern, and a company that hit its single
    recorded target would otherwise score a perfect 100.
    """
    tolerance = float(params.get("tolerance", 0.95))
    min_years = int(params.get("min_report_years", 2))

    by_year, reason = _entries_by_year(data, "promises_kept_ratio", schema.GUIDANCE)
    if not by_year:
        return MetricResult(error=reason)

    frames = _settling_frames(data)
    settled: list[dict] = []
    pending = discarded = not_a_promise = 0
    wrong_unit: set[str] = set()
    checkable_years: set = set()

    for report_year in sorted(by_year):
        company_said = []
        for entry in by_year[report_year]:
            if entry.get("subject") != schema.SUBJECT_COMPANY:
                # Stored, auditable, and not management's promise to keep.
                not_a_promise += 1
            else:
                company_said.append(entry)

        # A promise this system cannot check is not a promise it may count as
        # missed — an uncheckable target says nothing about management's
        # credibility either way.
        checkable, set_aside = schema.partition_by_unit(schema.GUIDANCE, company_said)
        wrong_unit.update(set_aside)
        if checkable:
            checkable_years.add(report_year)

        for entry in checkable:
            target = entry.get("target_value")
            if not schema.is_number(target):
                discarded += 1
                continue

            target_year = _year_of(entry.get("target_period"))
            if target_year is None:
                discarded += 1
                continue

            delivered, column = _delivered(frames, entry.get("metric"), target_year)
            if delivered is None:
                pending += 1
                continue

            kept = target > 0 and delivered >= target * tolerance
            settled.append({
                "report_year": report_year,
                "metric": entry.get("metric"),
                "target_period": entry.get("target_period"),
                "guided": float(target),
                "delivered": delivered,
                "kept": bool(kept),
                "settled_against": column,
                "source_sentence": entry.get("source_sentence"),
            })

    if not settled:
        units = (
            f", stated in {'/'.join(sorted(wrong_unit))} rather than a unit the "
            f"accounts can settle" if wrong_unit else ""
        )
        return MetricResult(
            error=(
                f"no guided period has come due yet ({pending} pending, "
                f"{discarded} unresolvable, {not_a_promise} about a market "
                f"rather than the company{units}) — zero kept out of zero due "
                f"is silence, not a credibility record"
            )
        )

    # **The credibility gate counts years that carried checkable guidance.**
    # Applied to the raw report years, a year holding nothing but market
    # forecasts — or nothing but figures in a unit the accounts cannot settle —
    # stood in for a year of guidance, so one company promise could satisfy
    # "needs 2" and score a perfect 100. That is exactly the snapshot-versus-
    # record confusion A5 exists to prevent. Pending promises still count here:
    # a target that has not come due yet is guidance the company gave, and
    # whether it settled is the denominator's question, not this one.
    if len(checkable_years) < min_years:
        return MetricResult(
            error=(
                f"checkable guidance from {len(checkable_years)} report year(s), "
                f"needs {min_years} — one year of targets is a snapshot, not a "
                f"credibility record"
            )
        )

    kept = sum(1 for s in settled if s["kept"])
    return MetricResult(
        value=round(kept / len(settled) * 100, 1),
        metadata={
            "kept": kept,
            "due": len(settled),
            "pending": pending,
            "discarded": discarded,
            "not_a_promise": not_a_promise,
            "set_aside_for_unit": sorted(wrong_unit),
            "report_years": sorted(by_year),
            "tolerance": tolerance,
            "settled": settled,
            "unit": "pct",
            "direction": "higher_is_better",
        },
    )


# ── tam_runway ─────────────────────────────────────────────────────────────


def compute_tam_runway(data: dict, params: dict) -> MetricResult:
    """Years of the company's own recent growth before it meets its stated market.

    The arithmetic question §7.2 asks: does the addressable market management
    describes leave room for the growth the thesis assumes, or does the company
    run into its own ceiling first? `ln(TAM / revenue) / ln(1 + g)`, with `g`
    the trailing revenue CAGR.

    A company already at or past its stated market has **zero** runway, which
    is a reading rather than an error — it is exactly the finding that should
    stop a compounding thesis. A non-positive growth rate is indeterminate:
    the arithmetic has no answer, and reporting an enormous runway because
    growth is flat would invert the signal.

    Saturates at `cap_years` rather than reporting a three-figure runway off a
    near-zero growth rate, the same way the reverse DCF pins to its search
    bounds and says so.
    """
    from boundless100x.compute_engine.metrics.builtin.growth import compute_cagr

    cap_years = float(params.get("cap_years", 50.0))

    by_year, reason, unusable = _entries_and_unusable(data, "tam_runway", schema.TAM)
    if not by_year:
        return MetricResult(error=reason)

    # The newest report's largest stated market: management restates and
    # revises TAM, and the current view is the one a thesis rests on.
    newest = max(by_year)
    # **A newer filing that could not be read is disclosed, not hidden.** The
    # metric answers from the newest *usable* report, which is the right
    # fallback — refusing outright would discard a real reading on a corpus
    # where roughly half of all report-years are fallback. What is not
    # acceptable is doing it silently: a market size two years stale reads
    # exactly like a current one, and a thesis rests on the current view.
    superseded = sorted(
        year for year in (u.split(" ")[0] for u in unusable) if year > newest
    )
    usable, wrong_unit = schema.partition_by_unit(schema.TAM, by_year[newest])
    sizes = [
        float(entry["market_size_inr_cr"])
        for entry in usable
        if schema.is_number(entry.get("market_size_inr_cr"))
    ]
    if not sizes:
        if wrong_unit:
            return MetricResult(
                error=_set_aside_reason(
                    "the stated addressable market", wrong_unit
                )
            )
        return MetricResult(error="No numeric addressable-market figure extracted")
    tam = max(sizes)
    if tam <= 0:
        return MetricResult(error="Non-positive addressable market")

    fin = _get_annual_rows(data.get("financials", pd.DataFrame()), 1)
    revenue = pd.to_numeric(fin.get("revenue", pd.Series(dtype=float)), errors="coerce").dropna()
    if revenue.empty or float(revenue.iloc[-1]) <= 0:
        return MetricResult(error="No positive annual revenue to measure runway from")
    latest_revenue = float(revenue.iloc[-1])

    growth = compute_cagr(data, {"field": "revenue", "years": params.get("growth_years", 5)})
    if not growth.ok:
        return MetricResult(error=f"No revenue growth rate for TAM runway: {growth.error}")
    rate = float(growth.value) / 100.0
    if rate <= 0:
        return MetricResult(
            error=(
                "revenue growth is not positive — runway to a market ceiling is "
                "undefined, and a large number here would invert the signal"
            )
        )

    if latest_revenue >= tam:
        years, saturated = 0.0, False
    else:
        years = math.log(tam / latest_revenue) / math.log(1 + rate)
        saturated = years > cap_years
        years = min(years, cap_years)

    return MetricResult(
        value=round(years, 1),
        flags=["tam_from_superseded_report"] if superseded else [],
        metadata={
            "superseded_by_unreadable_years": superseded,
            "tam_inr_cr": tam,
            "latest_revenue_inr_cr": latest_revenue,
            "revenue_cagr_pct": float(growth.value),
            "report_year": newest,
            "saturated": saturated,
            "cap_years": cap_years,
            "set_aside_for_unit": wrong_unit,
            "sources": by_year[newest],
            "unit": "years",
            "direction": "higher_is_better",
        },
    )


# ── quarterly_momentum ─────────────────────────────────────────────────────


def compute_quarterly_momentum(data: dict, params: dict) -> MetricResult:
    """Whether growth is speeding up or slowing down, right now.

    **A second difference, not a growth rate.** Each year-over-year figure
    compares a quarter against the same quarter four periods back, never the
    previous one, so seasonality does not read as a trend (Phase 1's checkpoint
    rule). Momentum is then the change *between consecutive YoY figures* —
    YoY(t) minus YoY(t-1), averaged over the recent window.

    That distinction is the whole metric. A single YoY number is a growth
    *level*: it answers "is this growing", not "is growth speeding up", and a
    company compounding a steady 20% would report +20 momentum while
    accelerating not at all.

    Two YoY figures therefore need at least six quarters; eight gives a reading
    not dominated by one quarter's noise. Screener renders ~11-13, so both are
    reachable.
    """
    field = params.get("field", "revenue")
    window = int(params.get("momentum_quarters", 4))

    frame = data.get("quarterly")
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return MetricResult(error="No quarterly results — refetch required")
    if field not in frame.columns:
        return MetricResult(error=f"Column '{field}' absent from quarterly results")

    if "quarter" not in frame.columns:
        return MetricResult(
            error=(
                "quarterly results carry no period labels, so a year-over-year "
                "comparison cannot be verified as spanning four quarters"
            )
        )

    # **Every comparison is matched by period label, never by position.**
    # Taking the value four *rows* back assumes the rows are contiguous, and
    # `dropna()` compresses the timeline before that offset is applied — so a
    # single missing interior quarter silently pairs a quarter against one five
    # or six periods earlier. On an otherwise flat 20% YoY series one such gap
    # produced -1.4pp of fabricated deceleration, and a slightly larger one
    # crosses the flag threshold and emits `quarterly_growth_decelerating` for
    # a company whose growth never moved.
    numeric = pd.to_numeric(frame[field], errors="coerce")
    by_period: dict[int, float] = {}
    for label, value in zip(frame["quarter"], numeric):
        index = quarter_index(label)
        if index is not None and pd.notna(value):
            by_period[index] = float(value)

    if len(by_period) < _MIN_QUARTERS:
        return MetricResult(
            error=(
                f"{len(by_period)} readable quarters of {field}, needs "
                f"{_MIN_QUARTERS} — two year-over-year figures are required to "
                f"form a second difference"
            )
        )

    # A YoY figure exists only where the same quarter one year earlier does.
    yoy: dict[int, float] = {}
    for index, value in by_period.items():
        base = by_period.get(index - _QUARTERS_PER_YEAR)
        if base is None or base == 0:
            continue
        yoy[index] = (value - base) / abs(base) * 100

    # A second difference exists only between *adjacent* quarters. Comparing
    # across a gap would reintroduce the same fabrication one level up.
    diffs = [
        (index, yoy[index] - yoy[index - 1])
        for index in sorted(yoy)
        if index - 1 in yoy
    ]
    if len(yoy) < 2 or not diffs:
        return MetricResult(
            error=(
                f"{len(yoy)} year-over-year figure(s) and {len(diffs)} adjacent "
                f"pair(s) — a second difference needs two consecutive quarters "
                f"that each have the same quarter a year earlier"
            )
        )

    recent = [d for _, d in diffs[-window:]]
    momentum = float(np.mean(recent))

    flags = []
    threshold = float(params.get("flag_threshold_pp", 2.0))
    if momentum >= threshold:
        flags.append("quarterly_growth_accelerating")
    elif momentum <= -threshold:
        flags.append("quarterly_growth_decelerating")

    return MetricResult(
        value=round(momentum, 2),
        flags=flags,
        metadata={
            "field": field,
            "quarters_used": len(by_period),
            "yoy_pct": [round(yoy[i], 2) for i in sorted(yoy)],
            "second_differences_pp": [round(v, 2) for v in recent],
            "latest_period": (
                str(frame["quarter"].iloc[-1]) if "quarter" in frame.columns else None
            ),
            "unit": "pp_change_in_yoy",
            "direction": "higher_is_better",
        },
    )
