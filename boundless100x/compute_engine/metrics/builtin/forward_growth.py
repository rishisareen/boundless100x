"""Forward-growth sub-metrics: what management said, and what the quarters say.

The engine measures backward and at a point in time. Growth carries 25% of the
composite and is entirely CAGRs and streaks, so a company whose growth is
decelerating *right now* scores identically to one accelerating. These four
metrics are the forward half — three read the extraction pass's output, one is
fully offline from the quarterly series.

All four carry **zero weight in an element deliberately absent from
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
    if "forward_growth" not in data:
        return {}, (
            "no forward-growth extraction available — run an analysis with the "
            "LLM enabled at least once to populate it"
        )

    by_year = data.get("forward_growth") or {}
    if not isinstance(by_year, dict) or not by_year:
        return {}, "forward-growth extraction is empty — no annual-report years read"

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
        return found, ""

    if unusable:
        return {}, (
            f"no usable {'/'.join(required)} section for {metric_id}: "
            f"{'; '.join(sorted(unusable))}"
        )
    return {}, (
        f"the {'/'.join(required)} section was read but carried no {kind} statements"
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
        if prior is None or prior == 0:
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

    if len(by_year) < min_years:
        return MetricResult(
            error=(
                f"guidance from {len(by_year)} report year(s), needs {min_years} — "
                f"one year of targets is a snapshot, not a credibility record"
            )
        )

    frames = _settling_frames(data)
    settled: list[dict] = []
    pending = discarded = not_a_promise = 0
    wrong_unit: list[str] = []

    for report_year in sorted(by_year):
        for entry in by_year[report_year]:
            if entry.get("subject") != schema.SUBJECT_COMPANY:
                # Stored, auditable, and not management's promise to keep.
                not_a_promise += 1
                continue

            if not schema.is_settleable(schema.GUIDANCE, entry, entry.get("metric")):
                # A promise this system cannot check is not a promise it may
                # count as missed — an uncheckable target says nothing about
                # management's credibility either way.
                wrong_unit.append(str(entry.get("unit")))
                continue

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
            f", {len(wrong_unit)} stated in {'/'.join(sorted(set(wrong_unit)))} "
            f"rather than a unit the accounts can settle" if wrong_unit else ""
        )
        return MetricResult(
            error=(
                f"no guided period has come due yet ({pending} pending, "
                f"{discarded} unresolvable, {not_a_promise} about a market "
                f"rather than the company{units}) — zero kept out of zero due "
                f"is silence, not a credibility record"
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
            "set_aside_for_unit": sorted(set(wrong_unit)),
            "report_years": sorted(by_year),
            "tolerance": tolerance,
            "settled": settled,
            "unit": "pct",
            "direction": "higher_is_better",
        },
    )


# ── capex_pipeline ─────────────────────────────────────────────────────────


def compute_capex_pipeline(data: dict, params: dict) -> MetricResult:
    """Announced, not-yet-commissioned capacity as a share of a year's revenue.

    Expressed against revenue rather than in rupees so it reads the same for a
    small company and a large one — "a pipeline worth 40% of a year's sales"
    is a runway statement, "Rs 500 crore" is not.

    Only projects commissioning *after* the latest reported year count: capacity
    already built is in the financials, not in front of them.

    Deduplicated across report years by (amount, commissioning year). A live
    project is restated in every report until it lands, and summing the
    restatements would make a company look like it was building twice over.
    """
    by_year, reason = _entries_by_year(data, "capex_pipeline", schema.CAPEX)
    if not by_year:
        return MetricResult(error=reason)

    fin = _get_annual_rows(data.get("financials", pd.DataFrame()), 1)
    if fin.empty or "revenue" not in fin.columns:
        return MetricResult(error="No annual revenue to size the capex pipeline against")

    revenue = pd.to_numeric(fin["revenue"], errors="coerce").dropna()
    if revenue.empty or float(revenue.iloc[-1]) <= 0:
        return MetricResult(error="Non-positive revenue — pipeline share is undefined")
    latest_revenue = float(revenue.iloc[-1])

    latest_year = _year_of(fin["year"].iloc[-1]) if "year" in fin.columns else None
    if latest_year is None:
        return MetricResult(error="Could not read the latest fiscal year label")

    projects: dict[tuple, dict] = {}
    wrong_unit: list[str] = []
    for report_year in sorted(by_year):
        for entry in by_year[report_year]:
            if not schema.is_settleable(schema.CAPEX, entry):
                # `amount_inr_cr` asserts a unit in its own name that `unit`
                # makes variable. Without this check a USD-stated commitment
                # would be summed straight into a rupee total and silently
                # corrupt the pipeline percentage — the easiest of the three
                # to miss, because nothing about the field name suggests a
                # check is needed.
                wrong_unit.append(str(entry.get("unit")))
                continue
            amount = entry.get("amount_inr_cr")
            if not schema.is_number(amount):
                continue
            commissioning = _year_of(entry.get("commissioning_year"))
            if commissioning is None or commissioning <= latest_year:
                continue
            # Later restatements win, so the metadata quotes the most recent
            # sentence describing a project rather than the oldest.
            projects[(round(float(amount), 2), commissioning)] = {
                "amount_inr_cr": float(amount),
                "commissioning_year": commissioning,
                "report_year": report_year,
                "description": entry.get("description"),
                "source_sentence": entry.get("source_sentence"),
            }

    if not projects:
        if wrong_unit:
            return MetricResult(
                error=(
                    f"every announced capex figure is stated in "
                    f"{'/'.join(sorted(set(wrong_unit)))} rather than INR crore — "
                    f"this pipeline holds no exchange rate, so the amounts are "
                    f"stored but cannot be sized against rupee revenue"
                )
            )
        return MetricResult(
            error=(
                f"no announced capacity commissioning after FY{latest_year} — "
                f"anything earlier is already in the accounts"
            )
        )

    announced = sum(p["amount_inr_cr"] for p in projects.values())
    ordered = sorted(projects.values(), key=lambda p: p["commissioning_year"])

    return MetricResult(
        value=round(announced / latest_revenue * 100, 1),
        metadata={
            "announced_inr_cr": round(announced, 2),
            "latest_revenue_inr_cr": latest_revenue,
            "from_fiscal_year": latest_year,
            "through_fiscal_year": ordered[-1]["commissioning_year"],
            "projects": ordered,
            "set_aside_for_unit": sorted(set(wrong_unit)),
            "unit": "pct_of_revenue",
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

    by_year, reason = _entries_by_year(data, "tam_runway", schema.TAM)
    if not by_year:
        return MetricResult(error=reason)

    # The newest report's largest stated market: management restates and
    # revises TAM, and the current view is the one a thesis rests on.
    newest = max(by_year)
    wrong_unit = sorted({
        str(entry.get("unit")) for entry in by_year[newest]
        if not schema.is_settleable(schema.TAM, entry)
    })
    sizes = [
        float(entry["market_size_inr_cr"])
        for entry in by_year[newest]
        if schema.is_settleable(schema.TAM, entry)
        and schema.is_number(entry.get("market_size_inr_cr"))
    ]
    if not sizes:
        if wrong_unit:
            return MetricResult(
                error=(
                    f"the stated addressable market is given in "
                    f"{'/'.join(wrong_unit)} rather than INR crore — this "
                    f"pipeline holds no exchange rate, so the figure is stored "
                    f"but cannot be compared with rupee revenue"
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
        metadata={
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

    values = pd.to_numeric(frame[field], errors="coerce").dropna().tolist()
    if len(values) < _MIN_QUARTERS:
        return MetricResult(
            error=(
                f"{len(values)} quarters of {field}, needs {_MIN_QUARTERS} — two "
                f"year-over-year figures are required to form a second difference"
            )
        )

    yoy: list[float] = []
    for index in range(_QUARTERS_PER_YEAR, len(values)):
        base = values[index - _QUARTERS_PER_YEAR]
        if base == 0:
            # No YoY basis; skipping keeps the *sequence* of comparable figures
            # intact rather than injecting a zero that would read as a stall.
            continue
        yoy.append((values[index] - base) / abs(base) * 100)

    if len(yoy) < 2:
        return MetricResult(
            error=(
                f"only {len(yoy)} usable year-over-year figure(s) — a second "
                f"difference needs two"
            )
        )

    diffs = [yoy[i] - yoy[i - 1] for i in range(1, len(yoy))]
    recent = diffs[-window:]
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
            "quarters_used": len(values),
            "yoy_pct": [round(v, 2) for v in yoy],
            "second_differences_pp": [round(v, 2) for v in recent],
            "latest_period": (
                str(frame["quarter"].iloc[-1]) if "quarter" in frame.columns else None
            ),
            "unit": "pp_change_in_yoy",
            "direction": "higher_is_better",
        },
    )
