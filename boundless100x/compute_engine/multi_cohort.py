"""Multi-cohort walk-forward: many scoring dates per company instead of one.

The single-cutoff diagnostic (`backtest.WalkForwardBacktest`) scores every
company once, at half history, and correlates that one score against one
realized return measured over whatever forward window happened to remain.
That leaves three questions it cannot answer:

  * One observation per company cannot separate the framework's signal from
    the one macro regime the observation happens to sit in — the report's
    own `shared_window` limitation.
  * Windows of different lengths (2 years, 11 years) enter one correlation,
    so the statistic mixes regimes as well as companies.
  * A rank correlation expresses nothing about distribution: whether high
    scorers concentrate the eventual multibaggers, or what a top-K pick
    list would have delivered.

This variant reuses the single-cutoff variant's leakage discipline unchanged
— `point_in_time.truncate_to_date` does the rewinding for both, with the
same per-frame filing lags and the same `rebuild_valuation=False` stance
(KTD0 is a capability for the simulator, not a default for diagnostics) —
and upgrades what is measured:

  * **Many cutoffs** — one per qualifying financial year (`stride_years`
    apart), so observations span several regimes. Consecutive cutoffs share
    most of their forward window; the `overlap` limitation says so rather
    than pretending pooled statistics are independent.
  * **Fixed-horizon labels** — every observation annualises over exactly
    `horizon_years` of forward price, so windows compare like with like. A
    company whose price history ends before the horizon completes
    contributes a *censored* observation, reported and excluded, never a
    quietly shorter window.
  * **Distribution evidence** — quintile buckets by composite, tail lift on
    multibagger outcomes, and precision@K per cohort date, which is the
    statistic that maps to real usage: a watchlist is exactly "the top few
    names by score on one date".

## Withheld evidence leaves the coverage denominator

Scoring at a rewound date cannot use everything production scores:
shareholding- and analyst-derived metrics were never loaded, and market-cap/
P/E metrics are omitted under `rebuild_valuation=False`. Under the plain
coverage definition these count as *wanted but absent* evidence, so every
backtest composite sits far below `low_coverage_threshold` — measured on the
live corpus, the single-cutoff variant's best coverage is 0.72 against the
0.85 bar, and its correlation reports n=0. The diagnostic is not weak; its
denominator is wrong.

The scorer's own doctrine already separates the two cases (`scorer.py`):
absent evidence the composite wanted drags coverage down; evidence that was
*never applicable here* leaves the denominator entirely. Metrics withheld by
backtest policy are the second kind — no analysis at that date could have
used them, so counting them punishes the rewind for being a rewind. This
module therefore determines, empirically and per run, the set of metrics
that failed at *every* observation (nothing in the corpus could score them
at any rewound date) and rescores through a scorer whose applicability
shim marks exactly that set not-applicable. A metric that fails at some
cutoffs but not others stays in the denominator — that is a young company's
genuine missing history, the case coverage exists to catch.

What this is not: calibration evidence. The universe remains the set of
tickers already fetched — hand-picked, survivorship-biased — and the verdict
line says so. Its job is to make the diagnostic sharp enough that a real
point-in-time universe (see `PIT and Backtest Design v01.md`, Part II) lands
on an instrument already trusted.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from boundless100x.compute_engine.backtest import (
    REQUIRED_FILES,
    WalkForwardBacktest,
    # Defined on the parent so BOTH variants score against the same coverage
    # denominator; re-exported here because that is where its tests and the
    # module docstring's discussion of it live.
    _WithheldEvidence,  # noqa: F401
)
from boundless100x.compute_engine.metrics.builtin._helpers import period_end_date
from boundless100x.compute_engine.metrics.builtin.profitability import (
    _get_annual_rows as _scoreable_rows,
)
from boundless100x.compute_engine.point_in_time import _annual_rows, truncate_to_date
from boundless100x.compute_engine.scorer import SQGLPScorer
from boundless100x.compute_engine.sector import applicability_labels

# Financial years knowable at a cutoff before that cutoff may score. The
# single-cutoff variant demands 8 (MIN_TOTAL_YEARS there); per observation
# the bar sits lower because one company contributes several observations
# and the scorer's own min-years rule still governs each metric inside a
# score. Five years is also where SQGLP's trend metrics start meaning
# anything at all.
MIN_HISTORY_YEARS = 5

# Every observation measures exactly this many forward years. Three, not
# five: the corpus's financials end Mar 2025 and its prices end at the last
# fetch, so a five-year horizon needs forward prices out to 2030 and censors
# nearly everything that exists today (measured: 170 of ~216 candidate
# cutoffs). Three-year windows complete for cutoffs up to roughly the last
# fetch minus three years, which keeps most of the corpus scoreable. Raise
# it as the corpus ages — the parameter exists precisely so that is a flag,
# not an edit.
DEFAULT_HORIZON_YEARS = 3

# Years between consecutive cutoffs for the same company. One year keeps
# every observation the corpus can offer; the windows then overlap almost
# completely (the `overlap` limitation), so raising it trades observations
# for independence.
DEFAULT_STRIDE_YEARS = 1

# Forward multiple that counts as a "winner" for tail lift. Scaled to the
# horizon above: 2x in three years is a 26% CAGR sustained three years —
# already top-decile compounding, and a threshold a fetched corpus actually
# reaches. A 5x-in-5y tail becomes measurable when the horizon can be raised.
DEFAULT_MULTIBAGGER_MULTIPLE = 2.0

# Top-K picks per cohort date for precision@K. A date needs 2*K
# observations before its picks say anything about selection (fewer, and
# "top 3 of 4" describes the date, not the score), and the fetched corpus
# rarely clears more than a handful per date.
DEFAULT_TOP_K = 3




class MultiCohortBacktest(WalkForwardBacktest):
    """Scores at many historical dates per company, measures fixed horizons.

    Inherits discovery, loading, the coverage bar, and the correlation and
    eligibility rollups from `WalkForwardBacktest`; overrides only what a
    multi-date view genuinely changes — cutoff selection, the label, the
    coverage denominator, and the report assembled around them.
    """

    def __init__(self, raw_data_dir, engine, scorer, eligibility=None,
                 min_history_years: int = MIN_HISTORY_YEARS,
                 horizon_years: int = DEFAULT_HORIZON_YEARS,
                 stride_years: int = DEFAULT_STRIDE_YEARS,
                 multibagger_multiple: float = DEFAULT_MULTIBAGGER_MULTIPLE,
                 top_k: int = DEFAULT_TOP_K,
                 **kwargs):
        # Refused rather than clamped, and named in the message. `min_history_years`
        # below 1 makes `range(min_history_years - 1, ...)` start at -1, and
        # `rows.iloc[-1]` is the NEWEST row rather than an out-of-range guard —
        # so the last cutoff is emitted twice and double-counted in the pooled
        # Spearman, the quintiles, tail lift and its own pick list. A
        # `horizon_years` of 0 resolves the same price bar as both endpoints
        # and divides by zero years. Both reach here straight off unvalidated
        # CLI flags.
        for name, value in (("min_history_years", min_history_years),
                            ("horizon_years", horizon_years),
                            ("stride_years", stride_years)):
            if value < 1:
                raise ValueError(
                    f"{name} must be at least 1, got {value} — a cutoff needs "
                    f"at least one year of history, at least one forward year "
                    f"to measure, and at least one year between cutoffs."
                )

        # `min_total_years` feeds the parent's split-half `_truncate`, which
        # this class never calls; setting it from `min_history_years` keeps
        # the inherited skip accounting honest if anything upstream reads it.
        super().__init__(raw_data_dir, engine, scorer, eligibility,
                         min_total_years=min_history_years, **kwargs)
        self.min_history_years = min_history_years
        self.horizon_years = horizon_years
        self.stride_years = stride_years
        self.multibagger_multiple = multibagger_multiple
        self.top_k = top_k

    # ── cutoffs ──────────────────────────────────────────────────────────

    def _cutoff_dates(self, financials: pd.DataFrame) -> list[pd.Timestamp]:
        """One candidate scoring date per qualifying annual row, strides apart.

        Row index i qualifies once i+1 annual rows are knowable at its
        cutoff — the row's own period end plus the reporting lag — so no
        observation ever rests on fewer than `min_history_years` scored
        years. This is `min_total_years`' per-company bar applied
        per-observation instead. Cutoffs derive from parsed period labels
        (never row position), exactly like the truncation itself, so a
        non-March fiscal year or a trailing part-year column produces a
        correct date rather than an approximate one.
        """
        # The rows the METRICS will see, not merely the ones that survive TTM
        # stripping. `point_in_time._annual_rows` drops a trailing TTM column
        # and nothing else, while every metric scores through
        # `profitability._get_annual_rows`, which also drops transition stubs
        # (`Mar 20169m`) and rows on a superseded fiscal calendar. Counting the
        # looser set made this function's own promise false: CAPLIPOINT's
        # earliest cutoff claimed five knowable years and the metrics scored
        # three.
        rows = _scoreable_rows(financials, len(financials))
        if "year" not in rows.columns:
            return []
        cutoffs: list[pd.Timestamp] = []
        for i in range(self.min_history_years - 1, len(rows), self.stride_years):
            period_end = period_end_date(rows.iloc[i]["year"])
            if period_end is not None:
                cutoffs.append(
                    period_end + pd.DateOffset(months=self.reporting_lag_months)
                )
        return cutoffs

    # ── labels ───────────────────────────────────────────────────────────

    def _forward_return(self, price: pd.DataFrame,
                        cutoff: pd.Timestamp) -> tuple[float | None, float | None, dict]:
        """CAGR and total multiple over EXACTLY `horizon_years` forward.

        Endpoint rules follow `_realized_return` — the shared column policy
        via `_return_column`, NaN rows dropped before picking, actual
        endpoint dates for annualisation — with one deliberate difference:
        the end bar is the FIRST bar at or after `cutoff + horizon_years`,
        not the latest close, so every window is the same length and
        different-length regimes stop entering one average. No bar at or
        after the target means the forward window never completed: the
        observation is censored (reported, never measured short).
        """
        column, column_reason = self._return_column(price)
        if column is None:
            return None, None, {"reason": column_reason}
        priced = price[pd.to_numeric(price[column], errors="coerce").notna()]

        at_or_before = priced[priced["date"] <= cutoff]
        if at_or_before.empty:
            return None, None, {"reason": f"no price on or before {cutoff.date()}"}
        start_row = at_or_before.iloc[-1]

        target = cutoff + pd.DateOffset(years=self.horizon_years)
        at_or_after = priced[priced["date"] >= target]
        if at_or_after.empty:
            return None, None, {
                "reason": f"price history ends before {target.date()} "
                "— forward window incomplete"
            }
        end_row = at_or_after.iloc[0]

        start, end = float(start_row[column]), float(end_row[column])
        if start <= 0 or end <= 0:
            return None, None, {"reason": "non-positive close price at a window endpoint"}
        days = (end_row["date"] - start_row["date"]).days
        years = days / 365.25
        multiple = end / start
        cagr = (multiple ** (1 / years) - 1) * 100
        return cagr, multiple, {
            "from": str(start_row["date"].date()),
            "to": str(end_row["date"].date()),
            "horizon_target": str(target.date()),
            "years": round(years, 2),
            "price_series": column,
        }

    # ── per-ticker collection ────────────────────────────────────────────

    def _collect_ticker(
        self, ticker: str, ticker_dir: Path
    ) -> tuple[list[dict], list[dict], list[dict], str | None]:
        """Truncate, compute, and label every cohort cutoff for one ticker.

        Returns `(candidates, censored, failed, skip)`. Every candidate
        cutoff lands in exactly one bucket: computed-and-labelled
        (`candidates`, holding the un-scored metric results so a second
        pass can score them against the run-wide withheld set),
        window-incomplete (`censored`), or unscorable-at-that-date
        (`failed` — the price series starts after the cutoff, e.g. a
        listing later than the financial history). `skip` is set only when
        the whole ticker is unusable, so partial failures stay visible per
        cutoff instead of hiding behind a company-level reason.
        """
        try:
            data = self._load(ticker_dir)
        except Exception as exc:
            return [], [], [], f"could not read data: {exc}"

        financials = _scoreable_rows(data["financials"], len(data["financials"]))
        if len(financials) < self.min_history_years:
            return [], [], [], (
                f"only {len(financials)} scoreable years of financials "
                f"(need {self.min_history_years})"
            )

        cutoffs = self._cutoff_dates(data["financials"])
        if not cutoffs:
            # Otherwise this ticker lands in NO bucket — not observations, not
            # censored, not failed, not skipped — and the report quietly
            # understates its own universe. `_cutoff_dates` returns nothing
            # when the frame carries no period column or no label parses, both
            # of which pass the length check above.
            return [], [], [], (
                "no parseable annual period labels to derive cutoffs from"
            )

        candidates: list[dict] = []
        censored: list[dict] = []
        failed: list[dict] = []

        for cutoff in cutoffs:
            truncated, reason = truncate_to_date(
                data, cutoff,
                annual_lag_months=self.reporting_lag_months,
                # Same stance as the single-cutoff variant: rebuilding
                # Market Cap / Stock P/E at historical dates is KTD0's
                # capability for the simulator, not this diagnostic's
                # default — the published numbers here stay built on the
                # pre-KTD0 omission.
                rebuild_valuation=False,
            )
            if truncated is None:
                failed.append({
                    "ticker": ticker,
                    "cutoff_date": str(cutoff.date()),
                    "reason": reason,
                })
                continue

            results = self.engine.run_all(truncated)

            entry = {
                "ticker": ticker,
                "cutoff_date": str(cutoff.date()),
                "results": results,
                "metadata": truncated.get("metadata", {}),
                "sector": truncated.get("metadata", {}).get("sector"),
                # What the METRICS see, not what survived truncation. The two
                # differ for any company that changed financial year end:
                # CAPLIPOINT's earliest cutoff kept five rows, of which the
                # metrics discard a nine-month stub and an old June year-end
                # and score three. Reporting the larger number made
                # `years_scored` wrong and broke `_cutoff_dates`' promise that
                # no observation rests on less history than asked for.
                "years_scored": len(_scoreable_rows(
                    truncated["financials"], len(truncated["financials"])
                )),
            }

            fwd_cagr, fwd_multiple, span = self._forward_return(data["price"], cutoff)
            if fwd_cagr is None:
                entry["status"] = "censored"
                entry["censor_reason"] = span["reason"]
                censored.append(entry)
            else:
                entry["realized_cagr_pct"] = round(fwd_cagr, 2)
                entry["fwd_multiple"] = round(fwd_multiple, 3)
                entry["forward_span"] = span
                candidates.append(entry)

        all_candidates = len(candidates) + len(censored) + len(failed)
        skip_reason = (
            failed[-1]["reason"] if all_candidates > 0 and len(failed) == all_candidates
            else None
        )
        return candidates, censored, failed, skip_reason


    # ── row assembly ─────────────────────────────────────────────────────

    def _score_entry(self, entry: dict, scorer: SQGLPScorer,
                     exclusions: dict[str, set],
                     record_exclusions: bool = True) -> dict:
        """Score one collected cutoff and attach its verdict, if gated.

        The scoring itself — sector labels, the applicability table, the
        gates' exclusions — is the parent's `_scored_fields`, so a rewound
        score means the same thing in both variants. This adds only what a
        cohort observation carries on top.

        `record_exclusions` is False for censored entries: `excluded_metrics`
        describes what shaped the published statistics, and a censored cutoff
        contributes to none of them.
        """
        row = {
            "ticker": entry["ticker"],
            "cutoff_date": entry["cutoff_date"],
            "years_scored": entry["years_scored"],
        }
        row.update(
            self._scored_fields(entry, scorer, exclusions, record_exclusions)
        )
        for key in ("status", "censor_reason", "realized_cagr_pct",
                    "fwd_multiple", "forward_span"):
            if key in entry:
                row[key] = entry[key]
        return row

    # ── distribution statistics ──────────────────────────────────────────

    @staticmethod
    def _quintile_buckets(rows: list[dict], n_buckets: int = 5) -> list[dict]:
        """Forward-return distribution by composite-score bucket.

        Buckets by rank (equal count per bucket), not by score range — on a
        hand-picked corpus the score distribution is lumpy and range buckets
        would regularly come up empty. `qcut` with duplicates="drop" yields
        fewer buckets when scores tie heavily; the caller reports how many
        materialised rather than promising five. Bucket 1 holds the lowest
        composites, so signal reads as ascending mean CAGR down the list.
        Scores tied everywhere are a special degenerate case: qcut labels
        every row NaN because nothing ranks against itself, and an empty
        result beats a fake one-bucket "distribution".
        """
        if len(rows) < n_buckets:
            return []
        composites = [r["composite_then"] for r in rows]
        bucket_ids = pd.qcut(composites, q=n_buckets, labels=False, duplicates="drop")
        paired = [
            (row, bid) for row, bid in zip(rows, bucket_ids) if pd.notna(bid)
        ]
        # One bucket is not a distribution. `duplicates="drop"` collapses a
        # lumpy score distribution into fewer bins, and with two distinct
        # composites across six rows every row lands in bucket 0 — rendered as
        # a "forward return by bucket" table whose single row is just the
        # pooled mean. The all-NaN case (total ties) was already caught; this
        # is the same degeneracy one step less extreme.
        if len({bid for _, bid in paired}) < 2:
            return []
        buckets = []
        for b in sorted({bid for _, bid in paired}):
            members = [row for row, bid in paired if bid == b]
            cagrs = [r["realized_cagr_pct"] for r in members]
            multiples = [r["fwd_multiple"] for r in members]
            buckets.append({
                "bucket": int(b) + 1,
                "n": len(members),
                "composite_range": [
                    min(r["composite_then"] for r in members),
                    max(r["composite_then"] for r in members),
                ],
                "mean_cagr_pct": round(float(np.mean(cagrs)), 2),
                "median_cagr_pct": round(float(np.median(cagrs)), 2),
                "mean_multiple": round(float(np.mean(multiples)), 2),
            })
        return buckets

    @staticmethod
    def _tail_lift(rows: list[dict],
                   multibagger_multiple: float) -> dict | None:
        """Does a high score concentrate the eventual big winners?

        Winners are observations whose forward multiple reached
        `multibagger_multiple` within the horizon. Lift compares the share
        of winners that sat in the top fifth of composite ranks against
        that fifth's base rate — 1.0 means the score said nothing about who
        the winners turned out to be. Winner counts are small on a fetched
        corpus; `winners` rides beside `lift` so the number can be
        discounted honestly instead of presented bare.
        """
        if not rows:
            return None
        ranked = sorted(rows, key=lambda r: r["composite_then"], reverse=True)
        top_n = max(1, len(ranked) // 5)
        winner_positions = {
            i for i, r in enumerate(ranked)
            if r["fwd_multiple"] >= multibagger_multiple
        }
        if not winner_positions:
            return {
                "winners": 0,
                "multibagger_multiple": multibagger_multiple,
                "lift": None,
            }
        top_positions = set(range(top_n))
        share = len(winner_positions & top_positions) / len(winner_positions)
        base_rate = top_n / len(ranked)
        return {
            "winners": len(winner_positions),
            "multibagger_multiple": multibagger_multiple,
            "top_fifth_share": round(share, 3),
            "base_rate": round(base_rate, 3),
            "lift": round(share / base_rate, 2),
        }

    @staticmethod
    def _precision_at_k(rows: list[dict], top_k: int) -> dict:
        """Per cohort date: what the top-K composite picks went on to do.

        The statistic that maps to real usage — a watchlist is exactly
        "the top few names by score on one date". Each date with at least
        2*K observations reports its picks' mean forward multiple against
        the date's universe mean; smaller dates are skipped (picking 3 of 4
        describes the date, not the score). The summary pools the per-date
        pick-to-universe ratios; across a handful of cohort dates it is a
        sketch, and the limitations block says so.
        """
        by_date: dict[str, list[dict]] = {}
        for r in rows:
            by_date.setdefault(r["cutoff_date"], []).append(r)

        dates = []
        ratios = []
        for date in sorted(by_date):
            members = by_date[date]
            if len(members) < 2 * top_k:
                continue
            picks = sorted(
                members, key=lambda r: r["composite_then"], reverse=True
            )[:top_k]
            pick_mean = float(np.mean([r["fwd_multiple"] for r in picks]))
            universe_mean = float(np.mean([r["fwd_multiple"] for r in members]))
            dates.append({
                "cutoff_date": date,
                "n": len(members),
                "picked": [r["ticker"] for r in picks],
                "pick_mean_multiple": round(pick_mean, 2),
                "universe_mean_multiple": round(universe_mean, 2),
            })
            if universe_mean > 0:
                ratios.append(pick_mean / universe_mean)

        summary = {"k": top_k, "dates_evaluated": len(dates)}
        if ratios:
            summary["mean_pick_to_universe_ratio"] = round(float(np.mean(ratios)), 2)
        return {"summary": summary, "dates": dates}

    # ── orchestration ────────────────────────────────────────────────────

    def run(self) -> dict:
        collected: list[dict] = []
        censored: list[dict] = []
        failed: list[dict] = []
        skipped: list[dict] = []
        exclusions: dict[str, set] = {}

        for ticker_dir in self.discover_candidates():
            ticker = ticker_dir.name
            missing = [f for f in REQUIRED_FILES if not (ticker_dir / f).exists()]
            if missing:
                skipped.append({"ticker": ticker, "reason": f"missing {', '.join(missing)}"})
                continue

            ticker_cols, ticker_censored, ticker_failed, skip = self._collect_ticker(
                ticker, ticker_dir
            )
            collected.extend(ticker_cols)
            censored.extend(ticker_censored)
            failed.extend(ticker_failed)
            if skip:
                skipped.append({"ticker": ticker, "reason": skip})

        # Second pass: score every collected cutoff against the run-wide
        # withheld set, so the coverage denominator excludes exactly the
        # metrics no rewound date could ever have scored (see module
        # docstring) — computed once, from this run's own evidence.
        withheld = self._withheld_metrics(collected)
        scorer = self._backtest_scorer(withheld)
        rows = [self._score_entry(entry, scorer, exclusions) for entry in collected]
        censored_rows = [
            self._score_entry(entry, scorer, exclusions, record_exclusions=False)
            for entry in censored
        ]

        usable = self._usable_rows(rows)

        return {
            "generated_for": str(self.raw_data_dir),
            "config": {
                "min_history_years": self.min_history_years,
                "horizon_years": self.horizon_years,
                "stride_years": self.stride_years,
                "multibagger_multiple": self.multibagger_multiple,
                "top_k": self.top_k,
                "min_coverage": self.min_coverage,
                "withheld_metric_count": len(withheld),
            },
            "companies": sorted(
                {r["ticker"] for r in rows} | {r["ticker"] for r in censored_rows}
            ),
            "observations": rows,
            "censored": censored_rows,
            "failed_cutoffs": failed,
            "skipped": skipped,
            "correlations": self._correlations(rows),
            "quintiles": self._quintile_buckets(usable),
            "tail_lift": self._tail_lift(usable, self.multibagger_multiple),
            "precision_at_k": self._precision_at_k(usable, self.top_k),
            "eligibility_cohorts": self._eligibility_cohorts(rows),
            "excluded_metrics": self._describe_exclusions(exclusions),
            "limitations": self._limitations(rows, censored_rows, failed, skipped),
        }

    @staticmethod
    def _limitations(rows: list[dict], censored: list[dict],
                     failed: list[dict], skipped: list[dict]) -> dict:
        return {
            "qualifying_companies": len(
                {r["ticker"] for r in rows} | {r["ticker"] for r in censored}
            ),
            "observations": len(rows),
            "censored_observations": len(censored),
            "failed_cutoffs": len(failed),
            "skipped_companies": len(skipped),
            "overlap": (
                "Consecutive cutoffs one year apart share all but one year of "
                "forward window, so observations are not independent and "
                "pooled statistics lean optimistic. Raising `stride_years` "
                "trades observations for independence."
            ),
            "survivorship": (
                "The universe is the set of tickers already fetched — companies "
                "chosen because they were interesting — not a point-in-time screen. "
                "Failures that were never fetched cannot appear."
            ),
            "fixed_horizon": (
                "Every observation measures the same number of forward years, so "
                "pooled statistics compare like with like — at the cost that "
                "recent history falls into the censored bucket, reported above "
                "rather than measured short."
            ),
            "tail_small_n": (
                "Winner counts are small on a hand-picked corpus; tail lift and "
                "precision@K carry wide error bars and are a sketch, not an "
                "estimate."
            ),
            "verdict": (
                "Sharper than the single-cutoff diagnostic — more observations, "
                "more regimes, comparable windows — but still not calibration "
                "evidence while the universe remains hand-picked."
            ),
        }
