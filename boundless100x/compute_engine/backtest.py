"""Walk-forward self-check: does the score predict the subsequent return?

Every threshold in this system encodes a finding about companies that already
compounded. That is survivorship-biased by construction, and until the scores
are compared against outcomes the weights are untested assumptions.

This scores each company using only the first half of its fetched history and
compares that score against the return it actually delivered afterwards.

Look-ahead leakage is the central correctness risk. Any input carrying today's
state into a historical score manufactures correlation: a company that already
re-rated would otherwise be scored on its post-re-rating market cap precisely
because it went on to do well. Two rules keep that out, and every exclusion is
reported rather than silently absorbed:

  * Inputs that cannot be rewound — shareholding history and analyst coverage —
    are withheld, so metrics needing them error and are listed as excluded.
  * Metadata is rebuilt from the truncation date, carrying only the price the
    series actually records. Market cap and P/E are deliberately omitted —
    stored closes are split- and dividend-adjusted, so a rebuilt ratio is not
    the one anyone saw — and the metrics needing them exclude themselves.

The result is a diagnostic, not calibration evidence. See `LIMITATIONS`.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from boundless100x.compute_engine.metrics.builtin._helpers import period_end_date
from boundless100x.compute_engine.point_in_time import (
    ANNUAL_FRAMES,
    ANNUAL_REPORTING_LAG_MONTHS as REPORTING_LAG_MONTHS,
    _annual_rows,
    truncate_to_date,
)
from boundless100x.data_fetcher.corpus_snapshot import TICKER_MARKER

logger = logging.getLogger(__name__)

# A directory with financials is a real ticker. BSE-code directories hold only
# annual report PDFs and are not tickers at all.
REQUIRED_FILES = ("financials.csv", "price_volume.csv")

MIN_TOTAL_YEARS = 8
MIN_FORWARD_DAYS = 365


class WalkForwardBacktest:
    """Scores on the first half of history, measures the second half's return."""

    def __init__(self, raw_data_dir, engine, scorer, eligibility=None,
                 min_total_years: int = MIN_TOTAL_YEARS,
                 reporting_lag_months: int = REPORTING_LAG_MONTHS,
                 min_coverage: float | None = None):
        self.raw_data_dir = Path(raw_data_dir)
        self.engine = engine
        self.scorer = scorer
        self.eligibility = eligibility
        self.min_total_years = min_total_years
        self.reporting_lag_months = reporting_lag_months
        # Same bar production uses to flag a composite as resting on thin
        # evidence (scorer.low_coverage_threshold, docstring in scorer.py) —
        # a backtest row below it is not comparable to a fully-scored one on
        # equal footing, so it must not enter the correlation either. Sourced
        # from the scorer rather than a second constant here so the two
        # cannot silently drift apart.
        self.min_coverage = (
            min_coverage if min_coverage is not None else scorer.low_coverage_threshold
        )
        # The size gate needs a market cap that cannot be rebuilt without leaking.
        self.gate_evaluator = None
        if eligibility is not None:
            testable = {k: v for k, v in eligibility.gates.items() if k != "size"}
            self.gate_evaluator = type(eligibility)(testable)

    # ── discovery ────────────────────────────────────────────────────────

    def discover_candidates(self) -> list[Path]:
        """Directories holding a real ticker's fetched data.

        BSE-code directories carry only annual report PDFs. They are not
        tickers at all, so they never pad the skip list — which exists to show
        genuine tickers that could not be evaluated. A ticker missing its price
        history *is* a genuine ticker, so it is skipped with a reason rather
        than dropped.
        """
        if not self.raw_data_dir.exists():
            return []
        return sorted(
            d for d in self.raw_data_dir.iterdir()
            if d.is_dir() and (d / TICKER_MARKER).exists()
        )

    # ── per-ticker evaluation ────────────────────────────────────────────

    def _load(self, ticker_dir: Path) -> dict:
        data = {}
        for name in ANNUAL_FRAMES:
            path = ticker_dir / f"{name}.csv"
            if path.exists():
                data[name] = pd.read_csv(path)
        price = pd.read_csv(ticker_dir / "price_volume.csv")
        price["date"] = pd.to_datetime(price["date"], errors="coerce", utc=True).dt.tz_localize(None)
        data["price"] = price.dropna(subset=["date"]).sort_values("date")
        meta_path = ticker_dir / "metadata.json"
        data["_metadata_raw"] = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return data

    def _truncate(self, data: dict) -> tuple[dict | None, pd.Timestamp | None, str]:
        """Split history in half and delegate to `point_in_time.truncate_to_date`.

        This method owns only the backtest's own policy — which row is the
        split-half cutoff — and hands the actual "what was knowable" logic to
        the shared module (KTD2), which both this class and a later simulator
        call for identical answers.
        """
        financials = _annual_rows(data["financials"])
        total_years = len(financials)
        if total_years < self.min_total_years:
            return None, None, f"only {total_years} years of financials (need {self.min_total_years})"

        keep = total_years // 2
        cutoff_period_end = (
            period_end_date(financials.iloc[keep - 1]["year"]) if "year" in financials else None
        )
        if cutoff_period_end is None:
            return None, None, "could not read a fiscal year label"

        # The accounts are not public until months after the period end.
        # Observing at the period end itself would score on figures nobody
        # could have read — look-ahead by another name. Deriving this from
        # the parsed label (rather than assuming March) also makes it correct
        # for a non-March fiscal year end, not just the common case.
        truncation_date = cutoff_period_end + pd.DateOffset(months=self.reporting_lag_months)

        truncated, reason = truncate_to_date(
            data,
            truncation_date,
            annual_lag_months=self.reporting_lag_months,
            annual_fallback_rows=keep,
            # KTD0(a) lives in the shared module so a later simulator caller
            # can opt in, but the backtest's own published correlations and
            # test suite were built on the pre-KTD0 omission — rebuilding
            # Market Cap/Stock P/E here would newly satisfy `market_cap`,
            # `pe_vs_historical` and the `price`/`size` eligibility gates for
            # any ticker whose corpus reconciles, which is a real scoring
            # change, not a refactor. This flag is what keeps this call
            # byte-identical to the pre-KTD2 behaviour.
            rebuild_valuation=False,
        )
        if truncated is None:
            return None, None, reason
        return truncated, truncation_date, ""

    @staticmethod
    def _realized_return(price: pd.DataFrame, truncation_date: pd.Timestamp) -> tuple[float | None, dict]:
        """Annualised return from the truncation date to the latest close.

        Uses the split/dividend-adjusted close when the series carries a
        genuine one — a raw close would read a 1:5 split as an 80% loss.
        `adj_close_is_estimated` marks fetches (jugaad-data fallback) where
        `adj_close` is just `close` under another name; that aliasing is not
        safe to score a realized return against, so those histories are
        excluded rather than silently measured on an unadjusted series.
        Legacy files predating the flag (single `close` column, no alias)
        fall back to the raw close, same as before.

        **Rows with no value in the chosen column are dropped before the
        endpoints are picked.** The source publishes the most recent bar's raw
        close before its adjusted one, so a series fetched today routinely ends
        in a single NaN `adj_close`. Reading the last row unconditionally turns
        that into a NaN return — and it stayed invisible while the tickers it
        affects had no adjusted series at all and fell back to `close`. The
        corpus refetch gave all 22 an adjusted series and five of them promptly
        produced NaN realized returns, which is how this surfaced.
        """
        if "adj_close" in price.columns and "adj_close_is_estimated" in price.columns:
            if bool(price["adj_close_is_estimated"].iloc[-1]):
                return None, {
                    "reason": "adj_close is an unadjusted-close fallback "
                    "(jugaad-data source) — cannot validate a realized return"
                }

        column = "adj_close" if "adj_close" in price.columns else "close"
        priced = price[pd.to_numeric(price[column], errors="coerce").notna()]

        at_or_before = priced[priced["date"] <= truncation_date]
        after = priced[priced["date"] > truncation_date]
        if at_or_before.empty or after.empty:
            return None, {"reason": "no price on both sides of the truncation date"}

        start_row, end_row = at_or_before.iloc[-1], after.iloc[-1]
        start, end = float(start_row[column]), float(end_row[column])
        days = (end_row["date"] - start_row["date"]).days
        if start <= 0 or end <= 0:
            return None, {"reason": "non-positive close price at a window endpoint"}
        if days < MIN_FORWARD_DAYS:
            return None, {"reason": f"only {days} days of forward price history"}

        years = days / 365.25
        return ((end / start) ** (1 / years) - 1) * 100, {
            "from": str(start_row["date"].date()),
            "to": str(end_row["date"].date()),
            "years": round(years, 2),
            "price_series": column,
        }

    # ── orchestration ────────────────────────────────────────────────────

    def run(self) -> dict:
        rows, skipped = [], []
        exclusions: dict[str, set] = {}

        for ticker_dir in self.discover_candidates():
            ticker = ticker_dir.name
            missing = [f for f in REQUIRED_FILES if not (ticker_dir / f).exists()]
            if missing:
                skipped.append({"ticker": ticker, "reason": f"missing {', '.join(missing)}"})
                continue

            try:
                data = self._load(ticker_dir)
            except Exception as exc:
                skipped.append({"ticker": ticker, "reason": f"could not read data: {exc}"})
                continue

            truncated, truncation_date, reason = self._truncate(data)
            if truncated is None:
                skipped.append({"ticker": ticker, "reason": reason})
                continue

            realized, span = self._realized_return(data["price"], truncation_date)
            if realized is None:
                skipped.append({"ticker": ticker, "reason": span["reason"]})
                continue

            results = self.engine.run_all(truncated)
            scores = self.scorer.score(results)

            for metric_id, result in results.items():
                if not result.ok:
                    exclusions.setdefault(metric_id, set()).add(ticker)

            row = {
                "ticker": ticker,
                "coverage_composite": scores.get("coverage", {}).get("composite"),
                "truncation_date": str(truncation_date.date()),
                "years_scored": len(truncated["financials"]),
                "forward_span": span,
                "composite_then": scores.get("composite"),
                "elements_then": scores.get("elements", {}),
                "realized_cagr_pct": round(realized, 2),
            }
            if self.gate_evaluator:
                verdict = self.gate_evaluator.evaluate(results)
                row["eligibility_then"] = {
                    "verdict": verdict["verdict"],
                    "gates_evaluated": sorted(self.gate_evaluator.gates),
                    "note": "size gate excluded — market cap is not reconstructable",
                }
            rows.append(row)

        return {
            "generated_for": str(self.raw_data_dir),
            "companies": rows,
            "correlations": self._correlations(rows),
            "eligibility_cohorts": self._eligibility_cohorts(rows),
            "excluded_metrics": self._describe_exclusions(exclusions),
            "skipped": skipped,
            "limitations": self._limitations(rows, skipped),
        }

    @staticmethod
    def _spearman(xs: list[float], ys: list[float]) -> float | None:
        """Rank correlation, computed without a hard scipy dependency."""
        if len(xs) < 3:
            return None
        rx, ry = pd.Series(xs).rank(), pd.Series(ys).rank()
        if rx.std() == 0 or ry.std() == 0:
            return None
        return round(float(np.corrcoef(rx, ry)[0, 1]), 3)

    def _correlations(self, rows: list[dict]) -> dict:
        # A composite whose coverage falls below the same bar production uses
        # to flag thin evidence (self.min_coverage) is not rank-comparable
        # with a fully evidenced one.
        usable = [
            r for r in rows
            if r["composite_then"] is not None
            and (r.get("coverage_composite") or 0) >= self.min_coverage
        ]
        returns = [r["realized_cagr_pct"] for r in usable]

        correlations = {
            "composite_vs_return": self._spearman(
                [r["composite_then"] for r in usable], returns
            ),
            "n": len(usable),
        }

        elements: dict[str, float | None] = {}
        for element in self.engine.element_weights:
            pairs = [
                (r["elements_then"].get(element), r["realized_cagr_pct"])
                for r in usable
                if r["elements_then"].get(element) is not None
            ]
            if len(pairs) >= 3:
                elements[element] = self._spearman([p[0] for p in pairs], [p[1] for p in pairs])
        correlations["elements_vs_return"] = elements
        return correlations

    def _eligibility_cohorts(self, rows: list[dict]) -> dict | None:
        """Forward-return distribution by 100x-eligibility verdict.

        The gates are a conjunctive filter separate from the additive
        composite above — "could this plausibly 100x?" rather than "is this
        a quality compounder?". Each row already carries its own verdict
        (eligibility_then.verdict), computed once per company and otherwise
        never rolled up anywhere. This is the one place that checks whether
        the verdict predicts anything: if the "eligible" cohort's returns
        aren't better than "not_eligible", the gates are not doing their job
        and the thresholds need to be revisited, not trusted on intuition.

        Every row with a verdict counts, including ones _correlations()
        would treat as too-thin-coverage for the composite comparison — the
        gates read their own inputs and go indeterminate on missing data
        independently of the general metric-coverage bar.
        """
        if not self.gate_evaluator:
            return None

        cohorts: dict[str, list[float]] = {}
        for r in rows:
            verdict = (r.get("eligibility_then") or {}).get("verdict")
            if verdict is None:
                continue
            cohorts.setdefault(verdict, []).append(r["realized_cagr_pct"])

        def summarize(returns: list[float]) -> dict:
            arr = np.array(returns)
            return {
                "n": len(returns),
                "mean_cagr_pct": round(float(arr.mean()), 2),
                "median_cagr_pct": round(float(np.median(arr)), 2),
                "min_cagr_pct": round(float(arr.min()), 2),
                "max_cagr_pct": round(float(arr.max()), 2),
            }

        return {verdict: summarize(returns) for verdict, returns in sorted(cohorts.items())}

    @staticmethod
    def _describe_exclusions(exclusions: dict[str, set]) -> list[dict]:
        return sorted(
            (
                {"metric": metric, "tickers_affected": len(tickers)}
                for metric, tickers in exclusions.items()
            ),
            key=lambda e: (-e["tickers_affected"], e["metric"]),
        )

    @staticmethod
    def _limitations(rows: list[dict], skipped: list[dict]) -> dict:
        windows = [r["truncation_date"] for r in rows]
        return {
            "qualifying_companies": len(rows),
            "skipped_companies": len(skipped),
            "score_dates": sorted(set(windows)),
            "shared_window": (
                "All companies are scored and measured over one overlapping "
                "macro period, so the result reflects that cycle as much as the "
                "framework."
            ),
            "survivorship": (
                "The universe is the set of tickers already fetched — companies "
                "chosen because they were interesting — not a point-in-time screen. "
                "Failures that were never fetched cannot appear."
            ),
            "truncated_history": (
                "Scores are computed on deliberately shortened histories, so "
                "short-window effects apply that a real analysis at that date "
                "would not have had."
            ),
            "verdict": (
                "Diagnostic smoke test, not calibration evidence. Do not retune "
                "thresholds on this sample alone."
            ),
        }
