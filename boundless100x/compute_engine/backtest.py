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
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Fetched artefacts that cannot be rewound to a historical date.
NON_TRUNCATABLE_INPUTS = ("shareholding", "analyst_coverage", "shareholding_bse")

# Frames that carry one row per financial year and can be truncated.
ANNUAL_FRAMES = ("financials", "balance_sheet", "cashflow", "ratios")

# A directory with financials is a real ticker. BSE-code directories hold only
# annual report PDFs and are not tickers at all.
TICKER_MARKER = "financials.csv"
REQUIRED_FILES = ("financials.csv", "price_volume.csv")

MIN_TOTAL_YEARS = 8
# Indian annual results are filed within months of the year end; scoring at the
# year end itself would use figures that were not yet public.
REPORTING_LAG_MONTHS = 6
MIN_FORWARD_DAYS = 365
# Minimum surviving metric weight before a company's composite is comparable.
MIN_SCORED_WEIGHT = 1.5


def _annual_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "year" not in df.columns:
        return df
    return df[~df["year"].astype(str).str.contains("TTM", case=False, na=False)]


def _year_of(label: str) -> int | None:
    match = re.search(r"(\d{4})", str(label))
    return int(match.group(1)) if match else None


class WalkForwardBacktest:
    """Scores on the first half of history, measures the second half's return."""

    def __init__(self, raw_data_dir, engine, scorer, eligibility=None,
                 min_total_years: int = MIN_TOTAL_YEARS,
                 reporting_lag_months: int = REPORTING_LAG_MONTHS):
        self.raw_data_dir = Path(raw_data_dir)
        self.engine = engine
        self.scorer = scorer
        self.eligibility = eligibility
        self.min_total_years = min_total_years
        self.reporting_lag_months = reporting_lag_months
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
        """Split history in half and rebuild a point-in-time view of the past."""
        financials = _annual_rows(data["financials"])
        total_years = len(financials)
        if total_years < self.min_total_years:
            return None, None, f"only {total_years} years of financials (need {self.min_total_years})"

        keep = total_years // 2
        cutoff_year = _year_of(financials.iloc[keep - 1]["year"]) if "year" in financials else None
        if cutoff_year is None:
            return None, None, "could not read a fiscal year label"

        # Indian fiscal years end 31 March, but the accounts are not public
        # until months later. Observing at the year end would score on figures
        # nobody could have read — look-ahead by another name.
        truncation_date = pd.Timestamp(year=cutoff_year, month=3, day=31) + pd.DateOffset(
            months=self.reporting_lag_months
        )

        truncated = {}
        for name in ANNUAL_FRAMES:
            if name not in data:
                continue
            frame = _annual_rows(data[name])
            # Cut on the fiscal-year label: frames do not all start in the same
            # year, so a positional cut would leave post-cutoff rows in place.
            years = frame["year"].map(_year_of) if "year" in frame.columns else None
            if years is not None and years.notna().any():
                frame = frame[years <= cutoff_year]
            else:
                frame = frame.head(keep)
            truncated[name] = frame.reset_index(drop=True)

        price = data["price"]
        past_price = price[price["date"] <= truncation_date]
        if past_price.empty:
            return None, None, f"price history starts after {truncation_date.date()}"
        truncated["price"] = past_price.reset_index(drop=True)

        truncated["metadata"] = self._point_in_time_metadata(
            data["_metadata_raw"], truncated, past_price
        )
        # Belt and braces: the loader never reads these, and nothing may add them.
        for leaky in NON_TRUNCATABLE_INPUTS:
            truncated.pop(leaky, None)
        return truncated, truncation_date, ""

    @staticmethod
    def _point_in_time_metadata(raw: dict, truncated: dict, past_price: pd.DataFrame) -> dict:
        """Rebuild only what is genuinely knowable at the truncation date.

        Market cap is omitted on purpose: reconstructing it would take a share
        count this pipeline does not store reliably, and today's value is the
        single worst leak available — a company that re-rated would be scored
        on its post-re-rating size precisely because it later did well.
        """
        close = float(past_price.iloc[-1]["close"])
        meta = {
            "name": raw.get("name"),
            "sector": raw.get("sector"),          # static enough to carry over
            "Face Value": raw.get("Face Value"),
            "Current Price": close,
        }

        # Stock P/E is omitted for the same reason as market cap: the stored
        # closes are split- and dividend-adjusted, so dividing one by an
        # unadjusted historical EPS would not be the P/E anyone saw. Metrics
        # needing it exclude themselves rather than score a fabricated ratio.
        return meta

    @staticmethod
    def _realized_return(price: pd.DataFrame, truncation_date: pd.Timestamp) -> tuple[float | None, dict]:
        """Annualised return from the truncation date to the latest close."""
        at_or_before = price[price["date"] <= truncation_date]
        after = price[price["date"] > truncation_date]
        if at_or_before.empty or after.empty:
            return None, {"reason": "no price on both sides of the truncation date"}

        start_row, end_row = at_or_before.iloc[-1], after.iloc[-1]
        start, end = float(start_row["close"]), float(end_row["close"])
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

            scored_weight = sum(
                d.get("weight", 0) or 0 for d in scores.get("details", {}).values()
            )
            row = {
                "ticker": ticker,
                "scored_weight": round(scored_weight, 3),
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
        # A composite normalised over a sliver of surviving weight is not
        # rank-comparable with a fully evidenced one.
        usable = [
            r for r in rows
            if r["composite_then"] is not None and r["scored_weight"] >= MIN_SCORED_WEIGHT
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
