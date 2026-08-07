"""Size metrics: Market cap, institutional holding, analyst coverage, turnover, promoter."""

import numpy as np
import pandas as pd

from boundless100x.compute_engine.metrics.base import MetricResult
from boundless100x.compute_engine.metrics.builtin._helpers import quarter_index
from boundless100x.compute_engine.metrics.builtin.profitability import _get_annual_rows


def compute_market_cap(data: dict, params: dict) -> MetricResult:
    """Market Cap from metadata (already in ₹ Cr from Screener.in)."""
    meta = data.get("metadata", {})
    mcap = meta.get("Market Cap")
    if mcap is None:
        return MetricResult(error="No market cap in metadata")

    flags = []
    if mcap < 5000:
        flags.append("small_cap")
    elif mcap < 20000:
        flags.append("mid_cap")
    else:
        flags.append("large_cap")

    return MetricResult(value=float(mcap), flags=flags)


def compute_institutional_holding(data: dict, params: dict) -> MetricResult:
    """FII + DII holding from latest shareholding quarter."""
    sh = data["shareholding"]
    if sh.empty:
        return MetricResult(error="No shareholding data")

    latest = sh.iloc[-1]
    fii = pd.to_numeric(latest.get("fii_pct"), errors="coerce")
    dii = pd.to_numeric(latest.get("dii_pct"), errors="coerce")

    if pd.isna(fii) and pd.isna(dii):
        return MetricResult(error="Missing FII/DII data")

    fii = 0.0 if pd.isna(fii) else float(fii)
    dii = 0.0 if pd.isna(dii) else float(dii)
    total = fii + dii

    flags = []
    if total < 5:
        flags.append("low_institutional_ownership")
    elif total > 40:
        flags.append("heavily_institutional")

    return MetricResult(
        value=total,
        flags=flags,
        metadata={"fii_pct": fii, "dii_pct": dii},
    )


def compute_institutional_accumulation_trend(data: dict, params: dict) -> MetricResult:
    """Consecutive quarter-on-quarter rises in FII + DII holding.

    The fast lane's flow gate (v05 §9.2, "FII+DII rising for 2+ consecutive
    quarters"). Built on `compute_promoter_trend`'s shape — a value, a change,
    and a `raw_series` — over the two institutional legs combined, since a
    re-rating usually shows up as somebody accumulating before it shows up in
    the accounts.

    Three things about the count are stated rather than implied, because the
    obvious phrasings each give a different number:

    * **The frame is read in file order, and the walk runs backward from the
      last row.** `shareholding.csv` is stored oldest first. Reading it the
      other way would report a company being sold as one being accumulated,
      which is the single most expensive mistake this metric could make — so
      the order is **verified rather than assumed**, and a frame that is not
      ascending errors. Left unchecked it failed *closed*, which sounds safe
      and is the wrong kind of wrong: a newest-first file breaks the adjacency
      test at the very first step, the walk returns 0, and the gate reads "no
      accumulation" indefinitely on a company being steadily accumulated, with
      nothing anywhere saying why. A silent zero and a real zero are the same
      number; only one of them is a reading.
    * **The unit counted is rises, not observations.** Four strictly increasing
      quarters yield 3 — three comparisons between four points — so the gate's
      `>= 2` asks for two consecutive rises across three quarters.
    * **A rise counts only between rows exactly one quarter apart.** This is
      deliberately stricter than `compute_promoter_trend`, which reads by
      position: a 20-quarter *trend* can absorb a missing filing, but a
      consecutive-quarters *gate* is defined by adjacency, and "FII+DII rose
      across a hole in the data" is missing evidence rather than a rise. Same
      rule, and the same reason, as `quarterly_momentum`'s period matching.

    A label this parser cannot read errors outright: adjacency is unverifiable
    on a frame whose periods cannot be placed, and falling back to position
    there would reintroduce exactly the gap-as-rise reading the rule above
    exists to prevent. An error reads as gate-indeterminate, never a pass.
    """
    min_rises_to_flag = int(params.get("rising_min_rises", 2))

    sh = data.get("shareholding")
    if not isinstance(sh, pd.DataFrame) or sh.empty:
        return MetricResult(error="No shareholding data")
    if "quarter" not in sh.columns:
        return MetricResult(
            error=(
                "shareholding data carries no period labels, so consecutive "
                "quarters cannot be told apart from a gap"
            )
        )
    for column in ("fii_pct", "dii_pct"):
        if column not in sh.columns:
            return MetricResult(error=f"No {column} column in shareholding data")

    fii = pd.to_numeric(sh["fii_pct"], errors="coerce")
    dii = pd.to_numeric(sh["dii_pct"], errors="coerce")

    periods: list[int] = []
    combined: list[float] = []
    labels: list[str] = []
    for label, foreign, domestic in zip(sh["quarter"], fii, dii):
        index = quarter_index(label)
        if index is None:
            return MetricResult(
                error=(
                    f"Unreadable shareholding period label {label!r} — quarter "
                    f"adjacency cannot be verified, so no rise can be confirmed"
                )
            )
        # **Both legs must be present.** A point-in-time read can treat a
        # missing FII as zero and lose only precision; a *difference* cannot,
        # because the next row's full figure would then read as a jump — a rise
        # manufactured out of a data gap. An unreadable row simply leaves the
        # series, where the adjacency rule below ends the walk at it.
        if pd.isna(foreign) or pd.isna(domestic):
            continue
        periods.append(index)
        combined.append(float(foreign) + float(domestic))
        labels.append(str(label))

    if len(combined) < 2:
        return MetricResult(
            error=(
                f"{len(combined)} readable quarter(s) of FII+DII holding, needs "
                f"2 — a rise is a comparison between two"
            )
        )

    # Checked before the walk, because the walk cannot tell "this file is
    # backwards" from "this company was not accumulated" — both come out as 0.
    # `<=` rather than `<`: a repeated period is equally unwalkable, since the
    # adjacency arithmetic below would compare a quarter against itself.
    out_of_order = next(
        (
            (labels[i - 1], labels[i])
            for i in range(1, len(periods))
            if periods[i] <= periods[i - 1]
        ),
        None,
    )
    if out_of_order:
        return MetricResult(
            error=(
                f"shareholding periods are not in ascending order "
                f"({out_of_order[0]} is followed by {out_of_order[1]}) — this "
                f"walk reads the file oldest-first, and reading it the other way "
                f"would report a company being sold as one being accumulated"
            )
        )

    rises = 0
    for i in range(len(combined) - 1, 0, -1):
        if periods[i] - periods[i - 1] != 1:
            break
        if combined[i] <= combined[i - 1]:
            break
        rises += 1

    flags = []
    if rises >= min_rises_to_flag:
        flags.append("institutional_accumulation_rising")

    return MetricResult(
        value=float(rises),
        raw_series=combined,
        flags=flags,
        metadata={
            "latest_combined_pct": combined[-1],
            "latest_quarter": labels[-1],
            "readable_quarters": len(combined),
            "quarters_in_file": len(sh),
        },
    )


def compute_analyst_count(data: dict, params: dict) -> MetricResult:
    """Number of analysts covering the company."""
    ac = data.get("analyst_coverage", {})
    count = ac.get("count")
    if count is None:
        return MetricResult(error="No analyst coverage data")

    flags = []
    if count <= 3:
        flags.append("under_researched")
    elif count <= 5:
        flags.append("lightly_covered")

    return MetricResult(value=float(count), flags=flags)


def compute_turnover_ratio(data: dict, params: dict) -> MetricResult:
    """Daily Turnover Ratio = Avg Daily Volume × Price / Market Cap × 100.

    Measures liquidity. Very low = hard to accumulate/exit.
    """
    price_df = data["price"]
    if price_df.empty or len(price_df) < 20:
        return MetricResult(error="Insufficient price data")

    meta = data.get("metadata", {})
    mcap = meta.get("Market Cap")
    if mcap is None or mcap == 0:
        return MetricResult(error="No market cap for turnover calculation")

    # Last 90 trading days average
    recent = price_df.tail(90)
    avg_volume = recent["volume"].mean()
    avg_price = recent["close"].mean()

    if pd.isna(avg_volume) or pd.isna(avg_price):
        return MetricResult(error="Cannot compute average volume/price")

    # Daily turnover value in Cr (volume × price / 1e7)
    daily_turnover_cr = avg_volume * avg_price / 1e7
    ratio = float(daily_turnover_cr / mcap * 100)

    return MetricResult(value=ratio)


def compute_promoter_trend(data: dict, params: dict) -> MetricResult:
    """Promoter holding trend over N quarters."""
    quarters = params.get("quarters", 20)
    sh = data["shareholding"]
    if sh.empty:
        return MetricResult(error="No shareholding data")

    promoter = pd.to_numeric(sh["promoter_pct"], errors="coerce").dropna()
    if len(promoter) < 4:
        return MetricResult(error="Insufficient promoter holding data")

    latest = float(promoter.iloc[-1])
    earliest = float(promoter.iloc[0])
    change = latest - earliest

    flags = []
    if change > 2:
        flags.append("promoter_increasing_stake")
    elif change < -5:
        flags.append("promoter_reducing_stake")

    return MetricResult(
        value=latest,
        raw_series=promoter.tolist(),
        flags=flags,
        metadata={"change_pp": change, "quarters_used": len(promoter)},
    )


def compute_promoter_pledge(data: dict, params: dict) -> MetricResult:
    """Promoter pledge percentage, from BSE data only.

    An unpledged promoter and an unknown one look identical if absence scores
    as 0 — the best possible outcome. Pledge is a risk flag; only a verified
    observation may clear it. When BSE has not supplied the figure, this
    reports unavailable rather than guessing.
    """
    # Try BSE supplemental data first
    bse_sh = data.get("shareholding_bse")
    if bse_sh is not None and not bse_sh.empty and "promoter_pledge_pct" in bse_sh.columns:
        pledge = pd.to_numeric(bse_sh["promoter_pledge_pct"], errors="coerce").dropna()
        if len(pledge) > 0:
            val = float(pledge.iloc[-1])
            flags = []
            if val > 10:
                flags.append("promoter_pledge_red_flag")
            return MetricResult(value=val, flags=flags)

    return MetricResult(error="Promoter pledge data not available from BSE")


def compute_owner_operator(data: dict, params: dict) -> MetricResult:
    """Owner-operator signal based on promoter holding level."""
    min_holding = params.get("min_promoter_holding", 40)
    sh = data["shareholding"]
    if sh.empty:
        return MetricResult(error="No shareholding data")

    promoter = pd.to_numeric(sh["promoter_pct"], errors="coerce").dropna()
    if len(promoter) == 0:
        return MetricResult(error="No promoter holding data")

    latest = float(promoter.iloc[-1])

    if latest >= 50:
        category = "founder_led_high_holding"
    elif latest >= min_holding:
        category = "founder_led_moderate"
    elif latest >= 20:
        category = "professional_mgmt"
    else:
        category = "low_promoter"

    return MetricResult(
        value=category,
        metadata={"promoter_pct": latest},
    )
