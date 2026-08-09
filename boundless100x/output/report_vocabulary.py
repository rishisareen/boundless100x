"""What the report *calls* things — labels, bands, and the sentences it repeats.

Four hundred lines of pure vocabulary: the human-readable name for every flag,
the display name and element for every metric, the interpretation bands the
Forward Signals section reads a bare number through, the lane and verdict
labels, and the friction and break-even wording R5 requires beside every
figure. No logic, no rendering, no I/O — only what a reader sees.

It is a module of its own because it is the part of the report layer that
**grows every time anything else does**. A new metric adds a display name; a
new flag adds a label and an element; a new zero-weight signal adds a band. All
of that accumulated in front of `ReportGenerator`, so the class that renders a
report sat six hundred lines below the top of its own file and the diff for
"added a metric" and the diff for "changed how a section renders" landed in the
same place.

Three of these carry rules rather than only text, and the rules live with them:

**`FLAG_ELEMENT_MAP` falls back to `composite`** (KTD6), so a flag nobody
registered renders as an SQGLP signal on a ticker whose score did not move.
Phase 3 shipped one unregistered and proved the point; the test that catches it
now derives the flag set from the registry rather than matching id prefixes.

**`LANE_VERDICT_LABELS` is keyed on the evaluator's own constants**, imported
rather than spelled again — a rename that missed this file would render a blank
badge, which reads as a company nobody evaluated rather than as a bug.

**The friction and break-even wording is not decoration.** Every figure the
lane section shows is a modeled estimate resting on a `probe` confirmation date
rather than a fill, and market bars rather than trade prices; the sentences
here are what stop it being read as money that was made.

`report_generator.py` imports every name it uses and re-exports the ones its
callers already reached for there, so nothing moves out from under anybody.
"""

from boundless100x.lifecycle.lane_gates import (
    INDETERMINATE as LANE_INDETERMINATE,
    NOT_QUALIFIED as LANE_NOT_QUALIFIED,
    QUALIFIES as LANE_QUALIFIES,
)

# ── Human-readable flag labels ──
# Maps raw flag strings to (display_label, sentiment) where sentiment ∈ {good, bad, neutral}
FLAG_LABELS: dict[str, tuple[str, str]] = {
    # Growth
    "growth_quality_high_quality": ("High-Quality Growth", "good"),
    "growth_quality_moderate": ("Moderate Growth Quality", "neutral"),
    "growth_quality_low_quality": ("Low-Quality Growth", "bad"),
    "very_short_history_unreliable": ("Very Short History — Unreliable", "bad"),
    "bonus_split_adjusted": ("Bonus/Split Adjusted", "neutral"),
    "high_dilution": ("Significant Equity Dilution", "bad"),
    "minimal_dilution": ("Minimal Equity Dilution", "good"),
    # Profitability
    "consistently_high_roce": ("Consistently High RoCE", "good"),
    "exceptional_roce": ("Exceptional RoCE (>25%)", "good"),
    "improving_roce": ("Improving RoCE Trend", "good"),
    "declining_roce": ("Declining RoCE Trend", "bad"),
    "high_operating_margin": ("High Operating Margin", "good"),
    "improving_margins": ("Improving Margins", "good"),
    "cash_cow": ("Cash Cow — Strong Cash Conversion", "good"),
    "cfi_dominated_by_acquisitions": ("Capex Dominated by Acquisitions", "bad"),
    "volatile_tax_rate": ("Volatile Tax Rate", "neutral"),
    # Valuation
    "very_expensive_pe": ("Very Expensive PE (>80x)", "bad"),
    "expensive_pe": ("Expensive PE (>50x)", "bad"),
    "cheap_pe": ("Cheap PE (<15x)", "good"),
    "attractively_valued_peg": ("Attractively Valued (PEG < 1)", "good"),
    "expensive_peg": ("Expensive PEG (>2.5x)", "bad"),
    "attractive_trailing_peg": ("Attractive Trailing PEG", "good"),
    "pe_above_historical_75th": ("PE Above 75th Percentile — Expensive", "bad"),
    "pe_below_historical_25th": ("PE Below 25th Percentile — Cheap", "good"),
    "pe_band_legacy_price_basis": ("P/E Band Built on Adjusted Prices — Refetch to Correct", "neutral"),
    "dcf_undervalued": ("DCF: Undervalued", "good"),
    "dcf_overvalued": ("DCF: Overvalued", "bad"),
    "negative_average_fcf": ("Negative Average Free Cash Flow", "bad"),
    "negative_fcf_even_after_outlier_removal": ("Negative FCF Even After Outlier Removal", "bad"),
    "reverse_dcf_overpriced": ("Market Overpricing Growth (Reverse DCF)", "bad"),
    "reverse_dcf_underpriced": ("Market Underpricing Growth (Reverse DCF)", "good"),
    "earnings_yield_above_gsec": ("Earnings Yield Beats G-Sec", "good"),
    "gsec_more_attractive": ("G-Sec More Attractive Than Earnings Yield", "bad"),
    # Leverage
    "debt_risk": ("High Debt Risk", "bad"),
    "virtually_debt_free": ("Virtually Debt-Free", "good"),
    "low_interest_coverage": ("Weak Interest Coverage", "bad"),
    "strong_interest_coverage": ("Strong Interest Coverage", "good"),
    # Efficiency
    "improving_working_capital": ("Improving Working Capital", "good"),
    "worsening_working_capital": ("Worsening Working Capital", "bad"),
    # Size
    "small_cap": ("Small Cap", "neutral"),
    "mid_cap": ("Mid Cap", "neutral"),
    "large_cap": ("Large Cap", "neutral"),
    "micro_cap": ("Micro Cap", "neutral"),
    "low_institutional_ownership": ("Low Institutional Ownership", "neutral"),
    "heavily_institutional": ("Heavily Institutional", "neutral"),
    "under_researched": ("Under-Researched (<5 Analysts)", "neutral"),
    "lightly_covered": ("Lightly Covered (5–10 Analysts)", "neutral"),
    "promoter_increasing_stake": ("Promoter Increasing Stake", "good"),
    "promoter_reducing_stake": ("Promoter Reducing Stake", "bad"),
    "promoter_pledge_red_flag": ("Promoter Pledge — Red Flag", "bad"),
    # Longevity
    "wide_moat_cap": ("Wide Moat (Market Cap Proxy)", "good"),
    "moderate_moat_cap": ("Moderate Moat (Market Cap Proxy)", "neutral"),
    "highly_stable_margins": ("Highly Stable Margins", "good"),
    "volatile_margins": ("Volatile Margins", "bad"),
    "heavy_reinvestment": ("Heavy Reinvestment", "neutral"),
    "consistent_fcf_generator": ("Consistent Free Cash Flow Generator", "good"),
    "consistent_organic_fcf_generator": ("Consistent Organic FCF (Excl. M&A)", "good"),
    # The sector-tailwind metric's own three flags. All three were
    # unregistered: `sector_tailwind` had a *metric* display name, which is
    # what every audit of this file saw, while the flags it emits had no
    # wording at all and reached a reader as auto-humanised text.
    "sector_strong_tailwind": ("Sector With a Strong Tailwind", "good"),
    "sector_non_consideration": (
        "Sector the Study Rules Out", "bad"
    ),
    "sector_unclassified": ("Sector Not in the Study's Lists", "neutral"),
    # Composite
    "possible_bonus_split": ("Possible Bonus/Split Event Detected", "neutral"),
    # `growth.py` appends one flag *per suspect year* (`..._year_1`,
    # `..._year_2`), so the base id above never matches what a metric actually
    # emits. Generated rather than listed: a hand-written set would stop at
    # whatever year someone happened to have seen, and the year index runs to
    # the length of the annual window. Same wording, so several spikes read as
    # several years rather than as several different findings.
    **{
        f"possible_bonus_split_year_{year}": (
            f"Possible Bonus/Split Event Detected (year {year} of the window)",
            "neutral",
        )
        for year in range(1, 13)
    },
    # Forward signals (Phase 2, zero weight — see FORWARD_SIGNALS_ELEMENT)
    "rerating_headroom_favourable": ("Re-rating Headroom — Favourable", "good"),
    "rerating_headroom_stretched": ("Re-rating Headroom — Stretched", "bad"),
    "quarterly_growth_accelerating": ("Quarterly Growth Accelerating", "good"),
    "quarterly_growth_decelerating": ("Quarterly Growth Decelerating", "bad"),
    "tam_from_superseded_report": (
        "TAM Read From a Superseded Report", "neutral"
    ),
    # Phase 3 fast-lane input, also zero weight. `good` because accumulation is
    # the direction the lane's flow gate rewards — but the label says what was
    # counted rather than what it implies, since the metric moves no score.
    "institutional_accumulation_rising": ("FII + DII Accumulating", "good"),
    # Emitted by `SQGLPScorer`, not by a metric, which is how it went
    # unregistered: every audit of this table has walked the metric registry.
    # It is not a cosmetic gap — this is the flag `action_policy` caps a
    # `buy` on, so the one signal that can change the displayed action was
    # also the one with no wording of its own.
    "low_data_coverage": ("Scored on Thin Evidence", "bad"),
}

# ── Metric-to-element mapping with display labels ──
# Used for SQGLP score drill-down: maps metric_id → (element, display_name)
METRIC_DISPLAY_NAMES: dict[str, tuple[str, str]] = {
    # Size
    "market_cap": ("size", "Market Cap"),
    "institutional_holding": ("size", "FII + DII Holding"),
    "analyst_coverage": ("size", "Analyst Coverage"),
    "daily_turnover_ratio": ("size", "Daily Turnover Ratio"),
    # Quality Business
    "roce_5yr_avg": ("quality_business", "RoCE 5yr Avg"),
    "roiic": ("quality_business", "ROIIC (Incremental Capital)"),
    "capital_reinvestment_rate": ("quality_business", "Capital Reinvestment Rate"),
    "roe_5yr_avg": ("quality_business", "ROE 5yr Avg"),
    "operating_margin_5yr": ("quality_business", "OPM 5yr Avg"),
    "dupont_margin": ("quality_business", "DuPont: Net Margin"),
    "dupont_turnover": ("quality_business", "DuPont: Asset Turnover"),
    "dupont_equity_multiplier": ("quality_business", "DuPont: Equity Multiplier"),
    "cash_conversion": ("quality_business", "Cash Conversion"),
    "fcf_yield": ("quality_business", "FCF Yield"),
    "debt_equity": ("quality_business", "Debt/Equity"),
    "interest_coverage": ("quality_business", "Interest Coverage"),
    "working_capital_days_trend": ("quality_business", "Working Capital Days"),
    # Quality Management
    "promoter_holding_trend": ("quality_management", "Promoter Holding Trend"),
    "promoter_pledge": ("quality_management", "Promoter Pledge %"),
    "owner_operator_signal": ("quality_management", "Owner-Operator Signal"),
    "equity_dilution": ("quality_management", "Equity Dilution 10yr"),
    "dividend_consistency": ("quality_management", "Dividend Consistency"),
    "effective_tax_rate_variance": ("quality_management", "Tax Rate Consistency"),
    # Growth
    "revenue_cagr_5yr": ("growth", "Revenue CAGR 5yr"),
    "pat_cagr_5yr": ("growth", "PAT CAGR 5yr"),
    "eps_cagr_5yr": ("growth", "EPS CAGR 5yr"),
    "pat_cagr_3yr": ("growth", "PAT CAGR 3yr"),
    "operating_leverage": ("growth", "Operating Leverage"),
    "financial_leverage_ratio": ("growth", "Financial Leverage"),
    "growth_quality_grade": ("growth", "Growth Quality Grade"),
    "revenue_growth_consistency": ("growth", "Revenue Growth Consistency"),
    "revenue_cagr_3yr": ("growth", "Revenue CAGR 3yr"),
    "ebit_cagr_5yr": ("growth", "EBIT CAGR 5yr"),
    "ebit_cagr_3yr": ("growth", "EBIT CAGR 3yr"),
    "price_lever_signal": ("growth", "Real Revenue Growth (unscored)"),
    # Longevity
    "roce_consistency": ("longevity", "RoCE >15% Years"),
    "cap_proxy": ("longevity", "CAP Proxy"),
    "revenue_growth_streak": ("longevity", "Growth Streak"),
    "gross_margin_stability": ("longevity", "Margin Stability"),
    "reinvestment_rate": ("longevity", "Reinvestment Rate (Capex/Depn)"),
    "sector_tailwind": ("longevity", "Sector Tailwind"),
    "fcf_consistency": ("longevity", "FCF+ Years"),
    # Price
    "pe_ttm": ("price", "PE TTM"),
    "peg_ratio": ("price", "PEG Ratio"),
    "trailing_peg": ("price", "Trailing PEG"),
    "ev_ebitda": ("price", "EV/EBITDA"),
    "pe_vs_historical": ("price", "P/E vs Traded History"),
    "dcf_margin_of_safety": ("price", "DCF Margin of Safety"),
    "reverse_dcf_growth": ("price", "Reverse DCF Implied"),
    "earnings_yield_vs_gsec": ("price", "EY Spread vs G-Sec"),
}

# ── Flag-to-SQGLP element mapping ──
# Maps raw flag strings to their SQGLP element for per-section grouping.
#
# **Registration is not optional for a Phase 2 flag** (KTD6). The lookup below
# falls back to "composite" for anything unrecognised, so a zero-weight
# forward-signal metric's flag would otherwise render as an SQGLP signal on a
# ticker whose score did not move — R7's four listed quantities would all still
# hold while the report said something new about the composite. Every flag a
# forward-signal metric emits is mapped to FORWARD_SIGNALS_ELEMENT, which is
# deliberately not an SQGLP element key, so the per-element template loops
# never pick it up.
FORWARD_SIGNALS_ELEMENT = "forward_signals"

FLAG_ELEMENT_MAP: dict[str, str] = {
    # Growth
    "growth_quality_high_quality": "growth",
    "growth_quality_moderate": "growth",
    "growth_quality_low_quality": "growth",
    "very_short_history_unreliable": "growth",
    "bonus_split_adjusted": "growth",
    "high_dilution": "growth",
    "minimal_dilution": "growth",
    # Quality Business (Profitability + Leverage + Efficiency)
    "consistently_high_roce": "quality_business",
    "exceptional_roce": "quality_business",
    "improving_roce": "quality_business",
    "declining_roce": "quality_business",
    "high_operating_margin": "quality_business",
    "improving_margins": "quality_business",
    "cash_cow": "quality_business",
    "cfi_dominated_by_acquisitions": "quality_business",
    "volatile_tax_rate": "quality_business",
    "debt_risk": "quality_business",
    "virtually_debt_free": "quality_business",
    "low_interest_coverage": "quality_business",
    "strong_interest_coverage": "quality_business",
    "improving_working_capital": "quality_business",
    "worsening_working_capital": "quality_business",
    # Price (Valuation)
    "very_expensive_pe": "price",
    "expensive_pe": "price",
    "cheap_pe": "price",
    "attractively_valued_peg": "price",
    "expensive_peg": "price",
    "attractive_trailing_peg": "price",
    "pe_above_historical_75th": "price",
    "pe_below_historical_25th": "price",
    "dcf_undervalued": "price",
    "dcf_overvalued": "price",
    "negative_average_fcf": "price",
    "negative_fcf_even_after_outlier_removal": "price",
    "reverse_dcf_overpriced": "price",
    "reverse_dcf_underpriced": "price",
    "earnings_yield_above_gsec": "price",
    "gsec_more_attractive": "price",
    # Size
    "small_cap": "size",
    "mid_cap": "size",
    "large_cap": "size",
    "micro_cap": "size",
    "low_institutional_ownership": "size",
    "heavily_institutional": "size",
    "under_researched": "size",
    "lightly_covered": "size",
    # Quality Management
    "promoter_increasing_stake": "quality_management",
    "promoter_reducing_stake": "quality_management",
    "promoter_pledge_red_flag": "quality_management",
    # Longevity
    "wide_moat_cap": "longevity",
    "moderate_moat_cap": "longevity",
    "highly_stable_margins": "longevity",
    "volatile_margins": "longevity",
    "heavy_reinvestment": "longevity",
    "consistent_fcf_generator": "longevity",
    "consistent_organic_fcf_generator": "longevity",
    "sector_strong_tailwind": "longevity",
    "sector_non_consideration": "longevity",
    "sector_unclassified": "longevity",
    # Composite
    "possible_bonus_split": "composite",
    **{
        f"possible_bonus_split_year_{year}": "composite"
        for year in range(1, 13)
    },
    # The scorer's own flag. Mapped explicitly to the same element the `.get`
    # default would have returned, so the registration is visible rather than
    # accidental — the default is what let it go unnoticed in FLAG_LABELS.
    "low_data_coverage": "composite",
    # Forward signals (Phase 2, zero weight)
    "rerating_headroom_favourable": FORWARD_SIGNALS_ELEMENT,
    "rerating_headroom_stretched": FORWARD_SIGNALS_ELEMENT,
    "quarterly_growth_accelerating": FORWARD_SIGNALS_ELEMENT,
    "quarterly_growth_decelerating": FORWARD_SIGNALS_ELEMENT,
    "tam_from_superseded_report": FORWARD_SIGNALS_ELEMENT,
    # Phase 3 (zero weight). Registered here rather than under `size` although
    # `institutional_accumulation_streak` lives in size.yaml: the rule is about
    # what moved the score, and this metric moved nothing. Under `size` the
    # report would show a new Size signal on a ticker whose Size score is
    # unchanged.
    "institutional_accumulation_rising": FORWARD_SIGNALS_ELEMENT,
}

# ── Forward signals (Phase 2, zero weight) ──
#
# Every one of these carries `weight: 0`, which means the scorer never gives it
# a score. So the number is all the reader gets, and a bare number is not
# signal: nobody can tell whether +40 is good news without recomputing the
# metric. R8 exists for that reason, and this table is how it is satisfied —
# each signal declares its direction of goodness, what it means in one line,
# and the bands that turn its value into a reading.
#
# `bands` is walked in order and the first threshold the value reaches wins;
# `low_label` catches everything below all of them. Thresholds here are
# STARTING POINTS, like every other number this phase introduces. A metric that
# supplies its own `metadata["band"]` overrides this — `rerating_headroom`
# does, because its bands are owner-editable in the metric's YAML params and a
# tuned band must win over a default declared here.
FORWARD_SIGNALS: dict[str, dict] = {
    "rerating_headroom": {
        "name": "Re-rating Headroom",
        "format": "{:+.0f}%",
        "direction": "higher is better",
        "meaning": (
            "How far above today's traded multiple the company's own RoCE, "
            "growth and consistency would justify."
        ),
        "bands": [(25.0, "favourable"), (-25.0, "fair")],
        "low_label": "stretched",
    },
    "promises_kept_ratio": {
        "name": "Promises Kept",
        "format": "{:.0f}%",
        "direction": "higher is better",
        "meaning": (
            "Share of management's own due targets that the accounts later met. "
            "Promises not yet due are excluded from both sides."
        ),
        "bands": [(80.0, "credible"), (50.0, "mixed")],
        "low_label": "unreliable",
    },
    "tam_runway": {
        "name": "TAM Runway",
        "format": "{:.0f} yrs",
        "direction": "higher is better",
        "meaning": (
            "Years at the current growth rate before revenue meets the "
            "addressable market management describes."
        ),
        "bands": [(15.0, "long"), (7.0, "adequate")],
        "low_label": "short",
    },
    "quarterly_momentum": {
        "name": "Quarterly Growth Momentum",
        "format": "{:+.1f}pp",
        "direction": "higher is better",
        "meaning": (
            "Change between consecutive year-over-year growth figures — whether "
            "growth is speeding up, not how fast it is."
        ),
        "bands": [(2.0, "accelerating"), (-2.0, "steady")],
        "low_label": "decelerating",
    },
}

FORWARD_SIGNALS_DISCLAIMER = (
    "These signals inform the thesis but do not contribute to the SQGLP "
    "composite. They carry zero weight, receive no score, and are excluded "
    "from the coverage denominator."
)

MOMENTUM_UNAVAILABLE_LABEL = "Not enough history yet"


# ── Lane & Friction (Phase 3) ──
# Everything below renders only when the caller supplied a `lane_context`,
# which only a watchlisted ticker has. A company analysed outside the watchlist
# renders exactly what it rendered before this section existed (KTD9).

LANE_LABELS: dict[str, str] = {
    "core": "Core — the compounder lane",
    "rerating": "Re-rating — the fast lane",
}

# The same two lanes where a heading's worth of words will not fit — a table
# column an owner scans down, rather than a section title they read once.
# `ELEMENT_CONFIG` already carries `label` beside `short` for exactly this, so
# a second length is the established shape here and not a new idea. Still one
# statement per lane: a test pins that these keys are `LANE_LABELS`'s keys, and
# that each short form opens the long one, so the two cannot come to name
# different things.
LANE_SHORT_LABELS: dict[str, str] = {
    "core": "Core",
    "rerating": "Re-rating",
}

# The lifecycle states, in words. `LANE_LABELS`'s sibling and added for the
# same surface: `watchlist show` printed `exit_review` and `probe` straight out
# of the store, which is exactly the lifecycle key R15 keeps off the page.
#
# **Each label says what the state means for capital**, because that is the
# distinction `states.py` opens on — the order is by commitment, and a reader
# scanning a column wants to know which rows have money in them. Short by
# necessity: this renders in a narrow table column beside six others.
#
# The keys are spelled rather than imported, matching `LANE_LABELS` above; a
# test derives the expected set from `lifecycle.states.STATES`, so a state
# added without a label fails the suite rather than rendering as its key.
STATE_LABELS: dict[str, str] = {
    "screen": "Screening",
    "qualify": "Qualifying",
    "watch": "Watching for entry",
    "probe": "Probe position",
    "scale": "Scaled position",
    "exit_review": "Exit under review",
    # Not "Exited" and not "Dropped": a key with a capital letter is the
    # identifier wearing a hat, which is the fallback this vocabulary exists to
    # replace rather than a shorter way of writing it. A test refuses any label
    # that is its own key with the underscores taken out.
    "exited": "Sold and closed",
    "dropped": "No longer tracked",
}

# A lane or state the vocabulary has no wording for. Never auto-humanised —
# see `FLAG_LABELS`'s note on why a derived label is a leak with better
# typography.
LIFECYCLE_UNKNOWN_LABEL = "a lifecycle value this system has no wording for"

# The fast lane's verdict vocabulary, kept **out of** the 100x badge's words on
# purpose. `not_qualified` and `not_eligible` are different findings about
# different questions — a company can fail every 100x gate and still be a sound
# re-rating candidate, which is the asymmetry the whole lane exists for — so a
# reader must never meet "eligible" in this section and carry the other
# question's meaning into it.
#
# The labels are this layer's own — presentation is not the evaluator's job —
# but the *keys* are imported, so the map is guaranteed to cover the verdicts
# that actually arrive.
#
# `_LABELS`, not `LANE_VERDICTS`: that name already belongs to `lane_gates`,
# where it is the *vocabulary* — a tuple of the three verdict strings. One name
# for two incompatible types across two modules is a reader's problem before it
# is anyone else's, and the two are imported into the same test file.
LANE_VERDICT_LABELS: dict[str, tuple[str, str, str]] = {
    LANE_QUALIFIES: (
        "Qualifies for the fast lane", "good",
        "Clears every fast-lane entry gate",
    ),
    LANE_NOT_QUALIFIED: (
        "Does not qualify for the fast lane", "bad",
        "Fails at least one fast-lane entry gate",
    ),
    LANE_INDETERMINATE: (
        "Fast-lane qualification unknown", "neutral",
        "A fast-lane entry gate could not be evaluated from available data",
    ),
}

FRICTION_UNAVAILABLE_LABEL = "Modeled friction unavailable"

# `recorded` says the two dates stopped moving, never that the figure stopped
# being a model — so both labels lead with the same word.
FRICTION_BASIS_LABELS: dict[str, str] = {
    "estimate": "Modeled estimate — the exit date is still moving",
    "recorded": "Modeled at the recorded exit — the dates are fixed, the figure is still a model",
}

# No backticks: one string reaches an HTML template that renders no markdown
# and a markdown template that does, so anything only one of them understands
# shows up as punctuation in the other.
FRICTION_NOTE = (
    "Every figure here is a model. The holding period runs from the probe "
    "confirmation date rather than a broker fill, the prices are market bars "
    "rather than trade prices, and no cost basis is recorded anywhere in this "
    "system."
)

# §8.2's break-even framing, for the fast lane only.
#
# **No hurdle is computed, and that is a decision rather than an omission.** A
# capital-gains rate applies to a gain; it is not a number of return points, so
# turning "20% STCG and 100bps round trip" into a single percentage the lane
# must beat would require an assumed holding period, an assumed turnover rate
# and an assumed alternative — three numbers nobody has supplied. A figure
# derived from invented inputs would be read as a threshold, and the fast lane
# would then look "accelerated" when it was merely busier. Phase 4's simulator
# derives one from owner cost assumptions; until then this states the roadmap's
# rough estimate as an estimate, with the rates it rests on listed beside it.
BREAKEVEN_ESTIMATE = "6–10 percentage points more per cycle"

BREAKEVEN_STATEMENT = (
    "A re-rating round trip pays capital-gains tax and slippage that a held "
    "position never pays, so it has to earn more just to come out level. The "
    "roadmap's rough estimate for that difference is 6–10 percentage points "
    "more per cycle — an estimate, stated with the assumptions it rests on."
)

BREAKEVEN_CAVEAT = (
    "No hurdle number is computed here. A tax rate applies to a gain, not to a "
    "number of return points, so any single figure would be arithmetic these "
    "inputs do not support — the Phase 4 simulator derives one from owner cost "
    "assumptions."
)


# ── Named grades (R15) ──
#
# Five metrics declare `presentation.unit: "category"`, and their value *is* a
# raw enum: `founder_led_high_holding`, `true_wealth_creator`, `discounting`.
# R15 forbids those reaching a reader, and the fallback everywhere else in this
# codebase — `value.replace("_", " ").title()` — is the defect the report
# clarity work exists to remove, not the fix for it. "Founder Led High Holding"
# is the identifier wearing a hat: it still says nothing about what the grade
# means, and it succeeds silently on a grade nobody has thought about.
#
# So: `{metric_id: {value: (label, gloss)}}`. Keyed by metric first because the
# grades are only unique inside their own metric — `unknown` means "this sector
# was not in the study's lists" for `sector_tailwind` and would mean something
# else entirely anywhere else — and because that shape lets a test derive the
# expected key set from each metric's `scoring.categories` table rather than
# from a list somebody has to remember to update.
#
# The `gloss` is the row's actual reading. These metrics declare no numeric
# bands, so without it the row would render the declared `bands_absent_reason`,
# which explains to a *developer* why a band walk was skipped. The glosses below
# are read off the implementations that emit the grades — the promoter-holding
# cutoffs in `compute_owner_operator`, the driver logic in
# `_grade_growth_quality`, the inflation comparison in `compute_price_lever`,
# `classify_sector`'s study buckets, and `compute_qg_quadrant`'s two lines — so
# no grade here is invented and none is a paraphrase of a paraphrase.
CATEGORICAL_VALUE_LABELS: dict[str, dict[str, tuple[str, str]]] = {
    "owner_operator_signal": {
        "founder_led_high_holding": (
            "Founder-led, majority stake",
            "The promoters hold at least half the company, so the people "
            "running it carry the same downside as the people who own it.",
        ),
        "founder_led_moderate": (
            "Founder-led, substantial stake",
            "The promoters hold a large minority — enough to think like "
            "owners, not enough to decide alone.",
        ),
        "professional_mgmt": (
            "Professionally managed",
            "The promoters hold a modest stake, so the business is run by "
            "managers rather than by its owners.",
        ),
        "low_promoter": (
            "Little promoter ownership",
            "Almost nobody running the company owns much of it, so the "
            "alignment this model looks for is not there.",
        ),
    },
    "growth_quality_grade": {
        "high_quality": (
            "High quality",
            "Growth came from selling more at better prices and from fixed "
            "costs spreading over a bigger base — the two levers that survive "
            "a downturn.",
        ),
        "moderate": (
            "Moderate",
            "Growth came from one durable lever rather than two, and not from "
            "borrowing.",
        ),
        "low_quality": (
            "Low quality",
            "Borrowed money was part of what drove growth, or none of the "
            "durable levers was.",
        ),
        "risky": (
            "Risky",
            "Borrowed money was the only thing driving growth, which works "
            "until the cycle turns.",
        ),
    },
    "price_lever_signal": {
        "strong_pricing_power": (
            "Well ahead of inflation",
            "Sales grew a good deal faster than prices generally rose. It "
            "cannot tell selling more from charging more.",
        ),
        "moderate_pricing": (
            "Ahead of inflation",
            "Sales outran inflation, but not by much. It cannot tell selling "
            "more from charging more.",
        ),
        "discounting": (
            "Behind inflation",
            "Sales failed to keep pace with prices generally rising, so the "
            "business shrank in real terms.",
        ),
    },
    "sector_tailwind": {
        "strong_tailwind": (
            "Strong tailwind",
            "Sits in one of the sectors the Dec 2025 study found compounders "
            "cluster in. Context about the pond, not a verdict on the fish.",
        ),
        "moderate_tailwind": (
            "Moderate tailwind",
            "Sits in a sector where the study found some compounders, though "
            "not a cluster of them.",
        ),
        "unknown": (
            "Sector not classified",
            "Either no sector is recorded for this company or its sector is "
            "not one the study placed, so this says nothing either way.",
        ),
        "non_consideration": (
            "Against the current",
            "Sits in a sector the study found compounders largely absent "
            "from.",
        ),
    },
    "quality_growth_quadrant": {
        "true_wealth_creator": (
            "True wealth creator",
            "High returns on capital and high profit growth together — the "
            "only corner of the grid the Dec 2025 study found wealth creators "
            "in.",
        ),
        "quality_trap": (
            "Quality trap",
            "Earns well on its capital, but profit growth sits below the bar, "
            "so there is little to compound.",
        ),
        "growth_trap": (
            "Growth trap",
            "Grows profits fast on capital that earns poorly, so growth "
            "consumes more than it returns.",
        ),
        "wealth_destroyer": (
            "Wealth destroyer",
            "Neither returns on capital nor profit growth clears the bar.",
        ),
    },
}


# ── The ten-point scale, in words ──
#
# Five bands, not three, and they are **not** the cutoffs the dashboard colours
# by. Those are 7 and 4, and lifting them was a mistake worth naming: a colour
# bucket answers "green, amber or red" and an interpretation band answers "what
# is this number telling me", and the two need different resolution. On a
# three-band table every one of the eight scored companies read `middling` —
# composites run 4.2 to 5.96 and the whole range sat in one bucket, so the
# headline sentence of every report said the same word. A band that cannot
# separate the corpus it describes carries no information, which is the exact
# defect this vocabulary exists to remove.
#
# The boundaries are **absolute**, not fitted to the corpus: they say what the
# model means by a score, so a report does not change its wording because other
# companies were analysed later. Against today's 48 element scores they fall
# 10 / 13 / 13 / 8 / 4, which is a real spread rather than a fitted one —
# a useful check that the words discriminate, not the reason they were chosen.
#
# Walked in order, first threshold reached wins — the same rule as every
# `presentation.bands` list, so there is one band-walking convention here
# rather than two.
SCORE_SCALE = 10
SCORE_BANDS: tuple[tuple[float, str], ...] = (
    (8.0, "exceptional"),
    (6.5, "strong"),
    (5.0, "fair"),
    (3.5, "thin"),
)
SCORE_LOW_LABEL = "weak"

# What each band looks like on a terminal. Declared beside the bands rather
# than in `cli.py`, where it was a literal map of three words that silently
# fell through to `dim` the moment a fourth existed — a score rendered with no
# colour reads as a score nobody could band, which is a different claim. A test
# pins these keys equal to the band words, so adding a band without a colour is
# a failure here rather than a grey row on the console.
SCORE_BAND_COLOURS: dict[str, str] = {
    "exceptional": "bright_green",
    "strong": "green",
    "fair": "yellow",
    "thin": "bright_red",
    "weak": "red",
}

# The composite, as a surface names it and reads it. `report_generator`'s
# `_clarity_lead` builds the note's opening line from the same band walk, so
# the word after "Reads" is the same word on every surface by construction —
# what this adds is that the *sentence* is too, rather than two spellings of
# one reading that nobody would notice diverging.
COMPOSITE_TITLE = "Composite"
COMPOSITE_READING = "Reads {band} across the six scored elements."

# What the console says instead of a score it has no band for. `section_reading`
# already produces an unknown-with-reason for an unscored element; this is the
# composite's equivalent, which no element-shaped builder covers.
COMPOSITE_UNKNOWN_REASON = (
    "nothing in this company could be scored, so there is no overall reading "
    "— which is not the same as a score of zero"
)


# ── The research note (U10) ───────────────────────────────────────────────
#
# Wording the new report needs and no existing table holds. It lives here for
# the same reason everything else in this file does: these are the words a
# reader meets, and the alternative is a literal inside a renderer where
# nothing keeps it in step with the rest of the report's voice.

# The action, as a reader meets it. The existing surfaces render this enum as
# `action | replace("_", " ") | upper`, which is one of the five routes around
# the vocabulary layer the problem frame names — "STRONG BUY" is the key
# shouting rather than a label. `guard_text` refuses any of the five raw keys
# as a whole field, so a new report cannot render one by accident; this is what
# it renders instead.
ACTION_LABELS: dict[str, str] = {
    "avoid": "Avoid",
    "watchlist": "Watchlist",
    "hold": "Hold",
    "buy": "Buy",
    "strong_buy": "Strong buy",
}

# An action outside `ACTION_ORDER`. Never auto-humanised — see `FLAG_LABELS`'s
# note on why a derived label is a leak with better typography.
ACTION_UNKNOWN_LABEL = "an action this report has no wording for"

RESEARCH_NOTE_TITLE = "Research Note"

# What a metric row says when the metric produced no figure at all. R4 forbids
# the empty cell and R12 forbids the bare number; a dash is the empty cell with
# a character in it, so the cell says what happened instead.
NO_FIGURE_LABEL = "no figure"

# The contribution cell for a metric that carries no weight — the zero-weight
# signals, and any metric the scorer waived. Distinct from a metric that scored
# zero, which has a score to show.
UNWEIGHTED_CONTRIBUTION = "Does not contribute to the score"

# R3's link text. The same words on both surfaces, so a reader who learns what
# the phrase means in one report recognises it in the other.
DISCLOSURE_LINK_TEXT = "what this measures"

# Why a collapsed section shows no rows, said once rather than per section.
# R5 is deliberate — length is the verdict (KD5) — but a reader meeting six
# one-line sections deserves to be told the rows still exist somewhere.
COLLAPSED_SECTIONS_NOTE = (
    "A section with nothing to explain renders as its score and one line. "
    "Every metric's own row, and the model's written thesis where there is "
    "one, stay in the full dashboard generated beside this note."
)

# The section carrying every metric that sits outside the six scored elements.
# They are not a seventh element and must not read like one: nothing in here
# reaches a score, a coverage denominator, or the composite.
UNSCORED_SECTION_TITLE = "Signals that move no score"

UNSCORED_SECTION_READING = (
    "Read for context. Nothing below contributed to any score on this page."
)


# ── SQGLP element display config ──
ELEMENT_CONFIG: dict[str, dict] = {
    "size": {"label": "Size", "short": "S", "weight": "10%"},
    "quality_business": {"label": "Quality — Business", "short": "QB", "weight": "20%"},
    "quality_management": {"label": "Quality — Management", "short": "QM", "weight": "10%"},
    "growth": {"label": "Growth", "short": "G", "weight": "25%"},
    "longevity": {"label": "Longevity", "short": "L", "weight": "20%"},
    "price": {"label": "Price", "short": "P", "weight": "15%"},
}

