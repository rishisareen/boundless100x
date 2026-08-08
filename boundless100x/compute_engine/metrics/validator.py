"""YAML schema validation for metric registry files."""

REQUIRED_FIELDS = [
    "name",
    "module",
    "function",
    "inputs",
    "scoring",
    "display",
    "presentation",
]
VALID_DIRECTIONS = ["higher_is_better", "lower_is_better"]
VALID_MODES = [
    "threshold",
    "range_optimal",
    "categorical",
    "sector_relative_percentile",
    "trend_direction",
    "comparison_to_actual",
]


# ── Presentation: what a number *is*, so a reader can read it (R11, R12) ──
#
# The report's Value column used to hold `25.7` (a percentage), `0.09` (a
# ratio), `2.0` (a count of years) and `0.84` (a spread in percentage points)
# with nothing to tell them apart. `presentation` is each metric's own answer
# to that: its unit, its direction of goodness, what it means in one line, and
# the bands that turn a number into a reading.
#
# **`unit` is not `display.format`, and neither replaces the other.** The
# format string beside it (`display.format`) is typography — decimals, sign,
# the ₹ prefix — and it can only decide how to *print*. `unit` is the
# dimension, drawn from a closed set, and it is what lets a component decide
# how to *label* a number and whether two numbers may be compared at all. A
# `format` key here would be a second statement of `display.format`, and two
# statements of one fact drift invisibly, so `presentation` deliberately has
# none. This block is otherwise the shape `report_vocabulary.FORWARD_SIGNALS`
# already uses for the four zero-weight signals — `direction`, `meaning`,
# `bands`, `low_label` — so those four can later collapse onto it rather than
# leaving the project with two declaration shapes. Their `format` strings are
# already byte-identical to the same metrics' `display.format`, so nothing is
# lost in that collapse.
#
# The vocabulary is **closed and enforced** for the reason KTD5 gives about
# extraction units: an open string field lets `pct`, `percent` and `%` all
# appear, and a unit nobody can group on is not a unit. It is also built from
# what the shipped registry actually measures rather than from first
# principles — a member no metric uses is a reading nobody can check.
PRESENTATION_UNITS: dict[str, str] = {
    "percent": (
        "A rate or share out of a hundred. A RoCE of 22 means 22%."
    ),
    "percentage_points": (
        "The gap between two percentages, or the spread of one. Kept apart "
        "from `percent` because the arithmetic differs and the mistake is "
        "silent: 20% rising to 25% is +5pp, but it is also +25%."
    ),
    "multiple": (
        "A ratio of two like quantities, read as 'times' — a P/E of 25 is "
        "25x, a debt/equity of 0.3 is 0.3x. Valuation multiples and "
        "structural ratios share this unit because they share a reading: the "
        "number says how many times over."
    ),
    "inr_crore": (
        "A rupee amount in crore, the unit every financial frame here is in."
    ),
    "years": (
        "A number of years — a count of good ones, a streak, or a runway."
    ),
    "days": (
        "A number of days. Deliberately not `years`: the two are both "
        "durations and a component that treated them alike would render "
        "'132' as a century of working capital."
    ),
    "count": (
        "A plain count of things that are not periods — analysts covering "
        "the company, consecutive quarterly rises."
    ),
    "percentile": (
        "A rank inside a distribution, 0 to 100. Not a percent: the 75th "
        "percentile of a P/E band is a position in a history, not a "
        "proportion of anything."
    ),
    "category": (
        "A named grade rather than a number. Has no band walk and no "
        "direction — see below."
    ),
}

# The direction of goodness **of the value the reader is shown**.
#
# The first two are the scoring config's own words (`VALID_DIRECTIONS`),
# reused rather than re-spelled so a metric cannot claim one direction to the
# scorer and another to the reader — `_validate_presentation` pins that
# agreement wherever the scoring config states a direction at all.
#
# The other two exist because forcing every metric into higher/lower would
# lie about a third of the registry. A range-optimal metric is worse at both
# ends; a categorical one has no ordering in its value at all, only in the
# scoring config's `categories` table.
PRESENTATION_DIRECTIONS: dict[str, str] = {
    "higher_is_better": "More of this is better, without limit.",
    "lower_is_better": "Less of this is better, without limit.",
    "range_optimal": (
        "Good sits inside a declared range; both ends are worse. Mirrors the "
        "`range_optimal` scoring mode."
    ),
    "not_directional": (
        "No monotone direction exists, because the value is a named grade "
        "rather than a quantity."
    ),
}


def _validate_presentation(prefix: str, config: dict, errors: list[str]) -> None:
    """Check one metric's `presentation` block (R11, R12, R3).

    Missing entirely is a startup error, the same way a duplicate metric id
    is: a metric with no declared unit renders as a bare number, and a bare
    number in a table of percentages, ratios, year-counts and multiples is
    unreadable by anyone not already holding the model in their head. Catching
    it at construction is what stops a metric being added without one.

    **Band semantics, stated once here because every declaration depends on
    them.** `bands` is a list of `[threshold, label]` walked in order, and the
    first entry whose threshold the value *reaches* (`value >= threshold`)
    wins; `low_label` catches everything below all of them. This is
    `report_generator._forward_band`, unchanged — the four zero-weight signals
    have been read this way since Phase 2.

    Two consequences worth being explicit about:

    * **Thresholds must descend, and that is checked.** A list authored
      ascending is not merely mis-sorted: its first entry has the lowest
      threshold, so it swallows every value and every band beneath it is
      unreachable. The reading would be wrong on every company and the config
      would look fine.
    * **Descending order is about the thresholds, not the labels.** For a
      `lower_is_better` metric the numbers still descend, so the *labels* run
      worst-first and `low_label` names the best outcome rather than the
      worst — `debt_equity` reads "leveraged" at 0.5x and "virtually
      debt-free" below 0.1x. Nothing about the walk changes; only which end of
      it is the good news.
    * A threshold is an inclusive lower bound, so a `range_optimal` metric's
      upper band opens exactly *at* the top of its optimal range. A value
      sitting on that boundary reads as above the range while scoring as
      inside it — a hair's-width disagreement accepted rather than papered
      over with an invented epsilon.

    An **empty** `bands` list is a legitimate declaration, not an oversight,
    and it must carry `bands_absent_reason` saying why. That covers the two
    honest cases: a categorical metric whose value is already its own reading,
    and a metric whose reading is genuinely relative (to a sector, to the
    company's own record) where an absolute band would be wrong on most
    companies. A wrong band is worse than a declared unknown, so the reason is
    required rather than optional — it is what the row renders instead.
    """
    block = config.get("presentation")
    if block is None:
        return  # Already reported by the REQUIRED_FIELDS sweep.
    if not isinstance(block, dict):
        errors.append(f"{prefix}: presentation must be a mapping")
        return

    unit = block.get("unit")
    if unit not in PRESENTATION_UNITS:
        errors.append(
            f"{prefix}: presentation.unit {unit!r} is not one of "
            f"{sorted(PRESENTATION_UNITS)}"
        )

    direction = block.get("direction")
    if direction not in PRESENTATION_DIRECTIONS:
        errors.append(
            f"{prefix}: presentation.direction {direction!r} is not one of "
            f"{sorted(PRESENTATION_DIRECTIONS)}"
        )

    meaning = block.get("meaning")
    if not isinstance(meaning, str) or not meaning.strip():
        errors.append(
            f"{prefix}: presentation.meaning must be a non-empty line saying "
            f"what the metric measures and what good looks like (R3)"
        )

    # One metric, one direction. A scored metric that told the scorer
    # `lower_is_better` and the reader `higher_is_better` would render a
    # reading that contradicts its own score, on every company, silently.
    # Trend and comparison modes state a direction the scorer applies to a
    # *series* (`declining_is_better`) rather than to the value on the page,
    # so they are outside this rule by construction — they are not in
    # VALID_DIRECTIONS.
    scoring_direction = (config.get("scoring") or {}).get("direction")
    if scoring_direction in VALID_DIRECTIONS and direction != scoring_direction:
        errors.append(
            f"{prefix}: presentation.direction {direction!r} contradicts "
            f"scoring.direction {scoring_direction!r}"
        )

    if unit == "category" and direction not in (None, "not_directional"):
        errors.append(
            f"{prefix}: a category-valued metric cannot be {direction!r} — "
            f"its value is a name, and names have no ordering"
        )

    bands = block.get("bands")
    if not isinstance(bands, list):
        errors.append(
            f"{prefix}: presentation.bands must be a list of "
            f"[threshold, label] pairs (empty is allowed, with a reason)"
        )
        return

    if unit == "category" and bands:
        # The band walk compares with `>=`, which a string value can never
        # satisfy — the metric would render a blank reading rather than an
        # error. The ranking of a categorical metric's grades is the scoring
        # config's `categories` table, stated there and nowhere else.
        errors.append(
            f"{prefix}: a category-valued metric cannot declare numeric "
            f"bands; its value is already its own reading"
        )

    previous: float | None = None
    for index, band in enumerate(bands):
        where = f"{prefix}: presentation.bands[{index}]"
        if not isinstance(band, (list, tuple)) or len(band) != 2:
            errors.append(f"{where} must be a [threshold, label] pair")
            continue
        threshold, label = band
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            errors.append(f"{where} threshold {threshold!r} is not a number")
            continue
        if not isinstance(label, str) or not label.strip():
            errors.append(f"{where} label {label!r} must be a non-empty string")
        if previous is not None and threshold >= previous:
            errors.append(
                f"{where} threshold {threshold} does not fall below the one "
                f"before it ({previous}) — the walk takes the first threshold "
                f"reached, so this band and every band under it are unreachable"
            )
        previous = threshold

    if bands:
        low_label = block.get("low_label")
        if not isinstance(low_label, str) or not low_label.strip():
            errors.append(
                f"{prefix}: presentation.low_label must name the reading for "
                f"values below every band"
            )
        if block.get("bands_absent_reason"):
            errors.append(
                f"{prefix}: presentation declares bands and a "
                f"bands_absent_reason — only one of them can be true"
            )
    else:
        reason = block.get("bands_absent_reason")
        if not isinstance(reason, str) or not reason.strip():
            errors.append(
                f"{prefix}: presentation declares no bands, so it must carry "
                f"a bands_absent_reason — that reason is what the row renders "
                f"in place of a reading"
            )
        if "low_label" in block:
            errors.append(
                f"{prefix}: presentation declares no bands, so low_label "
                f"names a reading nothing can produce"
            )


def validate_registry(metrics: dict) -> list[str]:
    """Validate all metric definitions. Returns list of errors (empty = valid)."""
    errors = []
    for metric_id, config in metrics.items():
        src = config.get("_source_file", "unknown")
        prefix = f"[{src}] {metric_id}"

        for field in REQUIRED_FIELDS:
            if field not in config:
                errors.append(f"{prefix}: missing '{field}'")

        # Presentation is checked for **every** metric, whatever its weight.
        # It is the one part of a definition a zero-weight metric needs *more*
        # of, not less: the scorer never gives it a score, so the number and
        # its band are the entire reading (R8).
        _validate_presentation(prefix, config, errors)

        scoring = config.get("scoring", {})
        if "weight" not in scoring:
            errors.append(f"{prefix}: scoring.weight required")

        mode = scoring.get("mode", "threshold")
        if mode not in VALID_MODES:
            errors.append(f"{prefix}: invalid mode '{mode}'")

        # A metric at weight 0 is display-only: the scorer's `weight == 0`
        # branch returns before `_compute_raw_score` is ever reached, so
        # thresholds, ranges and categories declared for it are dead config.
        # Demanding them produced exactly that — zero-weight metrics carrying
        # invented category tables purely to pass validation. `weight` and a
        # recognised `mode` are still required; only the mode's own inputs are
        # excused, because nothing will read them.
        if not (scoring.get("weight") or 0) > 0:
            continue

        if mode == "threshold":
            if "thresholds" not in scoring:
                errors.append(f"{prefix}: threshold mode needs 'thresholds'")
            if scoring.get("direction") not in VALID_DIRECTIONS:
                errors.append(f"{prefix}: invalid direction '{scoring.get('direction')}'")
        elif mode == "range_optimal":
            if "optimal_range" not in scoring:
                errors.append(f"{prefix}: range_optimal mode needs 'optimal_range'")
        elif mode == "categorical":
            if "categories" not in scoring:
                errors.append(f"{prefix}: categorical mode needs 'categories'")
        elif mode == "trend_direction":
            if "direction" not in scoring:
                errors.append(f"{prefix}: trend_direction mode needs 'direction'")

    return errors
