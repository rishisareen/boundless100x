"""What a number *means* — and, when nothing can be said, what stopped it.

A pure function of declarations plus computed values (KTD2). It reads the
`presentation:` blocks in `elements/*.yaml`, the sector-applicability table in
`compute_engine/sector_applicability.yaml`, and the per-element coverage the
scorer already computes; it turns each into a reader-facing reading. It calls
no model, opens no socket, and reads no file of its own. That is what makes R2
hold on `analyze --no-llm` and on `watchlist advance` **by construction**: those
paths run no model at all, so a reading sourced from one would leave every
section on them opening on a blank.

The dependency direction is deliberate and one-way. This module imports from
`compute_engine`; the report and CLI surfaces import *this*. It must never
import `boundless100x.llm_layer` (KTD2) and must never import
`report_generator`, which would invert the direction the plan sequences these
units in. A test walks this file's own AST to keep both edges closed, because
an import-time check on a module nobody imported passes vacuously.

**The unknown cases are the substance here, not the edge.** The band walk this
generalises — `report_generator._forward_band` — answers every absence with
`""`: an errored metric, a string where a number was expected, a metric whose
bands are deliberately undeclared. An empty string in a reading column is
indistinguishable from a reading nobody got round to writing, and that is
exactly the blank R4 forbids. So this layer has one success status and six
distinct failures, each carrying a sentence a reader can act on:

    read                 a band label resolved
    no_declaration       nothing declares how to read this metric
    metric_error         the computation failed, and here is what it said
    value_absent         nothing was computed, which is not the same as zero
    bands_not_declared   deliberately unbanded — nine shipped metrics are, and
                         the declared `bands_absent_reason` is what the row
                         renders instead of a reading
    value_not_bandable   a value the declared numeric bands cannot place
    not_applicable       the metric measures nothing for a company of this kind

Six buckets rather than one because they are six different things for a reader
to do, and a single "unknown" loses every one of them.

**R12 is enforced by the type, not by the docstring.** A number leaves this
module only inside a `Quantity`, which cannot be constructed without a unit and
a direction drawn from the validator's closed vocabularies, and whose `str()`
always renders the unit alongside the figure. `Reading` has no `value`
attribute at all — the only route to the number is through `quantity`. A
surface can still reach `quantity.value` when it needs the bare figure for
logic, and that reach is explicit and greppable rather than the path of least
resistance.

Two things this layer deliberately does **not** do. It does not humanise a
categorical value: `owner_operator_signal` reads `founder_led_high_holding` and
turning that into "Founder Led High Holding" here would be the auto-humanising
fallback the problem frame names as a defect. The raw grade travels with
`unit == "category"` so the component set (U9) can look up a real label and
R15 stays that unit's job. And it does not name an element: `quality_business`
is a raw key, so the coverage clause states the share without it and the caller
prepends the label from `report_vocabulary.ELEMENT_CONFIG`.
"""

from __future__ import annotations

import numbers
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from boundless100x.compute_engine.metrics.validator import (
    PRESENTATION_DIRECTIONS,
    PRESENTATION_UNITS,
)
from boundless100x.compute_engine.scorer import SQGLPScorer
from boundless100x.compute_engine.sector import (
    APPLIES,
    DOES_NOT_APPLY,
    INDETERMINATE,
)

# ── Reading statuses ──────────────────────────────────────────────────────
#
# One success and six failures. `NOT_APPLICABLE` is a *reading* status and is
# not the same name as `sector.DOES_NOT_APPLY`, which is an *applicability*
# verdict: the verdict is the table's answer about the metric, the status is
# what this layer then does with the number.

READ = "read"
NO_DECLARATION = "no_declaration"
METRIC_ERROR = "metric_error"
VALUE_ABSENT = "value_absent"
BANDS_NOT_DECLARED = "bands_not_declared"
VALUE_NOT_BANDABLE = "value_not_bandable"
NOT_APPLICABLE = "not_applicable"

READING_STATUSES = frozenset({
    READ, NO_DECLARATION, METRIC_ERROR, VALUE_ABSENT, BANDS_NOT_DECLARED,
    VALUE_NOT_BANDABLE, NOT_APPLICABLE,
})

# How each failure opens its sentence. The reason follows the colon, so the
# prefix is what separates "we could not" from "we chose not to" from "it would
# mean nothing here" before the reader has read a word of the reason.
STATUS_PREFIXES: dict[str, str] = {
    NO_DECLARATION: "No reading available",
    METRIC_ERROR: "Could not be computed",
    VALUE_ABSENT: "Nothing was computed",
    BANDS_NOT_DECLARED: "No interpretation band is declared",
    VALUE_NOT_BANDABLE: "No reading available",
    NOT_APPLICABLE: "Not applicable to this company",
}


# ── How a number is spoken ────────────────────────────────────────────────
#
# `display.format` is the primary rendering and always wins — it is the
# metric's own typography and the only statement of it (the validator's own
# note explains why `presentation` carries no `format` key). What follows is a
# **fallback and a label**, not a second copy of it, and it exists for one
# reason: a caller holding only a `presentation` block — which is how this
# layer is unit-tested, and how a future surface might reach it — must still be
# unable to render a bare numeral. Fallback text is never what a report shows;
# if it ever is, a metric lost its `display` block and that is a bug upstream.
#
# `(prefix, suffix)` per unit. `count` and `category` have no natural marker,
# which is handled below rather than papered over.
UNIT_AFFIXES: dict[str, tuple[str, str]] = {
    "percent": ("", "%"),
    "percentage_points": ("", "pp"),
    "multiple": ("", "x"),
    "years": ("", " years"),
    "days": ("", " days"),
    "count": ("", ""),
    "inr_crore": ("₹", " Cr"),
    "percentile": ("", "th percentile"),
    "category": ("", ""),
}

# The dimension in words, for a surface that needs to say what a figure is
# rather than print it — and for the two units with no affix, so even the
# fallback cannot emit a naked number.
UNIT_PHRASES: dict[str, str] = {
    "percent": "percent",
    "percentage_points": "percentage points",
    "multiple": "times",
    "years": "years",
    "days": "days",
    "count": "count",
    "inr_crore": "₹ crore",
    "percentile": "percentile rank",
    "category": "grade",
}

# The direction of goodness, in the reader's words. The first two match the
# strings `report_vocabulary.FORWARD_SIGNALS` already renders, so the four
# zero-weight signals read identically before and after they collapse onto the
# declaration. The other two exist because forcing every metric into
# higher/lower would lie about a third of the registry.
DIRECTION_PHRASES: dict[str, str] = {
    "higher_is_better": "higher is better",
    "lower_is_better": "lower is better",
    "range_optimal": "a middle range is best; both ends are worse",
    "not_directional": "a named grade, with no better or worse direction",
}

# The bar R18 measures against, **read off the scorer rather than restated**.
# `low_coverage_threshold` is already a constructor default there and already
# applied to the composite; writing `0.85` here again would be a second
# statement of one number, free to drift, and the drift would be silent — the
# composite would flag thin evidence at one bar while the section prose said
# nothing at another. A caller that constructed the scorer with a different
# threshold passes it explicitly.
LOW_COVERAGE_THRESHOLD: float = SQGLPScorer({}, {}).low_coverage_threshold

COVERAGE_ADEQUATE = "adequate"
COVERAGE_LOW = "low"
COVERAGE_UNKNOWN = "unknown"

# Why applicability reads indeterminate when nobody asked. A caller that did
# not consult the table has not been told anything, and "nobody looked" must
# not become the same value on the page as "we looked and it fits".
NOT_CONSULTED_REASON = (
    "Sector applicability was not consulted for this reading, so whether this "
    "metric measures anything for this kind of company is unknown"
)


def _is_number(value) -> bool:
    """A real number the band walk may compare against a threshold.

    `numbers.Real` rather than `(int, float)` because numpy scalars reach here
    from pandas-backed metrics and `numpy.int64` is not an `int`. Bools are
    excluded explicitly: `True >= 0` is True, so a bool would land quietly in
    whichever band happens to sit lowest — a reading, on a value that is not a
    quantity at all.
    """
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


# ── Quantity: a number that cannot exist without its unit and direction ───


@dataclass(frozen=True)
class Quantity:
    """R12, as a constructor invariant.

    There is no way to build one of these without a unit and a direction from
    the closed vocabularies, and `str()` on it always renders the unit beside
    the figure. That is the whole design: a caller can only get a number out of
    this layer by holding an object that already carries what R12 requires, so
    rendering a bare numeral takes deliberate effort rather than being what
    happens if you forget.
    """

    value: float | int | str
    unit: str
    direction: str
    display_format: str = ""

    def __post_init__(self):
        if self.unit not in PRESENTATION_UNITS:
            raise ValueError(
                f"unit {self.unit!r} is not one of {sorted(PRESENTATION_UNITS)} — "
                f"a number with no declared dimension cannot be rendered (R12)"
            )
        if self.direction not in PRESENTATION_DIRECTIONS:
            raise ValueError(
                f"direction {self.direction!r} is not one of "
                f"{sorted(PRESENTATION_DIRECTIONS)} — a number with no declared "
                f"direction of goodness cannot be rendered (R12)"
            )

    @property
    def unit_phrase(self) -> str:
        return UNIT_PHRASES[self.unit]

    @property
    def direction_phrase(self) -> str:
        return DIRECTION_PHRASES[self.direction]

    @property
    def is_numeric(self) -> bool:
        return _is_number(self.value)

    @property
    def text(self) -> str:
        """The figure with its unit. The only rendering this layer offers."""
        if self.display_format:
            try:
                return self.display_format.format(self.value)
            except (ValueError, TypeError, KeyError, IndexError):
                # A format string that cannot take this value is a declaration
                # bug, but losing the number over it would be worse than
                # rendering it plainly — and the fallback still carries a unit.
                pass
        return self._fallback_text()

    def _fallback_text(self) -> str:
        prefix, suffix = UNIT_AFFIXES[self.unit]
        if not self.is_numeric:
            # A named grade. Rendered verbatim on purpose: humanising it here
            # would be the auto-humanising fallback the problem frame names as
            # a defect, and no map of grade labels exists yet (U9's job).
            return f"{prefix}{self.value}{suffix}"
        figure = f"{prefix}{self.value:g}{suffix}"
        if suffix or prefix:
            return figure
        # `count` has no affix, so without this the fallback would emit "4".
        return f"{figure} ({self.unit_phrase})"

    def __str__(self) -> str:
        return self.text

    def __format__(self, spec: str) -> str:
        return format(self.text, spec)


# ── Applicability: does this metric mean anything for this company? ───────


@dataclass(frozen=True)
class Applicability:
    """`SectorApplicability.evaluate`'s three-valued answer, carried on a reading.

    Kept as a field rather than folded into the status because it answers a
    different question. The status says whether a reading was produced; this
    says whether one would have meant anything. Both can be unknown, for
    unrelated reasons, and a reader wants to know which.
    """

    verdict: str
    reason: str
    matched_sectors: tuple[str, ...] = ()

    @property
    def known(self) -> bool:
        """Whether anyone has actually answered. Indeterminate is not an answer."""
        return self.verdict in (APPLIES, DOES_NOT_APPLY)

    @property
    def excluded(self) -> bool:
        return self.verdict == DOES_NOT_APPLY

    @classmethod
    def not_consulted(cls) -> Applicability:
        return cls(verdict=INDETERMINATE, reason=NOT_CONSULTED_REASON)

    @classmethod
    def of(cls, outcome) -> Applicability:
        """Normalise whatever the caller passed into one of these.

        Accepts the raw dict `SectorApplicability.evaluate` returns, an already
        built `Applicability`, or `None` for "did not ask". `None` becomes
        indeterminate-with-a-reason rather than absent, per R4: an absent
        applicability field would let a surface render nothing where it should
        render "nobody has checked".
        """
        if outcome is None:
            return cls.not_consulted()
        if isinstance(outcome, cls):
            return outcome
        verdict = outcome.get("verdict", INDETERMINATE)
        return cls(
            verdict=verdict,
            reason=outcome.get("reason") or NOT_CONSULTED_REASON,
            matched_sectors=tuple(outcome.get("matched_sectors") or ()),
        )


# ── The reading itself ────────────────────────────────────────────────────


@dataclass(frozen=True)
class Reading:
    """One metric, read for one company.

    Note what is **not** here: a `value` field. The number is reachable only
    through `quantity`, which cannot exist without its unit and direction. A
    caller that wants the bare figure asks for `reading.quantity.value` and the
    reach is visible in the diff.
    """

    metric_id: str
    status: str = READ
    quantity: Quantity | None = None
    band: str = ""
    reason: str = ""
    meaning: str = ""
    source_error: str | None = None
    applicability: Applicability = field(default_factory=Applicability.not_consulted)

    def __post_init__(self):
        if self.status not in READING_STATUSES:
            raise ValueError(
                f"{self.metric_id}: status {self.status!r} is not one of "
                f"{sorted(READING_STATUSES)}"
            )
        if self.status == READ:
            if not self.band.strip():
                raise ValueError(
                    f"{self.metric_id}: a reading must name the band it resolved"
                )
            if self.quantity is None:
                raise ValueError(
                    f"{self.metric_id}: a reading must carry the quantity it read"
                )
        elif not self.reason.strip():
            # R4, structurally. Every absence carries the reason for it, so a
            # future branch inside this module cannot ship a silent unknown.
            raise ValueError(
                f"{self.metric_id}: status {self.status!r} must carry the reason "
                f"no reading was produced (R4)"
            )

    @property
    def known(self) -> bool:
        return self.status == READ

    @property
    def sentence(self) -> str:
        """The one line a surface renders. Never blank, never a bare number."""
        if self.known:
            return (
                f"{self.quantity.text} — {self.band} "
                f"({self.quantity.direction_phrase})"
            )
        head = STATUS_PREFIXES[self.status]
        if self.quantity is not None:
            return f"{self.quantity.text} — {head}: {self.reason}"
        return f"{head}: {self.reason}"


@dataclass(frozen=True)
class CoverageReading:
    """R18, for one element.

    The clause is empty when coverage is adequate, deliberately: R18 asks a
    below-the-bar section to say so, not every section to recite a number
    nobody needed. Unknown coverage is *not* adequate coverage, so it gets a
    clause of its own — an element with no denominator reads like a fully
    measured one otherwise, which is the exact confusion R18 exists to end.
    """

    element: str
    share: float | None
    threshold: float
    status: str
    clause: str = ""
    reason: str = ""

    @property
    def low(self) -> bool:
        return self.status == COVERAGE_LOW

    @property
    def known(self) -> bool:
        return self.status in (COVERAGE_ADEQUATE, COVERAGE_LOW)


# ── The band walk ─────────────────────────────────────────────────────────


def resolve_band(bands: Iterable, low_label: str | None, value) -> str | None:
    """First threshold reached wins; `low_label` catches the remainder.

    `report_generator._forward_band`'s semantics unchanged — the four
    zero-weight signals have been read this way since Phase 2 and the
    validator's docstring states the rule once for every declaration. The
    difference is the return: `None` for "could not place this", where
    `_forward_band` returns `""`. An empty label is a reading; the absence of
    one is not, and the caller has to be able to tell them apart to obey R4.

    Descending order is a validator invariant, not an assumption made here — a
    list authored ascending has its first entry swallow every value, and every
    band beneath it becomes unreachable on every company.
    """
    if not _is_number(value):
        return None
    for band in bands or ():
        try:
            threshold, label = band
        except (TypeError, ValueError):
            continue
        if value >= threshold:
            return label
    return low_label or None


# ── Entry points ──────────────────────────────────────────────────────────


def read_value(
    metric_id: str,
    presentation: Mapping | None,
    value,
    *,
    display_format: str = "",
    error: str | None = None,
    applicability=None,
) -> Reading:
    """The pure core: a declaration plus a value in, a reading out.

    Takes the pieces rather than a registry entry so a test — or any caller
    holding a declaration from somewhere other than `elements/*.yaml` — can
    reach it without assembling a whole metric config. `read_metric` is the
    adapter over this for the ordinary case.

    **The resolution order is load-bearing.** Declaration first, because with
    no unit and no direction there is nothing R12 permits rendering at all.
    Applicability second, matching F1's "sector mismatch is decided first": if
    the metric measures nothing for this kind of company, *why* it also failed
    to compute is a detail the reader does not need, and the table's sentence
    is the more useful fact either way. Then the computation's own outcome, and
    only then the bands.
    """
    applies = Applicability.of(applicability)
    meaning = ""

    if not isinstance(presentation, Mapping) or not presentation:
        return Reading(
            metric_id=metric_id,
            status=NO_DECLARATION,
            reason=(
                "nothing declares how to read this metric — it carries no "
                "unit, no direction of goodness and no interpretation bands"
            ),
            source_error=error,
            applicability=applies,
        )

    meaning = str(presentation.get("meaning") or "")
    unit = presentation.get("unit")
    direction = presentation.get("direction")
    if unit not in PRESENTATION_UNITS or direction not in PRESENTATION_DIRECTIONS:
        return Reading(
            metric_id=metric_id,
            status=NO_DECLARATION,
            reason=(
                f"this metric's declaration names no usable unit or direction "
                f"(unit {unit!r}, direction {direction!r}), so its value cannot "
                f"be rendered with what R12 requires beside it"
            ),
            meaning=meaning,
            source_error=error,
            applicability=applies,
        )

    quantity = (
        None if value is None
        else Quantity(
            value=value, unit=unit, direction=direction, display_format=display_format
        )
    )

    if applies.excluded:
        # The number is kept: R7 puts the reason in front of the reader, and
        # AE1's section shows the row rather than hiding it. What is withheld
        # is the *band* — calling a lender "asset-heavy" for lending is the
        # misreading this whole path exists to stop.
        return Reading(
            metric_id=metric_id,
            status=NOT_APPLICABLE,
            quantity=quantity,
            reason=applies.reason,
            meaning=meaning,
            source_error=error,
            applicability=applies,
        )

    if error:
        return Reading(
            metric_id=metric_id,
            status=METRIC_ERROR,
            quantity=quantity,
            reason=f"the metric reported: {error}",
            meaning=meaning,
            source_error=error,
            applicability=applies,
        )

    if quantity is None:
        return Reading(
            metric_id=metric_id,
            status=VALUE_ABSENT,
            reason=(
                "the metric ran and produced no value, which is not the same "
                "as a value of zero"
            ),
            meaning=meaning,
            applicability=applies,
        )

    bands = presentation.get("bands") or []
    if not bands:
        # A first-class declaration, not an oversight. Nine shipped metrics are
        # here on purpose and the validator makes the reason mandatory,
        # because a wrong band is worse than a declared unknown.
        return Reading(
            metric_id=metric_id,
            status=BANDS_NOT_DECLARED,
            quantity=quantity,
            reason=str(
                presentation.get("bands_absent_reason")
                or "this metric declares no interpretation bands, and no reason "
                   "why not"
            ),
            meaning=meaning,
            applicability=applies,
        )

    if not quantity.is_numeric:
        return Reading(
            metric_id=metric_id,
            status=VALUE_NOT_BANDABLE,
            quantity=quantity,
            reason=(
                f"its declared bands are numeric and this value is "
                f"{type(value).__name__}, so no band can place it"
            ),
            meaning=meaning,
            applicability=applies,
        )

    band = resolve_band(bands, presentation.get("low_label"), value)
    if not band:
        # Bands with no `low_label`: a reading for high values and none for low
        # ones. The validator rejects this at startup, so reaching here means a
        # declaration got past it — and the value must not silently adopt the
        # lowest declared band, which would read as a finding on a company
        # nobody had written a finding for.
        return Reading(
            metric_id=metric_id,
            status=NO_DECLARATION,
            quantity=quantity,
            reason=(
                "this value falls below every declared band and the "
                "declaration names no reading for that"
            ),
            meaning=meaning,
            applicability=applies,
        )

    return Reading(
        metric_id=metric_id,
        status=READ,
        quantity=quantity,
        band=band,
        meaning=meaning,
        applicability=applies,
    )


def read_metric(metric_id: str, config: Mapping | None, result=None, *,
                applicability=None) -> Reading:
    """A registry entry plus its `MetricResult`, read for one company.

    `config` is the whole metric definition — `presentation` for the reading
    and `display.format` for the typography, which stays where it is because
    unit is the dimension and format is how it prints, and two statements of
    one fact drift invisibly.

    `result` of `None` means the engine never reached this metric, which is a
    different fact from a metric that ran and found nothing: one is a pipeline
    gap and the other is a data gap, and they carry different reasons.
    """
    config = config if isinstance(config, Mapping) else {}
    presentation = config.get("presentation")
    display_format = str((config.get("display") or {}).get("format") or "")

    if result is None:
        reading = read_value(
            metric_id, presentation, None,
            display_format=display_format, applicability=applicability,
        )
        if reading.status != VALUE_ABSENT:
            return reading
        return Reading(
            metric_id=metric_id,
            status=VALUE_ABSENT,
            reason="this metric was not computed for this company",
            meaning=reading.meaning,
            applicability=reading.applicability,
        )

    return read_value(
        metric_id,
        presentation,
        getattr(result, "value", None),
        display_format=display_format,
        error=getattr(result, "error", None),
        applicability=applicability,
    )


def read_metrics(configs: Mapping, results: Mapping, *, sector: str | None = None,
                 applicability=None) -> dict[str, Reading]:
    """Every metric on one company, declared or computed or both.

    The union of the two keysets, not the intersection: a computed metric with
    no declaration reads unknown-with-reason rather than vanishing, which is
    the behaviour the current drill-down gets wrong — it silently drops a
    metric it has no display name for, and a dropped row is invisible in a way
    a stated absence is not.

    `applicability` is anything with an `evaluate(metric_id, sector)` — a
    `SectorApplicability` in practice. Absent, every reading reports
    applicability as indeterminate-because-nobody-asked, never as applying.
    """
    metric_ids = sorted(set(configs) | set(results))
    readings: dict[str, Reading] = {}
    for metric_id in metric_ids:
        outcome = (
            applicability.evaluate(metric_id, sector)
            if applicability is not None else None
        )
        readings[metric_id] = read_metric(
            metric_id, configs.get(metric_id), results.get(metric_id),
            applicability=outcome,
        )
    return readings


def read_element_coverage(element: str, share, *,
                          threshold: float = LOW_COVERAGE_THRESHOLD
                          ) -> CoverageReading:
    """R18 for one element: how much of its declared weight actually scored.

    `share` is the figure `SQGLPScorer._coverage` already puts in
    `coverage["elements"][element]` — nothing new is computed and no score
    moves. It is `None` for an element with no declared weight, which is a
    missing denominator rather than a low numerator and reads as unknown.

    The clause names no element. `quality_business` is a raw key and R15 keeps
    those off the page, so the caller prepends the label from
    `report_vocabulary.ELEMENT_CONFIG` and this stays a pure statement about a
    share.
    """
    if not _is_number(share):
        return CoverageReading(
            element=element, share=None, threshold=threshold,
            status=COVERAGE_UNKNOWN,
            reason=(
                "no metric in this element carries declared weight, so there "
                "is nothing to measure the score's coverage against"
            ),
            clause=(
                "How much of this element's declared weight was scored is "
                "unknown: no metric in it carries declared weight."
            ),
        )

    if not 0.0 <= float(share) <= 1.0:
        return CoverageReading(
            element=element, share=float(share), threshold=threshold,
            status=COVERAGE_UNKNOWN,
            reason=(
                f"the reported coverage share {float(share):.3f} is not a "
                f"proportion between 0 and 1, so it cannot be read"
            ),
            clause=(
                f"How much of this element's declared weight was scored could "
                f"not be read: the reported share was {float(share):.3f}."
            ),
        )

    share = float(share)
    if share >= threshold:
        # Nothing to say. R18 asks a thin section to declare itself, not every
        # section to recite a number that changes no decision.
        return CoverageReading(
            element=element, share=share, threshold=threshold,
            status=COVERAGE_ADEQUATE,
        )

    return CoverageReading(
        element=element, share=share, threshold=threshold, status=COVERAGE_LOW,
        clause=(
            f"Scored on {share:.0%} of this element's declared weight, below "
            f"the {threshold:.0%} bar — the rest could not be computed, so "
            f"this score rests on part of the evidence."
        ),
    )


def read_element_coverages(coverage: Mapping | None, *,
                           threshold: float = LOW_COVERAGE_THRESHOLD
                           ) -> dict[str, CoverageReading]:
    """Every element's coverage clause, straight off the scorer's own output.

    Takes the whole `coverage` block the scorer returns and reads its
    `elements` mapping, so a caller passes what it already has rather than
    reaching into it and getting the shape wrong somewhere else.
    """
    elements = (coverage or {}).get("elements") or {}
    return {
        element: read_element_coverage(element, share, threshold=threshold)
        for element, share in elements.items()
    }
