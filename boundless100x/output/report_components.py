"""The six things the new report is allowed to say, and nothing else (R13–R15).

Every part of the new report is one of six components — a finding, a metric
row, a reading, a disclosure, an unknown-with-reason, or a caveat — and each is
**data a surface renders, not markup a surface parses**. That is what makes R14
achievable at all: HTML, Markdown and the console are three renderings of one
declaration, and the only way three surfaces stay in agreement is if none of
them is holding a string the other two cannot understand.

Which is also why the closed set has to be closed. A seventh member added in
passing is a member two of the three surfaces do not render, and the failure is
silent — a section that simply is not there. `COMPONENT_KINDS` is the
enumeration, `component_surface` is the decorator that refuses a renderer
missing one of them at *import* time, and `EXPECTED_SURFACES` names the three
R14 requires. U10 and U11 are the surfaces; this module is their contract.

── R15 as an invariant, not a convention ─────────────────────────────────

The problem frame lists five paths that route a raw identifier onto the page:
an unregistered flag falls through to `f.replace("_", " ").title()`, an
unregistered metric is dropped from the drill-down entirely, the CLI keeps its
own element labels, an action badge renders snake-derived text, and
`service.py`'s `f"...failed: {e}"` warnings reach the reader verbatim. Four of
those are *omissions*, and an omission cannot be fixed by remembering harder.

So no component can be constructed carrying text that fails `guard_text`. The
guard refuses three shapes:

    a raw identifier   any snake_case token anywhere in the string, plus a
                       whole field whose entire content is a key from one of
                       the closed vocabularies this system already declares
    markup             tags, entities, pipes, brackets, backticks, bold runs,
                       leading block markers, ANSI, and any control character
    an exception       tracebacks, `File "...", line N`, a CamelCase name
                       ending in Error or Exception, a dotted module path, an
                       absolute filesystem path

**Be clear about what the third one cannot do.** It detects *shape*, never
provenance. `str(exc)` on `Exception("Screener returned 404")` is ordinary
prose and no scanner will ever tell it from a sentence somebody wrote. That is
why `caveat_from_run_error` does not trust the guard to save it: it never
renders the untrusted half of a run error at all, keeping only the authored
clause in front of the colon — the part `service.py` wrote — and sending the
rest to the log. The guard is the second line of defence, for a caller that
builds a `Caveat` by hand.

The first rule has a matching limit worth stating rather than discovering. A
single-word key — `core`, `probe`, `exited`, `unknown`, `qualifies` — is
indistinguishable from prose in the middle of a sentence, so it is caught only
when it is the *whole* field. Those cases are prevented by routing, not by
detection: a lane is rendered through `LANE_LABELS`, a verdict through
`LANE_VERDICT_LABELS`, a grade through `CATEGORICAL_VALUE_LABELS`. The guard
catches the hand-built mistake; the vocabulary is what stops the ordinary path.

── Auto-humanising is the defect, so unknown is the answer ───────────────

`FLAG_LABELS.get(f, (None, None))` falling through to
`f.replace("_", " ").title()` is not a fallback, it is a leak with better
typography — "Cfi Dominated By Acquisitions" is the metric id wearing a hat.
`finding_from_flag` returns an `Unknown` for a flag nobody registered: the
reader is told a signal fired that the report has no words for, which is true,
short, and actionable by the one person who can add the label. The flag id
stays on the component as a handle so a surface can log it.

── The disclosure is deferred by construction (R3) ───────────────────────

R3 wants the explanation *reachable* and *never inline*. A component in the
reading flow therefore never holds one: a `MetricRow` carries a
`DisclosureRef`, which has an anchor and a title and no body at all, and the
bodies live in `Section.disclosures`, which `Section.flow` excludes. A renderer
walking the flow cannot reach the explanation text, because it is not there.
That is the guarantee — not that a renderer is forbidden from calling
`render_disclosure`, which no type system here can prevent, but that the flow
does not contain the words.

── The vocabulary boundary ───────────────────────────────────────────────

`Vocabulary.narrate` is the one place declaration prose is turned into reader
prose. It strips backtick delimiters and replaces any metric id it recognises
with that metric's registered name — which is lookup, not humanising: the
difference is that a lookup fails loudly for an id nobody registered, and
`.title()` succeeds on everything. Six shipped declarations name another metric
inside backticks and read correctly once substituted; two more name Python
parameters and a function, and those are refused (see the note on
`narrate`). Nothing here computes, scores or formats a number — `Quantity`
already did that, and a second statement of it would drift.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import ClassVar

from boundless100x.compute_engine.eligibility import DEFAULT_GATES
from boundless100x.compute_engine.metrics.validator import (
    PRESENTATION_DIRECTIONS,
    PRESENTATION_UNITS,
)
from boundless100x.lifecycle.lane_gates import LANE_VERDICTS
from boundless100x.lifecycle.states import LANES, STATES
from boundless100x.output.report_expansion import (
    TRIGGER_LABELS,
    ExpansionReason,
    MetricDecision,
    SectionDecision,
)
from boundless100x.output.report_reading import (
    READING_STATUSES,
    CoverageReading,
    Reading,
    is_number,
)
from boundless100x.output.report_vocabulary import (
    CATEGORICAL_VALUE_LABELS,
    COMPOSITE_READING,
    COMPOSITE_TITLE,
    COMPOSITE_UNKNOWN_REASON,
    ELEMENT_CONFIG,
    FLAG_ELEMENT_MAP,
    FLAG_LABELS,
    METRIC_DISPLAY_NAMES,
    SCORE_BANDS,
    SCORE_LOW_LABEL,
    SCORE_SCALE,
)

logger = logging.getLogger(__name__)


# ── The closed set ────────────────────────────────────────────────────────
#
# Six members. Adding a seventh is an edit here, an edit to the dataclass list
# below, and a new `render_*` on all three surfaces — the decorator makes the
# last one mandatory rather than hoped for.

FINDING = "finding"
METRIC_ROW = "metric_row"
READING = "reading"
DISCLOSURE = "disclosure"
UNKNOWN = "unknown"
CAVEAT = "caveat"

COMPONENT_KINDS: tuple[str, ...] = (
    FINDING, METRIC_ROW, READING, DISCLOSURE, UNKNOWN, CAVEAT,
)

# R14's three renderings. Named here rather than in each surface so a surface
# registering under a fourth name is a test failure rather than a section that
# quietly exists in one place.
EXPECTED_SURFACES: tuple[str, ...] = ("html", "markdown", "console")

# How a finding leans. The same three words the existing flag badges use, so a
# surface that already styles `good`/`bad`/`neutral` needs no second stylesheet.
GOOD, BAD, NEUTRAL = "good", "bad", "neutral"
SENTIMENTS = frozenset({GOOD, BAD, NEUTRAL})

# A caveat is either something to keep in mind or something that cost the run
# part of its evidence. Two levels, because a third would need a rule for
# choosing between them that nobody has written.
NOTE, WARNING = "note", "warning"
SEVERITIES = frozenset({NOTE, WARNING})


class ComponentContentError(ValueError):
    """Text that must not reach a reader was passed to a component."""


class IncompleteSurface(TypeError):
    """A renderer does not handle every member of the closed set (R14)."""


# ── The guard ─────────────────────────────────────────────────────────────

# A whole snake_case token: two or more lowercase segments joined by
# underscores. This is the workhorse — every metric id, flag, state, lane
# verdict and category value in this system has this shape, and ordinary
# English prose never does. It catches ids nobody has registered anywhere,
# which is the half a lookup table can never cover.
_SNAKE_TOKEN = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")

# The evaluators write their conditions with the comparator key they were
# declared with — "Market Cap 5000.00 lte 3000" — and `lte` is as much a raw
# identifier to a reader as the metric id beside it was. It survives
# `_SNAKE_TOKEN` because it has no underscore, which is why it outlived the
# first pass at this. Symbols rather than words ("is at most") because the
# reason is already phrased as a condition that was or was not met, and
# inserting a verb into someone else's sentence reads worse than the maths.
# Keys mirror `eligibility.COMPARATORS`; a test pins the two sets equal, so a
# fifth comparator cannot ship without wording.
COMPARATOR_SYMBOLS: dict[str, str] = {
    "lt": "<",
    "lte": "≤",
    "gt": ">",
    "gte": "≥",
}
_COMPARATOR_TOKEN = re.compile(
    r"\b(?:%s)\b" % "|".join(sorted(COMPARATOR_SYMBOLS, key=len, reverse=True))
)

# Markup, by surface. `<` and `>` are *not* banned on their own: shipped labels
# read "Exceptional RoCE (>25%)" and "RoCE > 15% Count (10yr)", and a rule that
# rejected those would be a rule the vocabulary has to be bent around.
_MARKUP_SHAPES: tuple[tuple[str, re.Pattern], ...] = (
    ("an HTML tag", re.compile(r"<\s*/?[A-Za-z][^>]*>")),
    ("an HTML entity", re.compile(r"&(?:[A-Za-z]+|#\d+);")),
    ("a table pipe", re.compile(r"\|")),
    ("a bracket, which Rich reads as console markup", re.compile(r"[\[\]]")),
    ("a backtick, which only Markdown understands", re.compile(r"`")),
    ("a bold or underline run", re.compile(r"\*\*|__")),
    ("an ANSI escape", re.compile(r"\x1b")),
    ("a line break or control character",
     re.compile(r"[\n\r\t\x00-\x08\x0b\x0c\x0e-\x1f]")),
    ("a leading Markdown block marker",
     re.compile(r"^\s*(?:#{1,6}\s|[-*+]\s|>\s|\d+[.)]\s)")),
)

# What a leaked exception looks like. Shape only — see the module docstring.
_EXCEPTION_SHAPES: tuple[tuple[str, re.Pattern], ...] = (
    ("a traceback", re.compile(r"Traceback \(most recent call last\)")),
    ("a traceback frame", re.compile(r'File "[^"]+", line \d+')),
    ("an exception class name",
     re.compile(r"\b[A-Za-z_][\w.]*(?:Error|Exception)\b")),
    ("an errno", re.compile(r"Errno\s+\d+")),
    # `str(KeyError("borrowings"))` is `"'borrowings'"` — a quoted bare key and
    # nothing else. PFC's Debt/Equity row reaches the reader as "the metric
    # reported: 'borrowings'" today, and the snake_case rule cannot see it
    # because the key is one word. Narrow on purpose: only a lone quoted
    # identifier closing the string, which is the position a re-raised KeyError
    # actually lands in.
    ("a bare quoted key, the shape a KeyError takes",
     re.compile(r"(?:^|:\s)'[A-Za-z_]\w*'\s*$")),
    ("a dotted module path",
     re.compile(r"\b[a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*){2,}\b")),
    ("an absolute filesystem path",
     re.compile(r"(?:^|[\s(])/(?:[\w.\-]+/)+[\w.\-]*")),
)

# A number with nothing beside it. R12's failure, and KD6's own example: a
# uniform row showing `0.84` is consistent and still unreadable.
_BARE_NUMERAL = re.compile(r"^[-+]?[\d][\d,.\s]*$")


def _reserved_keys() -> frozenset[str]:
    """Every raw key this system already declares somewhere.

    Imported rather than retyped, for `LANE_VERDICT_LABELS`'s reason: a rename
    that missed a copy here would leave the guard passing exactly the key the
    rename made stale. Single-word members are only ever checked against a
    whole field — `probe` and `core` are English words, and a guard that
    refused them mid-sentence would refuse most sentences worth writing.
    """
    keys: set[str] = set()
    keys.update(STATES)
    keys.update(LANES)
    keys.update(LANE_VERDICTS)
    keys.update(READING_STATUSES)
    keys.update(ELEMENT_CONFIG)
    keys.update(FLAG_LABELS)
    keys.update(FLAG_ELEMENT_MAP)
    keys.update(METRIC_DISPLAY_NAMES)
    keys.update(TRIGGER_LABELS)
    keys.update(PRESENTATION_UNITS)
    keys.update(PRESENTATION_DIRECTIONS)
    keys.update(DEFAULT_GATES)
    keys.update(("eligible", "not_eligible", "indeterminate"))
    # The action vocabulary, spelled here rather than imported: `action_policy`
    # imports the report layer's siblings and the one thing this module must
    # not acquire is a cycle. The list is five words and a test pins it.
    keys.update(("avoid", "watchlist", "hold", "buy", "strong_buy"))
    for values in CATEGORICAL_VALUE_LABELS.values():
        # **Multi-segment grades only.** A grade's value belongs to its own
        # metric rather than to the report's global vocabulary, and four of
        # them are single English words: `moderate`, `risky`, `unknown`,
        # `discounting`. Three shipped `presentation.bands` declarations —
        # `dupont_turnover`, `revenue_growth_streak`, `reverse_dcf_growth` —
        # read "moderate", which is a band label somebody wrote and not an enum
        # that leaked, and the whole-field rule refused every one of them: U10
        # could not render a real company at all until this narrowed.
        #
        # Nothing is given up. Every multi-segment grade is still refused
        # *anywhere in a string* by the snake-case rule, which is the stronger
        # of the two checks, and the ordinary path for a grade is
        # `CATEGORICAL_VALUE_LABELS` routing rather than this scan — the module
        # docstring already says single-word keys are covered by routing and
        # not by detection. This is that sentence applied to the one family of
        # keys where a false positive costs a reading.
        keys.update(value for value in values if "_" in value)
    return frozenset(keys)


RESERVED_KEYS = _reserved_keys()


def guard_text(text, *, field: str, subject: str = "", required: bool = True) -> str:
    """R15, R14 and R4 as one check, run from every component's constructor.

    Returns the text unchanged when it is safe; raises `ComponentContentError`
    naming the field, the shape and the offending fragment when it is not. The
    error is for a developer and says so — it is never rendered, which is the
    whole point of raising rather than substituting.
    """
    where = f"{subject}: {field}" if subject else field
    value = "" if text is None else str(text)

    if not value.strip():
        if required:
            raise ComponentContentError(
                f"{where} is blank — a component with nothing in it renders as "
                f"the absent line R4 forbids"
            )
        return value

    for name, pattern in _MARKUP_SHAPES:
        match = pattern.search(value)
        if match:
            raise ComponentContentError(
                f"{where} contains {name} ({match.group(0)!r}) — a component "
                f"carries data, and three surfaces render it three ways (R14)"
            )

    for name, pattern in _EXCEPTION_SHAPES:
        match = pattern.search(value)
        if match:
            raise ComponentContentError(
                f"{where} looks like {name} ({match.group(0).strip()!r}) — "
                f"raw exception text must not reach a reader (R15)"
            )

    match = _SNAKE_TOKEN.search(value)
    if match:
        raise ComponentContentError(
            f"{where} contains the raw identifier {match.group(0)!r} — every "
            f"id, flag and enum reaches a reader through a registered label "
            f"or not at all (R15)"
        )

    if value.strip() in RESERVED_KEYS:
        raise ComponentContentError(
            f"{where} is the raw key {value.strip()!r} rather than its label "
            f"(R15)"
        )

    return value


def guard_quantity(text, *, field: str, subject: str = "") -> str:
    """R12 on a rendered figure: never a number standing on its own.

    `Quantity.text` already carries the unit in every shipped declaration, so
    this fires only on a component built by hand or on a `display.format`
    somebody stripped. `metric_row` pre-empts it by appending the unit phrase,
    which is why this can afford to raise rather than paper over.
    """
    value = guard_text(text, field=field, subject=subject, required=False)
    if value.strip() and _BARE_NUMERAL.match(value.strip()):
        raise ComponentContentError(
            f"{(subject + ': ') if subject else ''}{field} is the bare figure "
            f"{value.strip()!r} — no number reaches a reader without its unit "
            f"and its direction (R12)"
        )
    return value


# ── The components ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Component:
    """Shared machinery: the kind, and which fields a reader can see.

    `TEXT_FIELDS` is guarded; `HANDLE_FIELDS` is not, and is for values a
    surface uses to find something again — an anchor, an id for a log line.
    A test asserts the two together cover every declared field of every member,
    so a field added without a decision about which it is fails the suite
    rather than slipping onto the page unchecked.
    """

    KIND: ClassVar[str] = ""
    TEXT_FIELDS: ClassVar[tuple[str, ...]] = ()
    HANDLE_FIELDS: ClassVar[tuple[str, ...]] = ()

    @property
    def kind(self) -> str:
        return self.KIND

    def _guard(self, *, required: tuple[str, ...] = ()) -> None:
        for name in self.TEXT_FIELDS:
            guard_text(
                getattr(self, name),
                field=name,
                subject=type(self).__name__,
                required=name in required,
            )


@dataclass(frozen=True)
class Unknown(_Component):
    """R4's answer: something could not be said, and here is why.

    Both fields are mandatory. An unknown with no reason is the blank R4
    forbids with an extra word in front of it, and it is the shape every
    auto-humanising fallback degrades into once the label is taken away.
    """

    subject: str
    reason: str

    KIND: ClassVar[str] = UNKNOWN
    TEXT_FIELDS: ClassVar[tuple[str, ...]] = ("subject", "reason")
    HANDLE_FIELDS: ClassVar[tuple[str, ...]] = ()

    def __post_init__(self):
        self._guard(required=("subject", "reason"))

    @property
    def sentence(self) -> str:
        return f"{self.subject}: {self.reason}"


@dataclass(frozen=True)
class Finding(_Component):
    """Something the report found and is telling the reader about.

    Two sources today: a fired expansion trigger, whose `text` is R7's reason
    in the reader's words, and a metric flag, which has a headline and nothing
    else to say. `headline` is mandatory and `text` is not, because a finding
    with no headline is a paragraph nobody can skim past and a finding with no
    body is a badge, which is a legitimate thing to be.
    """

    headline: str
    text: str = ""
    subject: str = ""
    sentiment: str = NEUTRAL
    source: str = ""

    KIND: ClassVar[str] = FINDING
    TEXT_FIELDS: ClassVar[tuple[str, ...]] = ("headline", "text", "subject")
    HANDLE_FIELDS: ClassVar[tuple[str, ...]] = ("sentiment", "source")

    def __post_init__(self):
        self._guard(required=("headline",))
        if self.sentiment not in SENTIMENTS:
            raise ComponentContentError(
                f"Finding: sentiment {self.sentiment!r} is not one of "
                f"{sorted(SENTIMENTS)}"
            )


@dataclass(frozen=True)
class DisclosureRef(_Component):
    """A pointer to an explanation, carrying none of it.

    This is what makes R3's second half structural. A row in the reading flow
    holds one of these — a title it already displays and an anchor a surface
    can link, footnote or key a `--explain` lookup on — and the explanation
    itself is somewhere else entirely. There is no route from the flow to the
    words.

    It shares the disclosure's kind because it is the same member of the set
    seen from the flow side; a surface renders a ref inline and the bodies in
    its deferred section, both through `render_disclosure`.
    """

    title: str
    anchor: str

    KIND: ClassVar[str] = DISCLOSURE
    TEXT_FIELDS: ClassVar[tuple[str, ...]] = ("title",)
    HANDLE_FIELDS: ClassVar[tuple[str, ...]] = ("anchor",)

    def __post_init__(self):
        self._guard(required=("title",))
        if not str(self.anchor).strip():
            raise ComponentContentError(
                "DisclosureRef: anchor is blank — a reference nothing can be "
                "reached through is not reachable (R3)"
            )


@dataclass(frozen=True)
class Disclosure(_Component):
    """R3's explanation of what a metric measures and what good looks like.

    `DEFERRED` is a statement about placement, and `Section.flow` enforces it:
    these never appear among the components a surface walks in order. A body
    that is blank raises, because R3 asks for the explanation to be reachable
    and an empty one is not reachable, only present.
    """

    title: str
    body: str
    anchor: str

    DEFERRED: ClassVar[bool] = True
    KIND: ClassVar[str] = DISCLOSURE
    TEXT_FIELDS: ClassVar[tuple[str, ...]] = ("title", "body")
    HANDLE_FIELDS: ClassVar[tuple[str, ...]] = ("anchor",)

    def __post_init__(self):
        self._guard(required=("title", "body"))
        if not str(self.anchor).strip():
            raise ComponentContentError(
                "Disclosure: anchor is blank — nothing in the reading flow "
                "could point at this explanation (R3)"
            )

    @property
    def ref(self) -> DisclosureRef:
        return DisclosureRef(title=self.title, anchor=self.anchor)


@dataclass(frozen=True)
class MetricRow(_Component):
    """One metric, as a row: what it is, what it read, and how to find out more.

    `reading` and `unknown` are exclusive and one of them is mandatory. That
    pair is R4 written as a type — there is no way to build a row whose reading
    cell is empty, and no way to build one whose absence has no reason. AE4 is
    the case that matters: a metric with no declared bands still shows its
    value with its unit, and the reason no band placed it.

    `direction` is R12's other half and is a field rather than something a
    surface looks up. The unit travels inside `value` because `Quantity.text`
    puts it there; the direction of goodness has nowhere else to be, and a row
    reading "0.09x — poor" tells a reader nothing about whether they wanted the
    number higher. It is `Quantity.direction_phrase` verbatim, including the
    honest one a named grade gets.
    """

    label: str
    value: str = ""
    reading: str = ""
    unknown: Unknown | None = None
    direction: str = ""
    score: str = ""
    weight: str = ""
    disclosure: DisclosureRef | None = None
    metric_id: str = ""

    KIND: ClassVar[str] = METRIC_ROW
    TEXT_FIELDS: ClassVar[tuple[str, ...]] = (
        "label", "reading", "direction", "score", "weight",
    )
    HANDLE_FIELDS: ClassVar[tuple[str, ...]] = (
        "value", "unknown", "disclosure", "metric_id",
    )

    def __post_init__(self):
        self._guard(required=("label",))
        # `value` is guarded separately: it is reader-facing like the rest, and
        # additionally must never be a naked figure.
        guard_quantity(self.value, field="value", subject=f"MetricRow {self.label}")
        if bool(self.reading.strip()) == (self.unknown is not None):
            raise ComponentContentError(
                f"MetricRow {self.label}: a row carries either a reading or an "
                f"unknown-with-reason, never both and never neither (R4)"
            )

    @property
    def known(self) -> bool:
        return self.unknown is None


@dataclass(frozen=True)
class ReadingLine(_Component):
    """The one line a collapsed section gets (R5), and its coverage clause (R18).

    Same exclusive pair as a row, for the same reason: a section that could not
    be scored says so with its reason rather than opening on a dash.
    `qualifier` is R18's clause and is empty when coverage is adequate, which
    is deliberate — R18 asks a thin section to declare itself, not every
    section to recite a number that changed nothing.
    """

    subject: str
    text: str = ""
    unknown: Unknown | None = None
    headline: str = ""
    qualifier: str = ""
    key: str = ""

    KIND: ClassVar[str] = READING
    TEXT_FIELDS: ClassVar[tuple[str, ...]] = ("subject", "text", "qualifier")
    HANDLE_FIELDS: ClassVar[tuple[str, ...]] = ("unknown", "headline", "key")

    def __post_init__(self):
        self._guard(required=("subject",))
        guard_quantity(self.headline, field="headline", subject="ReadingLine")
        if bool(self.text.strip()) == (self.unknown is not None):
            raise ComponentContentError(
                f"ReadingLine {self.subject}: a reading carries either its line "
                f"or an unknown-with-reason, never both and never neither (R4)"
            )

    @property
    def known(self) -> bool:
        return self.unknown is None


@dataclass(frozen=True)
class Caveat(_Component):
    """A caution that qualifies what the reader has just been told.

    The constructor is where `service.py`'s warnings stop. Exception-shaped
    text raises here rather than being softened, because softening is what
    produced "Cfi Dominated By Acquisitions": the substitution succeeds, the
    leak survives, and nobody finds out. Callers holding untrusted text use
    `caveat_from_run_error`, which never renders the untrusted half at all.
    """

    text: str
    subject: str = ""
    severity: str = NOTE

    KIND: ClassVar[str] = CAVEAT
    TEXT_FIELDS: ClassVar[tuple[str, ...]] = ("text", "subject")
    HANDLE_FIELDS: ClassVar[tuple[str, ...]] = ("severity",)

    def __post_init__(self):
        self._guard(required=("text",))
        if self.severity not in SEVERITIES:
            raise ComponentContentError(
                f"Caveat: severity {self.severity!r} is not one of "
                f"{sorted(SEVERITIES)}"
            )


COMPONENT_TYPES: dict[str, tuple[type, ...]] = {
    FINDING: (Finding,),
    METRIC_ROW: (MetricRow,),
    READING: (ReadingLine,),
    DISCLOSURE: (Disclosure, DisclosureRef),
    UNKNOWN: (Unknown,),
    CAVEAT: (Caveat,),
}


# ── The arrangement ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class Section:
    """One section's components, arranged. Not a component itself.

    The distinction matters for R13: a section is *where* components go, and
    the closed set is *what* may go there. `flow` is the ordered walk a surface
    renders top to bottom and it contains no `Disclosure` — the explanations
    are reachable through `disclosures`, which is a separate, deferred surface
    (R3).
    """

    key: str
    title: str
    reading: ReadingLine
    findings: tuple[Finding, ...] = ()
    rows: tuple[MetricRow, ...] = ()
    unknowns: tuple[Unknown, ...] = ()
    caveats: tuple[Caveat, ...] = ()
    disclosures: tuple[Disclosure, ...] = ()
    expanded: bool = False

    def __post_init__(self):
        for name, allowed in (
            ("findings", Finding),
            ("rows", MetricRow),
            ("unknowns", Unknown),
            ("caveats", Caveat),
            ("disclosures", Disclosure),
        ):
            for item in getattr(self, name):
                if not isinstance(item, allowed):
                    raise ComponentContentError(
                        f"Section {self.key}: {name} holds a "
                        f"{type(item).__name__}, which is not a {allowed.__name__} "
                        f"— a section renders nothing outside the closed set (R13)"
                    )

    @property
    def flow(self) -> tuple[_Component, ...]:
        """What a surface renders in order. Explanations are not in it (R3)."""
        return (self.reading, *self.findings, *self.rows, *self.unknowns,
                *self.caveats)


# ── The vocabulary boundary ───────────────────────────────────────────────


_BACKTICKED = re.compile(r"`([^`]*)`")


class Vocabulary:
    """Declarations in, reader-facing words out. The only place labels are chosen.

    Constructed from the engine's `metrics` mapping, so the display name is the
    registry's own `name` — `contradiction.py` and `report_expansion.py` both
    made this choice already, and for the same reason: `METRIC_DISPLAY_NAMES`
    is hand-maintained and its omission path is the silent drop the problem
    frame names. That map is still consulted, second, for a caller holding no
    registry at all.
    """

    def __init__(self, metric_configs: Mapping | None = None):
        self.metric_configs = dict(metric_configs or {})
        # R3's explanations, cached per metric. `presentation.meaning` is a
        # property of the declaration and not of the company, so one entry
        # serves every ticker this instance is used for — and the log line for
        # a declaration that cannot be cleaned is printed once rather than once
        # per row and once per section.
        self._disclosures: dict[str, Disclosure | None] = {}
        self.names: dict[str, str] = {}
        for metric_id, config in self.metric_configs.items():
            name = str((config or {}).get("name") or "").strip()
            if name:
                self.names[metric_id] = name
        for metric_id, (_element, name) in METRIC_DISPLAY_NAMES.items():
            self.names.setdefault(metric_id, name)

    # ── Labels ────────────────────────────────────────────────────────────

    def metric_name(self, metric_id: str) -> str | None:
        """The registered name, or `None` — never the id and never a guess."""
        return self.names.get(metric_id)

    def element_title(self, element: str) -> str | None:
        config = ELEMENT_CONFIG.get(element)
        return str(config["label"]) if config else None

    def category(self, metric_id: str, value) -> tuple[str, str] | None:
        """`(label, gloss)` for a named grade, or `None` if nobody wrote one.

        `None` is the answer that matters. A grade added to a `categories:`
        table without a label here must read as unknown, not as
        "Founder Led High Holding" — and a test derives the expected key set
        from the registry so the gap is a failing suite rather than a leak.
        """
        entry = (CATEGORICAL_VALUE_LABELS.get(metric_id) or {}).get(str(value))
        return tuple(entry) if entry else None  # type: ignore[return-value]

    # ── Prose ─────────────────────────────────────────────────────────────

    def narrate(self, text) -> str:
        """Declaration prose, with the identifiers in it resolved to labels.

        Two passes: backtick delimiters come off (they are Markdown, and one of
        the three surfaces renders them as punctuation), then any token that is
        a metric id becomes that metric's registered name.

        Substituting a *registered* id is lookup, and lookup fails loudly for
        anything unregistered — which is the whole difference from
        `.title()`. Six shipped declarations name a sibling metric this way and
        read correctly afterwards. Two do not survive it and are not meant to:
        `price_lever_signal` and `quality_growth_quadrant` explain their
        thresholds by naming Python parameters and a function, and there is no
        reader-facing label for `strong_real_growth_pct`. Both are categorical
        metrics whose reading comes from `CATEGORICAL_VALUE_LABELS` instead, so
        the refused text is text nothing renders — but the refusal is real, and
        a caller that reaches for it gets `ComponentContentError` rather than
        the parameter name.
        """
        value = "" if text is None else str(text)
        value = _BACKTICKED.sub(lambda m: self._resolve(m.group(1)), value)
        value = _SNAKE_TOKEN.sub(lambda m: self._resolve(m.group(0)), value)
        return _COMPARATOR_TOKEN.sub(lambda m: COMPARATOR_SYMBOLS[m.group(0)], value)

    def _resolve(self, token: str) -> str:
        """A metric id, else a flag id, else the token untouched.

        Flags reach here because an evaluator names one in its prose — an
        eligibility veto says "absence of `reverse_dcf_overpriced` cannot be
        confirmed", and that id rendered raw on the dashboard beside the metric
        ids that *were* being resolved. A flag is exactly as much an identifier
        to a reader as a metric is, and `FLAG_LABELS` already holds its wording.

        An unresolved token is still returned as-is rather than humanised:
        `guard_text` refuses it downstream, which is the loud failure. Guessing
        a label from the id is the leak with better typography that this whole
        layer exists to refuse.
        """
        key = token.strip()
        if key in self.names:
            return self.names[key]
        label = FLAG_LABELS.get(key)
        return label[0] if label else token


# ── Builders ──────────────────────────────────────────────────────────────
#
# Every one of these is a total function: it always returns a component, and
# when it cannot produce the one it was asked for it produces an `Unknown`
# carrying the reason. Nothing here returns `None`, drops a row, or falls back
# to a derived label — those three are the problem frame's defects, in order.


def finding_from_reason(reason: ExpansionReason) -> Finding:
    """A fired expansion trigger, as the reader meets it (R7)."""
    return Finding(
        headline=TRIGGER_LABELS[reason.trigger],
        text=reason.text,
        subject=reason.metric_name,
        sentiment=BAD,
        source=reason.trigger,
    )


def finding_from_flag(flag: str) -> Finding | Unknown:
    """A metric flag, or an honest admission that nothing names it.

    The named defect lives here. `FLAG_LABELS.get(f, ...)` currently falls
    through to `f.replace("_", " ").title()`, which puts the id on the page in
    title case and looks deliberate. An unregistered flag is a gap in the
    vocabulary, and the reader is better served by being told a signal fired
    that this report has no words for — a sentence that is true, that costs one
    line, and that the person who can fix it will recognise.
    """
    entry = FLAG_LABELS.get(flag)
    if entry is None:
        logger.warning(
            f"Flag {flag!r} has no entry in FLAG_LABELS, so it renders as "
            f"unknown rather than as a label derived from its id (R15)"
        )
        return Unknown(
            subject="A signal this report has no wording for",
            reason=(
                "the analysis raised a signal that nothing in the report's "
                "vocabulary names, so it cannot be put into words a reader can "
                "act on — it is recorded in the run's data and in the log"
            ),
        )
    label, sentiment = entry
    return Finding(
        headline=label,
        sentiment=sentiment if sentiment in SENTIMENTS else NEUTRAL,
        source=flag,
    )


def disclosure_for(metric_id: str, reading: Reading, vocabulary: Vocabulary
                   ) -> Disclosure | None:
    """R3's explanation for one metric, or `None` when the metric declares none.

    `None` rather than an `Unknown`: a missing explanation is already stated by
    the row itself — a metric with no `presentation` block reads
    `no_declaration` and says so — and a second unknown in the deferred section
    would say it twice in a place nobody is looking.
    """
    if metric_id in vocabulary._disclosures:
        return vocabulary._disclosures[metric_id]

    built: Disclosure | None = None
    name = vocabulary.metric_name(metric_id)
    body = vocabulary.narrate(reading.meaning) if name else ""
    if name and str(body).strip():
        try:
            built = Disclosure(title=name, body=body, anchor=metric_id)
        except ComponentContentError as exc:
            # A declaration the vocabulary cannot clean. Dropped rather than
            # rendered, and logged loudly: the fix is a label or an edit to the
            # declaration, and neither happens if this fails quietly.
            logger.warning(f"Explanation for {metric_id} is not renderable: {exc}")

    vocabulary._disclosures[metric_id] = built
    return built


def _quantity_text(reading: Reading, vocabulary: Vocabulary) -> tuple[str, str | None]:
    """The figure as a reader sees it, plus a gloss when it is a named grade.

    Three cases. No quantity at all is `("", None)`. A named grade goes through
    `CATEGORICAL_VALUE_LABELS` — the raw value is never shown, per R15, and an
    unmapped grade returns `("", None)` so the caller can raise the unknown. A
    number is `Quantity.text`, which already carries its unit in every shipped
    declaration; the unit phrase is appended only if it somehow does not, which
    is R12 holding even when a `display.format` is wrong.
    """
    quantity = reading.quantity
    if quantity is None:
        return "", None

    if quantity.unit == "category":
        entry = vocabulary.category(reading.metric_id, quantity.value)
        if entry is None:
            return "", None
        label, gloss = entry
        return label, gloss

    text = quantity.text
    if _BARE_NUMERAL.match(text.strip()):
        text = f"{text} ({quantity.unit_phrase})"
    return text, None


def metric_row(
    metric_id: str,
    reading: Reading,
    vocabulary: Vocabulary,
    *,
    score=None,
    weight_share: float | None = None,
) -> MetricRow | Unknown:
    """One metric's row, or an unknown when the report has no name for it.

    An unregistered metric becomes a visible unknown rather than a dropped
    row — the drill-down's current behaviour is to skip it silently, and a row
    that is not there is invisible in a way a stated absence is not.
    """
    name = vocabulary.metric_name(metric_id)
    if not name:
        logger.warning(
            f"Metric {metric_id!r} has no registered display name, so its row "
            f"renders as unknown rather than being dropped (R15)"
        )
        return Unknown(
            subject="A metric this report has no name for",
            reason=(
                "the analysis computed something the report's vocabulary does "
                "not name, so it cannot be labelled — it is shown here rather "
                "than dropped, because a row nobody can see is worse than one "
                "nobody can name"
            ),
        )

    value, gloss = _quantity_text(reading, vocabulary)
    disclosure = disclosure_for(metric_id, reading, vocabulary)

    if reading.known:
        line = reading.band
    elif gloss:
        # A named grade with no numeric band. The grade's own gloss *is* the
        # reading — far better than the declared `bands_absent_reason`, which
        # explains to a developer why a band walk was skipped.
        line = gloss
    else:
        line = ""

    unknown = None
    if not line:
        try:
            reason = vocabulary.narrate(reading.reason)
            unknown = Unknown(subject=f"No reading for {name}", reason=reason)
        except ComponentContentError as exc:
            logger.warning(
                f"The declared reason for {metric_id} is not renderable: {exc}"
            )
            unknown = Unknown(
                subject=f"No reading for {name}",
                reason=(
                    "this metric could not be read, and the explanation the "
                    "model recorded is written in terms only the code uses"
                ),
            )

    return MetricRow(
        label=name,
        value=value,
        reading=line,
        unknown=unknown,
        direction=reading.quantity.direction_phrase if reading.quantity else "",
        score=_score_text(score),
        weight=_weight_text(weight_share),
        disclosure=disclosure.ref if disclosure else None,
        metric_id=metric_id,
    )


def _score_text(score) -> str:
    """A *metric's* score, which is a 0–1 fraction and not the element's 0–10.

    Two scales share the word "score" in this system — `details[id]["score"]`
    is `SQGLPScorer._compute_raw_score`'s fraction of full marks, while
    `elements[key]` is out of ten — and a row rendering `0.8 / 10` for a metric
    that scored 80% is the kind of wrong that looks plausible. Percent of full
    marks is spelled out rather than left as a bare `82%`, because the value
    cell beside it is frequently a percentage of something else entirely.

    A fraction outside 0–1 raises rather than rendering `700% of full marks`.
    No shipped caller can produce one — the scorer's contract is the unit
    interval — so reaching here means an element score was passed to a row, and
    silently rendering it would put a confident wrong number in front of a
    reader.
    """
    if not is_number(score):
        return ""
    if not 0.0 <= float(score) <= 1.0:
        raise ComponentContentError(
            f"a metric's score is a 0–1 share of full marks and this one is "
            f"{float(score)} — an element's 0–{SCORE_SCALE} score has been "
            f"passed to a row"
        )
    return f"{float(score):.0%} of full marks"


def _weight_text(share: float | None) -> str:
    if not is_number(share):
        return ""
    return f"{float(share):.0%} of this element"


def score_band(score) -> str | None:
    """`strong` / `middling` / `weak`, or `None` when there is no score."""
    if not is_number(score):
        return None
    for threshold, label in SCORE_BANDS:
        if float(score) >= threshold:
            return label
    return SCORE_LOW_LABEL


def section_reading(
    element: str,
    vocabulary: Vocabulary,
    *,
    score=None,
    coverage: CoverageReading | None = None,
) -> ReadingLine:
    """R5's one line, with R18's clause when the section is thin.

    An unscored element is an unknown-with-reason rather than a dash. The
    reason distinguishes the two ways it happens, because they are different
    facts: an element whose metrics all failed is a data gap, and one nobody
    scored is a pipeline gap.
    """
    title = vocabulary.element_title(element) or ""
    if not title:
        raise ComponentContentError(
            f"Section {element!r} is not one of the report's declared "
            f"sections, so it has no title a reader could be shown (R15)"
        )

    qualifier = ""
    if coverage is not None and coverage.clause:
        qualifier = coverage.clause

    # Band the figure the reader is actually shown, not the one behind it.
    # PFC's Price element scores 6.9658: banding the raw value called it
    # "middling" while the headline rounded to "7.0 / 10", so the line read
    # `7.0 / 10 — Reads middling` against a band boundary at exactly 7. A
    # number disagreeing with its own interpretation on the same line is the
    # defect this report exists to remove, so the rounding happens once.
    shown = round(float(score), 1) if is_number(score) else score
    band = score_band(shown)
    if band is None:
        return ReadingLine(
            subject=title,
            unknown=Unknown(
                subject=f"No score for {title}",
                reason=(
                    "nothing in this section could be scored, so there is no "
                    "reading to give — which is not the same as a score of zero"
                ),
            ),
            qualifier=qualifier,
            key=element,
        )

    return ReadingLine(
        subject=title,
        text=f"Reads {band} for this element.",
        headline=f"{shown:.1f} / {SCORE_SCALE}",
        qualifier=qualifier,
        key=element,
    )


def composite_reading(composite, *, subject: str = COMPOSITE_TITLE,
                      qualifier: str = "") -> ReadingLine:
    """The whole-company reading — `section_reading`'s sibling for the composite.

    The composite is not an element, so no element-shaped builder covered it,
    and both surfaces that needed one built their own. They then diverged on
    the rule `section_reading` exists to state: the note banded the raw figure
    while rounding the headline, the console rounded first. At a composite of
    6.97 the note read `7.0 / 10 — Reads middling` and the console read
    `7.0 / 10 — Reads strong`, for the same company on the same run. That is
    both the rounding defect and an R14 breach, and one builder is the only
    fix that keeps them from drifting apart again.

    `subject` differs by surface on purpose — the note opens on a heading, the
    console labels a table row — but the figure, the band and the sentence do
    not.
    """
    shown = round(float(composite), 1) if is_number(composite) else None
    band = score_band(shown)

    if band is None:
        return ReadingLine(
            subject=subject,
            unknown=Unknown(
                subject=f"No score for {COMPOSITE_TITLE.lower()}",
                reason=COMPOSITE_UNKNOWN_REASON,
            ),
            qualifier=qualifier,
            key="composite",
        )

    return ReadingLine(
        subject=subject,
        text=COMPOSITE_READING.format(band=band),
        headline=f"{shown:.1f} / {SCORE_SCALE}",
        qualifier=qualifier,
        key="composite",
    )


def caveat_from_run_error(raw, *, severity: str = WARNING) -> Caveat:
    """A run error, with the untrusted half never rendered.

    `service.py` writes `f"Data fetch failed: {e}"` into `result.errors` and
    the current report prints the whole thing. The clause in front of the colon
    is authored prose and worth keeping; everything after it is `str(exc)` and
    is not, whatever shape it happens to have — `Exception("Screener returned
    404")` is indistinguishable from a sentence somebody wrote, so no scanner
    can be trusted to catch it and the split is made on structure instead.

    The raw text goes to the log, where it is useful and nobody reads it as
    prose.
    """
    text = "" if raw is None else str(raw)
    logger.warning(f"Run error, kept out of the report: {text}")

    head, separator, _tail = text.partition(":")
    detail = "the technical detail is in the run log rather than here"
    if separator and head.strip():
        try:
            return Caveat(
                text=f"{guard_text(head.strip(), field='text')} — {detail}",
                severity=severity,
            )
        except ComponentContentError:
            pass

    return Caveat(
        text=f"Part of this analysis did not complete — {detail}",
        severity=severity,
    )


def build_section(
    element: str,
    decision: SectionDecision,
    readings: Mapping[str, Reading],
    vocabulary: Vocabulary,
    scores: Mapping | None = None,
    *,
    flags: Sequence[str] = (),
    weight_shares: Mapping[str, float] | None = None,
) -> Section:
    """One element's whole section, from the declarations three surfaces share.

    Assembled here rather than in each renderer, which is what R14 actually
    requires: "the same content from the same declarations" is a promise about
    where the content is built, not about how carefully three templates were
    written. A surface receives a `Section` and decides only how it looks.

    Rows come from `decision.metrics`, which is every metric declared in this
    element — including the zero-weight ones, which are excluded from the
    expansion decision but are still readings a reader wants. A metric with no
    reading at all is not invented: `read_metrics` produces one for every
    declared metric, so its absence means the caller did not read it, and the
    section says so in its unknowns rather than pretending.
    """
    title = vocabulary.element_title(element)
    if not title:
        raise ComponentContentError(
            f"Section {element!r} is not one of the report's declared "
            f"sections, so it has no title a reader could be shown (R15)"
        )

    element_scores = ((scores or {}).get("elements") or {})
    details = ((scores or {}).get("details") or {})
    shares = dict(weight_shares or {})

    reading_line = section_reading(
        element, vocabulary,
        score=element_scores.get(element),
        coverage=decision.coverage,
    )

    findings = [finding_from_reason(reason) for reason in decision.reasons]
    unknowns: list[Unknown] = []
    rows: list[MetricRow] = []
    disclosures: list[Disclosure] = []

    for metric in decision.metrics:
        built = _row_for(metric, readings, details, shares, vocabulary)
        if isinstance(built, Unknown):
            unknowns.append(built)
            continue
        rows.append(built)
        disclosure = disclosure_for(metric.metric_id, readings[metric.metric_id],
                                    vocabulary)
        if disclosure is not None:
            disclosures.append(disclosure)

    for flag in flags:
        built_flag = finding_from_flag(flag)
        if isinstance(built_flag, Unknown):
            unknowns.append(built_flag)
        else:
            findings.append(built_flag)

    caveats = [
        Caveat(text=text, severity=NOTE)
        for text in decision.unresolved
        if str(text).strip()
    ]

    return Section(
        key=element,
        title=title,
        reading=reading_line,
        findings=tuple(findings),
        rows=tuple(rows),
        unknowns=tuple(unknowns),
        caveats=tuple(caveats),
        disclosures=tuple(disclosures),
        expanded=decision.expand,
    )


def _row_for(
    metric: MetricDecision,
    readings: Mapping[str, Reading],
    details: Mapping,
    shares: Mapping[str, float],
    vocabulary: Vocabulary,
) -> MetricRow | Unknown:
    """One `MetricDecision` turned into a row, or an unknown naming the gap."""
    reading = (readings or {}).get(metric.metric_id)
    if reading is None:
        name = vocabulary.metric_name(metric.metric_id) or "One metric"
        return Unknown(
            subject=f"No reading for {name}",
            reason=(
                "this metric was not read for this company, so nothing about "
                "it — not even that it failed — can be stated here"
            ),
        )
    detail = details.get(metric.metric_id) or {}
    score = detail.get("score") if isinstance(detail, Mapping) else None
    return metric_row(
        metric.metric_id, reading, vocabulary,
        score=score,
        weight_share=shares.get(metric.metric_id),
    )


# ── The surface contract (R14) ────────────────────────────────────────────
#
# "A component set member missing from any surface's renderer is a test
# failure." U10 and U11 have not been written, so the mechanism has to bind
# them when they arrive rather than rely on their authors reading this file.
# The decorator does that at import time: a renderer missing a handler cannot
# be defined at all, so the failure lands on the line that omitted it.

HANDLER_FOR_KIND: dict[str, str] = {kind: f"render_{kind}" for kind in COMPONENT_KINDS}

SURFACES: dict[str, type] = {}


def missing_members(surface) -> tuple[str, ...]:
    """Which components this renderer cannot render, in the set's order."""
    return tuple(
        kind for kind in COMPONENT_KINDS
        if not callable(getattr(surface, HANDLER_FOR_KIND[kind], None))
    )


def verify_surface(surface, name: str = "") -> None:
    """Raise unless the renderer handles every member of the closed set."""
    missing = missing_members(surface)
    if missing:
        label = name or getattr(surface, "__name__", type(surface).__name__)
        raise IncompleteSurface(
            f"{label} does not render {', '.join(missing)} — every surface "
            f"renders every component, or a section exists in one place and "
            f"not the others (R14). Add "
            f"{', '.join(HANDLER_FOR_KIND[kind] for kind in missing)}."
        )


def component_surface(name: str):
    """Register a renderer, refusing it if it is missing a member.

    Registration is by name from `EXPECTED_SURFACES`, so a fourth surface — or
    a typo in one of the three — is caught here rather than showing up as a
    report format nobody checked.
    """
    def decorate(cls):
        if name not in EXPECTED_SURFACES:
            raise IncompleteSurface(
                f"{name!r} is not one of the surfaces R14 names "
                f"({', '.join(EXPECTED_SURFACES)})"
            )
        verify_surface(cls, name)
        SURFACES[name] = cls
        return cls
    return decorate


def component_fields(component_type: type) -> tuple[str, ...]:
    """Every declared field. Used by the test that pins the text/handle split."""
    return tuple(f.name for f in fields(component_type))


__all__ = [
    "BAD",
    "CAVEAT",
    "COMPONENT_KINDS",
    "COMPONENT_TYPES",
    "DISCLOSURE",
    "EXPECTED_SURFACES",
    "FINDING",
    "GOOD",
    "HANDLER_FOR_KIND",
    "METRIC_ROW",
    "NEUTRAL",
    "NOTE",
    "READING",
    "RESERVED_KEYS",
    "SEVERITIES",
    "SENTIMENTS",
    "SURFACES",
    "UNKNOWN",
    "WARNING",
    "Caveat",
    "ComponentContentError",
    "Disclosure",
    "DisclosureRef",
    "Finding",
    "IncompleteSurface",
    "MetricRow",
    "ReadingLine",
    "Section",
    "Unknown",
    "Vocabulary",
    "build_section",
    "caveat_from_run_error",
    "component_fields",
    "component_surface",
    "disclosure_for",
    "finding_from_flag",
    "finding_from_reason",
    "guard_quantity",
    "guard_text",
    "metric_row",
    "missing_members",
    "score_band",
    "section_reading",
    "verify_surface",
]
