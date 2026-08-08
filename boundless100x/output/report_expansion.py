"""Whether a section has earned the space to say more than its score (R5–R9).

F1's decision, and nothing else. For each metric in a section this asks three
questions in a fixed order — does the metric measure anything for a company of
this kind, does it belong to a declared pair whose two readings disagree, and
did it score zero while carrying enough weight to have mattered — then ORs the
per-metric answers. A section with no fired trigger renders collapsed; one with
any fires expanded, listing **every** reason (R7, R9).

This module **decides**. It renders nothing: U10 and U11 turn a
`SectionDecision` into a page through U9's components, so every string here is
data a surface may print rather than markup it must parse. The reasons are
nonetheless the deliverable, not a byproduct — KD3 makes length information a
reader skims by, and a section that grew without saying why has spent the
reader's attention and given nothing back.

**The evaluation order is load-bearing and is F1's, not a convenience.** Sector
mismatch is decided first and terminates that metric's path, which is what
makes R8's last sentence true by construction: a metric that both fails to
apply to this sector *and* reads zero across the corpus still expands, because
it never reaches the corpus-relative test that would have suppressed it. Order
here is the mechanism; a later `or` over three independently computed booleans
would look equivalent and quietly lose it.

── Which unknown fires, and which does not ───────────────────────────────

"Indeterminate is never a silent pass" is this system's spine, and applying it
here needs one distinction, because the unknowns in this decision are not all
the same kind of thing.

An unknown **trigger condition** does not fire. Applicability reads
indeterminate for every sector nobody has reviewed — which is most of them, two
being declared today — and a contradiction reads indeterminate on every run
that evaluated no eligibility gates, which is every `watchlist advance`. Firing
on those would expand every section of nearly every company, and an expansion
that happens to everyone tells a reader nothing about anyone: it would destroy
the very signal KD3 built. So the check is recorded in `unresolved` with the
sentence explaining what could not be run, and the section's *shape* stays
honest about the fact that nobody made a finding.

An unknown **suppression** does not suppress. R8 says so outright, and the
asymmetry is principled rather than a special case: the zero-score condition is
already *known* — this metric scored zero and it carries real weight — and only
the corpus-relative rule that would hide it is unknown. Suppression is the
mechanism that hides a real gap, so an unknown one must not get to hide
anything. It fires, and the reason states how many scored reports exist and how
many are needed (AE8).

The two rules point in opposite directions and reach the same place: an unknown
never invents a finding, and an unknown never buries one.

── The corpus (R8, KTD5) ─────────────────────────────────────────────────

`score_history.jsonl` cannot answer R8's question: it carries element scores,
not per-metric ones, so it cannot say whether *this metric* scored zero. The
per-metric detail exists only in the `scores.json` written beside each
generated report, so that is the corpus — and it is machine-local and
gitignored, which is why `load_scored_corpus` degrades to an empty reading with
a reader-facing explanation rather than raising, and why an empty reading reads
as *below the minimum* rather than as *nothing to suppress*. Those two are
opposite outcomes for the same input and only one of them is safe.

The corpus is read once and handed to the decider, not loaded per section:
`lifecycle/pace.py` establishes the shape for a corpus-median input, and a
module-level cache would be wrong here for a reason the YAML loaders do not
face — a declaration file does not change while the process runs, and the
reports directory gains an entry every time `analyze` finishes.

One bound worth stating: a `scores.json` carries no `config_hash`, so unlike
`trajectory.py` this cannot refuse to compare across scoring regimes. A corpus
spanning a registry change is read at face value. The exposure is bounded by
what is being counted — whether a score was exactly zero is a coarse reading,
and a threshold edit moves scores off zero far less often than it moves them at
all — but it is a real limitation rather than an oversight.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from boundless100x.output.report_reading import (
    CoverageReading,
    Reading,
    is_number,
    read_element_coverages,
)

logger = logging.getLogger(__name__)

# Where the per-metric corpus lives. Mirrors `ReportGenerator.__init__`'s own
# default rather than importing it: this module must not import
# `report_generator`, which imports the whole output stack and would invert the
# dependency direction U6 established and a test keeps closed.
DEFAULT_REPORTS_DIR = Path(__file__).parent / "reports"

# ── The trigger vocabulary ────────────────────────────────────────────────
#
# Four kinds for three triggers: the zero-score test has two distinct fired
# outcomes and they say different things to a reader. One is a finding about
# this company; the other is an admission that the finding cannot be made yet.
# Collapsing them would let "we checked and this is unusual" and "we cannot
# check" print the same way, which is the confusion R8's last clause exists to
# prevent.

SECTOR_MISMATCH = "sector_mismatch"
CONTRADICTION = "contradiction"
ZERO_SCORE_GAP = "zero_score_gap"
ZERO_SCORE_NOT_COMPARABLE = "zero_score_not_comparable"

TRIGGERS = (
    SECTOR_MISMATCH,
    CONTRADICTION,
    ZERO_SCORE_GAP,
    ZERO_SCORE_NOT_COMPARABLE,
)

# What a surface calls each one. Data only, following `report_vocabulary`'s
# convention, and kept beside the constants it names so the two cannot drift —
# a label file listing three of four kinds would render the fourth through
# whatever fallback the surface happens to have, which is the auto-humanising
# failure the problem frame names. U9 and U10 read these; they are never a
# substitute for the reason itself, which is what R7 actually asks for.
TRIGGER_LABELS: dict[str, str] = {
    SECTOR_MISMATCH: "Measures the wrong thing for this kind of company",
    CONTRADICTION: "Two readings here disagree",
    ZERO_SCORE_GAP: "Scored zero on something that matters",
    ZERO_SCORE_NOT_COMPARABLE: "Scored zero, with nothing yet to compare it to",
}

# R6's weight bar: a metric must carry at least this share of its *element's*
# declared weight before a zero in it is worth a reader's attention. A share of
# the element, not of the composite and not the raw weight — 0.05 is a
# twentieth of Quality — Business and a twentieth of Longevity, but the two
# elements declare different totals, and a rule stated in raw weights would
# drift the moment one of them gained a metric.
MIN_WEIGHT_SHARE = 0.10

# KTD5's simple majority: strictly more than half. Measured across the seven
# scored reports on disk (2026-08-08), `>0.5` and `>0.6` both suppress exactly
# `dcf_margin_of_safety` while `>0.75` suppresses nothing and the test does no
# work at all. Strictly-greater matters at even denominators: three of six is
# not a majority, and `ebit_cagr_3yr`, `roce_consistency` and `analyst_coverage`
# all sit at exactly that today.
CORPUS_MAJORITY = 0.5

# How many reports must have *computed this metric* before its zero-rate is
# allowed to suppress anything. Measured rather than chosen: over every subset
# of the seven scored reports, a leave-one-out check at six comparable readings
# reproduces the whole corpus's answer in 7 of 7 subsets, while at five it
# agrees in 4 of 21 and the suppressed set swells from one metric to four. The
# instability below six is not noise in the threshold — it is metrics whose
# score was computed for two or three companies reading zero in all of them,
# which is a fact about a handful of companies dressed as a fact about the
# model. Six is where dropping any single company stops changing the answer.
#
# It is a **per-metric** count, not the corpus size, because it is the
# denominator of the test being run: a fifty-report corpus in which this metric
# computed twice still cannot say whether its zero is unusual.
MIN_COMPARABLE_REPORTS = 6

# Why a metric is not eligible to fire anything. KTD5's argument, generalised
# from the contradiction pool it was written about to all three triggers:
# expansion is prominence, and a signal that deliberately cannot move a score
# must not be able to move the report's shape instead. `contradiction.py`
# already refuses these structurally at load; stating it once here keeps the
# other two triggers from acquiring the coupling by omission.
ZERO_WEIGHT_EXCLUSION = (
    "carries no weight in this element's score, so nothing it reads can make "
    "the section's score need explaining"
)

UNKNOWN_METRIC_EXCLUSION = (
    "is not a metric this registry computes, so there is nothing here to decide"
)


def _plural(count: int, noun: str) -> str:
    """`3 reports` / `1 report`. A reason string is prose and reads like it."""
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _is_zero(score) -> bool:
    """Whether a raw score is exactly zero.

    Exact equality on purpose. Every branch of `SQGLPScorer._compute_raw_score`
    reaches zero exactly — `0 / len(thresholds)`, `max(0.0, ...)`, a literal
    `0.0`, `categories.get(value, 0) / 10.0` — so a tolerance would only widen
    the rule to values the scorer never produces, and R6 says zero rather than
    nearly zero. Bools are excluded for the reason `report_reading._is_number`
    gives: `False == 0` is True, and a bool is not a score.
    """
    if not is_number(score):
        return False
    return score == 0


# ── The corpus ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CorpusZeroRate:
    """How often one metric reads zero across the corpus, and whether that counts.

    `suppresses` is three-valued and the `None` is the substance: it means the
    corpus could not answer, which under R8 must not suppress. A `False` here
    says the corpus was asked and this metric is not a corpus-wide zero.
    """

    metric_id: str
    zero: int
    comparable: int
    reports: int
    minimum: int
    unreadable: int = 0
    corpus_error: str = ""

    @property
    def comparable_enough(self) -> bool:
        return self.comparable >= self.minimum

    @property
    def share(self) -> float | None:
        return self.zero / self.comparable if self.comparable else None

    @property
    def suppresses(self) -> bool | None:
        """True to suppress, False not to, `None` when the corpus cannot say."""
        if not self.comparable_enough:
            return None
        return self.zero > self.comparable * CORPUS_MAJORITY


@dataclass(frozen=True)
class ScoredCorpus:
    """Per-metric zero counts across the scored reports on disk.

    `reports` counts companies, not directories: a ticker analysed four times
    contributes once. Counting the directories would let one company carry four
    votes in a majority test whose whole claim is that several *companies* read
    the same way.
    """

    reports: int = 0
    tickers: tuple[str, ...] = ()
    zero_counts: Mapping[str, int] = field(default_factory=dict)
    comparable_counts: Mapping[str, int] = field(default_factory=dict)
    minimum: int = MIN_COMPARABLE_REPORTS
    source: str = ""
    unreadable: tuple[str, ...] = ()
    # A reader-facing sentence, never `str(exc)` — R15 keeps exception strings
    # off the page, and this one travels into a reason a surface prints. The
    # exception itself goes to the log, where it is useful and nobody reads it
    # as prose.
    error: str = ""

    def rate_for(self, metric_id: str) -> CorpusZeroRate:
        return CorpusZeroRate(
            metric_id=metric_id,
            zero=int(self.zero_counts.get(metric_id, 0)),
            comparable=int(self.comparable_counts.get(metric_id, 0)),
            reports=self.reports,
            minimum=self.minimum,
            unreadable=len(self.unreadable),
            corpus_error=self.error,
        )


def _ticker_and_date(directory_name: str) -> tuple[str, str]:
    """Split `PFC_20260808` into its ticker and its date stamp.

    `rpartition` rather than `split`, because a ticker may contain an
    underscore and the date never does. A name with no separator at all is its
    own ticker with no date — it still dedupes against itself, which is the
    only property this needs to preserve.
    """
    ticker, _, date = directory_name.rpartition("_")
    return (ticker, date) if ticker else (directory_name, "")


def load_scored_corpus(
    reports_dir: str | Path | None = None,
    *,
    minimum: int = MIN_COMPARABLE_REPORTS,
) -> ScoredCorpus:
    """Read every generated report's `scores.json` and count the zeros.

    **Latest report per ticker.** Reports are date-stamped directories and a
    company re-analysed weekly accumulates them; counting each one would let a
    single company decide a majority test about the model.

    **A metric that errored scores `None`, and `None` is not zero.** It is
    dropped from the numerator *and* the denominator, per R18's own wording. It
    has to be both: counted in the denominator only, an element whose metrics
    mostly error would look like a metric that mostly does not read zero, and
    the suppression rule would quietly stop suppressing.

    An unreadable directory degrades to an empty corpus carrying the reason. It
    never raises — this runs inside report generation, and losing a report over
    a corpus that is by design optional and machine-local would be the wrong
    trade — and it never resolves to "nothing to suppress", which is the
    dangerous direction: a missing corpus that read as clean would hide every
    real gap on every company, silently and forever.
    """
    target = Path(reports_dir) if reports_dir else DEFAULT_REPORTS_DIR
    source = str(target)

    try:
        entries = sorted(p for p in target.iterdir() if p.is_dir())
    except OSError as exc:
        logger.warning(f"Could not read the scored-report corpus at {target}: {exc}")
        return ScoredCorpus(
            minimum=minimum,
            source=source,
            error=(
                "the folder holding previously generated reports could not be "
                "read, so there is nothing to compare this company against"
            ),
        )

    # Latest date wins per ticker. Dates are `YYYYMMDD`, so the lexical maximum
    # is the chronological one and no parsing is needed to order them.
    latest: dict[str, Path] = {}
    for entry in entries:
        ticker, date = _ticker_and_date(entry.name)
        current = latest.get(ticker)
        if current is None or _ticker_and_date(current.name)[1] <= date:
            latest[ticker] = entry

    zero_counts: dict[str, int] = {}
    comparable_counts: dict[str, int] = {}
    counted: list[str] = []
    unreadable: list[str] = []

    for ticker, directory in sorted(latest.items()):
        path = directory / "scores.json"
        try:
            payload = json.loads(path.read_text())
            details = payload["details"]
            if not isinstance(details, Mapping):
                raise TypeError(f"details is {type(details).__name__}, not a mapping")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            # One malformed report must not cost the whole corpus, but it must
            # not vanish either: it shifts every denominator it should have
            # been in, so the count of skipped reports travels with the reading.
            logger.warning(f"Skipping unreadable scores at {path}: {exc}")
            unreadable.append(ticker)
            continue

        counted.append(ticker)
        for metric_id, detail in details.items():
            score = (detail or {}).get("score") if isinstance(detail, Mapping) else None
            if score is None:
                continue
            comparable_counts[metric_id] = comparable_counts.get(metric_id, 0) + 1
            if _is_zero(score):
                zero_counts[metric_id] = zero_counts.get(metric_id, 0) + 1

    error = ""
    if not counted:
        error = (
            "no previously generated report could be read, so there is nothing "
            "to compare this company against"
        )

    return ScoredCorpus(
        reports=len(counted),
        tickers=tuple(counted),
        zero_counts=zero_counts,
        comparable_counts=comparable_counts,
        minimum=minimum,
        source=source,
        unreadable=tuple(unreadable),
        error=error,
    )


# ── What a fired trigger says ─────────────────────────────────────────────


@dataclass(frozen=True)
class ExpansionReason:
    """One fired trigger, in the words R7 asks for.

    `metric_id` is a handle for a caller that needs to find the row again; the
    id must never be rendered (R15), which is what `metric_name` is for. `text`
    is non-blank by construction: a trigger that fires with nothing to say
    makes a section longer and then shrugs at the reader, which is worse than
    the collapse it replaced.
    """

    trigger: str
    metric_id: str
    metric_name: str
    text: str

    def __post_init__(self):
        if self.trigger not in TRIGGERS:
            raise ValueError(
                f"{self.metric_id}: trigger {self.trigger!r} is not one of "
                f"{', '.join(TRIGGERS)}"
            )
        if not self.text.strip():
            raise ValueError(
                f"{self.metric_id}: a fired trigger must carry the reason it "
                f"fired (R7)"
            )

    @property
    def label(self) -> str:
        return TRIGGER_LABELS[self.trigger]


@dataclass(frozen=True)
class MetricDecision:
    """One metric's walk through F1, and where it stopped.

    Three outcomes, and the third is the one worth naming: fired (with its
    reasons), did not fire, or was never eligible to — `excluded_reason` marks
    a zero-weight signal, which is not the same as a metric that was checked
    and had nothing to report.
    """

    metric_id: str
    metric_name: str
    element: str
    reasons: tuple[ExpansionReason, ...] = ()
    unresolved: tuple[str, ...] = ()
    excluded_reason: str = ""

    @property
    def fired(self) -> bool:
        return bool(self.reasons)

    @property
    def considered(self) -> bool:
        return not self.excluded_reason


@dataclass(frozen=True)
class SectionDecision:
    """R6's OR across one section's metrics, and every reason behind it.

    `expand` is two-valued rather than three, deliberately: a section renders
    collapsed or expanded and there is no third rendering, so an indeterminate
    here would have to resolve to one of them anyway and would only hide which.
    The three-valued-ness lives where it can be acted on — per metric, and in
    `unresolved`, which lists the checks that could not be run at all.

    No cap and no roll-up (KD5, KD4, R9). Every fired reason is carried, and a
    finding several sections reach independently is stated in each of them
    rather than deduplicated into one.
    """

    element: str
    reasons: tuple[ExpansionReason, ...] = ()
    unresolved: tuple[str, ...] = ()
    metrics: tuple[MetricDecision, ...] = ()
    coverage: CoverageReading | None = None

    @property
    def expand(self) -> bool:
        return bool(self.reasons)

    @property
    def fired_triggers(self) -> tuple[str, ...]:
        """The distinct trigger kinds behind this section's size, in F1's order."""
        fired = {reason.trigger for reason in self.reasons}
        return tuple(trigger for trigger in TRIGGERS if trigger in fired)


# ── Registry arithmetic, needing no collaborators ─────────────────────────
#
# Module functions rather than methods because they are pure functions of the
# registry: they read no corpus, no declared pairs, and no company. A caller
# that wants a weight share was otherwise forced to build a whole
# `ExpansionDecider` — which validates the contradiction table and globs and
# parses every `reports/*/scores.json` on disk — to perform one division. The
# console did exactly that on every `compute`, and that directory grows by one
# entry per ticker per analysis date and is never pruned.


def declared_element_weights(metric_configs) -> dict[str, float]:
    """Total weight each element would carry if every metric computed.

    `SQGLPScorer._declared_weights`'s rule, restated for the reason
    `contradiction.py` restates `_scored`: it is four lines, and importing a
    private method to reach it would tie the output layer to the scorer's
    internals for one sum. The risk in restating is that the two drift, so a
    test asserts they agree on the shipped registry — the drift is the failure,
    not the duplication.
    """
    declared: dict[str, float] = {}
    for config in (metric_configs or {}).values():
        weight = (config.get("scoring") or {}).get("weight", 0) or 0
        if weight > 0:
            element = config.get("element", "")
            declared[element] = declared.get(element, 0) + weight
    return declared


def declared_weight_shares(metric_configs) -> dict[str, float | None]:
    """Every metric's share of its own element's declared weight.

    `None` for a metric carrying no weight — it has no share of anything, and
    a zero would read as one that contributes nothing rather than one that
    was never in the denominator.
    """
    totals = declared_element_weights(metric_configs)
    shares: dict[str, float | None] = {}
    for metric_id, config in (metric_configs or {}).items():
        weight = (config.get("scoring") or {}).get("weight", 0) or 0
        total = totals.get(config.get("element", ""), 0)
        shares[metric_id] = weight / total if weight > 0 and total > 0 else None
    return shares


# ── The decider ───────────────────────────────────────────────────────────


class ExpansionDecider:
    """F1, for every section of one company.

    All three collaborators are **required and positional**, following
    `SectorApplicability` and `ContradictionPairs` rather than
    `LaneGateEvaluator`: an optional collaborator whose absence disables a check
    is a check that silently never runs, and here the silence would be
    indistinguishable from a company with nothing wrong with it.

    `contradictions` is a `ContradictionPairs`. `corpus` is a `ScoredCorpus` —
    passed in rather than loaded here so the source is visible at every call
    site, so a test never depends on a machine-local directory, and so the
    seven-file read happens once per run rather than once per section.
    """

    def __init__(
        self,
        metric_configs: Mapping,
        contradictions,
        corpus: ScoredCorpus,
        *,
        min_weight_share: float = MIN_WEIGHT_SHARE,
    ):
        self.metric_configs = dict(metric_configs or {})
        self.contradictions = contradictions
        self.corpus = corpus
        self.min_weight_share = min_weight_share
        self.declared_element_weights = self._declared_element_weights()

    # ── Registry arithmetic ───────────────────────────────────────────────

    def _declared_element_weights(self) -> dict[str, float]:
        """Total weight each element would carry if every metric computed."""
        return declared_element_weights(self.metric_configs)

    def _weight(self, metric_id: str) -> float:
        config = self.metric_configs.get(metric_id) or {}
        return (config.get("scoring") or {}).get("weight", 0) or 0

    def weight_share(self, metric_id: str) -> float | None:
        """This metric's share of its own element's declared weight.

        The *declared* weight from the registry, not the `weight` the scorer
        wrote into `details` — the scorer zeroes that field for a waived or
        errored metric, so reading it here would compute a share of zero for
        exactly the metrics R18 already says are excluded, and arrive at the
        right answer by the wrong route.
        """
        weight = self._weight(metric_id)
        if weight <= 0:
            return None
        element = (self.metric_configs.get(metric_id) or {}).get("element", "")
        total = self.declared_element_weights.get(element, 0)
        return weight / total if total > 0 else None

    def _metric_name(self, metric_id: str) -> str:
        """R15's display name, read off the registry.

        The registry's own `name` rather than `report_vocabulary`'s
        `METRIC_DISPLAY_NAMES`, matching `contradiction.py`. That map is
        hand-maintained and can omit a metric, and its omission path is the
        silent drop the problem frame names as a defect; every metric the
        engine discovers has a `name`.
        """
        return str((self.metric_configs.get(metric_id) or {}).get("name") or metric_id)

    def elements(self) -> list[str]:
        """Every element that declares weight, alphabetically.

        Alphabetical for determinism only. Report order is the renderer's
        business — `report_vocabulary.ELEMENT_CONFIG` states it — and a caller
        that wants it passes `elements=` to `evaluate`.
        """
        return sorted(self.declared_element_weights)

    # ── The three triggers, in F1's order ─────────────────────────────────

    def _sector_reason(self, metric_id: str, reading: Reading) -> ExpansionReason:
        return ExpansionReason(
            trigger=SECTOR_MISMATCH,
            metric_id=metric_id,
            metric_name=self._metric_name(metric_id),
            # The table's own sentence, verbatim (R7). This layer supplies the
            # lead-in that says why the reader is being shown it and never
            # paraphrases the explanation itself.
            text=(
                f"{self._metric_name(metric_id)} does not measure anything for a "
                f"company of this kind, so what it scored here says nothing "
                f"about this business. {reading.applicability.reason}"
            ),
        )

    def _contradiction_reasons(
        self, metric_id: str, outcome: Mapping
    ) -> list[ExpansionReason]:
        name = self._metric_name(metric_id)
        return [
            ExpansionReason(
                trigger=CONTRADICTION,
                metric_id=metric_id,
                metric_name=name,
                # Again the declaration verbatim — `contradiction_pairs.yaml`'s
                # reasons are written to reconcile the two readings for an
                # owner, which is precisely what R7 asks an expanded section to
                # put in front of them.
                text=f"{name} disagrees with another reading in this report. {text}",
            )
            for text in outcome.get("reasons", [])
            if str(text).strip()
        ]

    def _zero_score_reason(
        self, metric_id: str, share: float, rate: CorpusZeroRate
    ) -> ExpansionReason | None:
        """F1's `D → M → E` tail: fire, fire-but-cannot-tell, or suppress."""
        name = self._metric_name(metric_id)
        suppresses = rate.suppresses

        if suppresses is True:
            # KTD5. The metric is describing the model, not the company, and
            # saying so on every company would be a finding about none of them.
            return None

        if suppresses is None:
            if rate.corpus_error:
                shortfall = rate.corpus_error
            else:
                shortfall = (
                    f"only {rate.comparable} of the {rate.reports} scored "
                    f"reports on file computed it, against the "
                    f"{rate.minimum} needed"
                )
                if rate.unreadable:
                    shortfall += (
                        f", and {_plural(rate.unreadable, 'further report')} on "
                        f"disk could not be read at all"
                    )
            return ExpansionReason(
                trigger=ZERO_SCORE_NOT_COMPARABLE,
                metric_id=metric_id,
                metric_name=name,
                text=(
                    f"{name} scored zero, and it carries {share:.0%} of this "
                    f"element's declared weight — enough to pull the score down "
                    f"on its own. Whether that is unusual cannot be told yet: "
                    f"{shortfall}. Until then it is shown rather than filtered "
                    f"out, because a gap nobody could check is not a gap that "
                    f"went away."
                ),
            )

        return ExpansionReason(
            trigger=ZERO_SCORE_GAP,
            metric_id=metric_id,
            metric_name=name,
            text=(
                f"{name} scored zero, and it carries {share:.0%} of this "
                f"element's declared weight — enough to pull the score down on "
                f"its own. It reads zero in {rate.zero} of the "
                f"{rate.comparable} analysed companies that computed it, so "
                f"this is a gap in this company rather than a reading that "
                f"comes out zero for everyone."
            ),
        )

    # ── Per metric ────────────────────────────────────────────────────────

    def evaluate_metric(
        self,
        metric_id: str,
        readings: Mapping[str, Reading],
        scores: Mapping | None = None,
        eligibility=None,
    ) -> MetricDecision:
        """F1's per-metric walk. Applicability, then the pair, then the zero.

        Each fired trigger **terminates the walk**, exactly as the diagram's
        branches do. That is what makes R8's closing sentence structural: a
        metric excluded by sector never reaches the corpus-relative test, so no
        amount of corpus-wide zero can suppress a category mismatch.
        """
        config = self.metric_configs.get(metric_id)
        name = self._metric_name(metric_id)
        element = str((config or {}).get("element", ""))

        if config is None:
            return MetricDecision(
                metric_id=metric_id, metric_name=name, element="",
                excluded_reason=f"{name} {UNKNOWN_METRIC_EXCLUSION}",
            )

        if self._weight(metric_id) <= 0:
            return MetricDecision(
                metric_id=metric_id, metric_name=name, element=element,
                excluded_reason=f"{name} {ZERO_WEIGHT_EXCLUSION}",
            )

        unresolved: list[str] = []
        reading = (readings or {}).get(metric_id)
        if reading is None:
            # Not a missing *value* — `read_metrics` produces a reading for every
            # declared metric, unknown-with-reason included — but a caller that
            # never read this one. None of the three checks can run on it, and
            # a section quietly deciding it had nothing to say is the failure.
            return MetricDecision(
                metric_id=metric_id, metric_name=name, element=element,
                unresolved=(
                    f"{name} was not read for this company, so none of the "
                    f"checks that decide whether this section expands could be "
                    f"run on it",
                ),
            )

        # 1. Does it apply to this sector? Decided first, and not suppressible.
        applies = reading.applicability
        if applies.excluded:
            return MetricDecision(
                metric_id=metric_id, metric_name=name, element=element,
                reasons=(self._sector_reason(metric_id, reading),),
            )
        if not applies.known:
            unresolved.append(
                f"Whether {name} means anything for a company of this kind has "
                f"not been checked, so a category mismatch here would not have "
                f"been caught. {applies.reason}"
            )

        # 2. Does a declared pair disagree?
        outcome = self.contradictions.evaluate(metric_id, readings, eligibility)
        if outcome.get("contradicts") is True:
            reasons = self._contradiction_reasons(metric_id, outcome)
            if reasons:
                return MetricDecision(
                    metric_id=metric_id, metric_name=name, element=element,
                    reasons=tuple(reasons), unresolved=tuple(unresolved),
                )
        elif outcome.get("contradicts") is None:
            unresolved.extend(
                f"A declared contradiction check on {name} could not be run. "
                f"{text}"
                for text in outcome.get("reasons", [])
                if str(text).strip()
            )

        # 3. Did it score zero while carrying enough weight to matter?
        detail = ((scores or {}).get("details") or {}).get(metric_id) or {}
        score = detail.get("score") if isinstance(detail, Mapping) else None
        if not _is_zero(score):
            # Includes the errored and waived metrics, whose score is `None`.
            # They are not zeros and R18 answers them: the section's collapsed
            # reading states the coverage they cost it. A line here per errored
            # metric would put eight of them in front of a PFC reader and say
            # the same thing eight times.
            return MetricDecision(
                metric_id=metric_id, metric_name=name, element=element,
                unresolved=tuple(unresolved),
            )

        share = self.weight_share(metric_id)
        if share is None or share < self.min_weight_share:
            return MetricDecision(
                metric_id=metric_id, metric_name=name, element=element,
                unresolved=tuple(unresolved),
            )

        reason = self._zero_score_reason(
            metric_id, share, self.corpus.rate_for(metric_id)
        )
        return MetricDecision(
            metric_id=metric_id, metric_name=name, element=element,
            reasons=(reason,) if reason is not None else (),
            unresolved=tuple(unresolved),
        )

    # ── Per section ───────────────────────────────────────────────────────

    def metrics_in(self, element: str) -> list[str]:
        return [
            metric_id
            for metric_id, config in self.metric_configs.items()
            if config.get("element") == element
        ]

    def evaluate_section(
        self,
        element: str,
        readings: Mapping[str, Reading],
        scores: Mapping | None = None,
        *,
        eligibility=None,
        coverage: CoverageReading | None = None,
    ) -> SectionDecision:
        """R6's OR across one section, carrying every reason (R7, R9).

        Reasons are ordered by the metric name a reader sees, which is
        arbitrary but stable — two runs of the same report must not shuffle
        their own paragraphs. Within one metric, declaration order is kept.
        """
        decisions = [
            self.evaluate_metric(metric_id, readings, scores, eligibility)
            for metric_id in self.metrics_in(element)
        ]
        decisions.sort(key=lambda decision: (decision.metric_name, decision.metric_id))

        unresolved: list[str] = []
        readings_absent = not readings
        if readings_absent:
            unresolved.append(
                "No metric in this section was read for this company, so "
                "nothing here has been checked for whether it needs explaining"
            )
        if not ((scores or {}).get("details")):
            unresolved.append(
                "No scores were supplied for this company, so a metric that "
                "scored zero on something that matters could not be spotted"
            )

        reasons: list[ExpansionReason] = []
        for decision in decisions:
            reasons.extend(decision.reasons)
            if readings_absent:
                # Every metric's own line says the same thing, thirteen times
                # in Quality — Business. The section-level line above says it
                # once. The per-metric lines are still on the `MetricDecision`,
                # so a caller that wants them per row has lost nothing.
                continue
            unresolved.extend(decision.unresolved)

        return SectionDecision(
            element=element,
            reasons=tuple(reasons),
            unresolved=tuple(unresolved),
            metrics=tuple(decisions),
            coverage=coverage,
        )

    def evaluate(
        self,
        readings: Mapping[str, Reading],
        scores: Mapping | None = None,
        *,
        eligibility=None,
        elements: Sequence[str] | None = None,
    ) -> dict[str, SectionDecision]:
        """Every section of one company.

        The R18 coverage clause is derived here from the same `scores` block the
        expansion decision is made against, rather than accepted as a second
        argument. Two inputs could disagree — a section could state the coverage
        of one run while deciding its size from another — and there is no route
        by which a reader could tell.
        """
        coverages = read_element_coverages((scores or {}).get("coverage"))
        targets = list(elements) if elements is not None else self.elements()
        return {
            element: self.evaluate_section(
                element, readings, scores,
                eligibility=eligibility, coverage=coverages.get(element),
            )
            for element in targets
        }


def expanded_sections(
    decisions: Mapping[str, SectionDecision]
) -> list[str]:
    """The sections that earned their space, in the order they were decided.

    A convenience for a caller that wants the shape of a report before its
    content — AE5's "every section collapsed" is this list being empty, and
    KD5's "the length is the verdict" is this list being long.
    """
    return [element for element, decision in decisions.items() if decision.expand]


__all__ = [
    "CONTRADICTION",
    "CORPUS_MAJORITY",
    "CorpusZeroRate",
    "DEFAULT_REPORTS_DIR",
    "ExpansionDecider",
    "ExpansionReason",
    "MIN_COMPARABLE_REPORTS",
    "MIN_WEIGHT_SHARE",
    "MetricDecision",
    "SECTOR_MISMATCH",
    "ScoredCorpus",
    "SectionDecision",
    "TRIGGERS",
    "TRIGGER_LABELS",
    "ZERO_SCORE_GAP",
    "ZERO_SCORE_NOT_COMPARABLE",
    "expanded_sections",
    "load_scored_corpus",
]
