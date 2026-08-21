"""Everything this system knows about a company from its sector name.

Two declarations, two questions, one set of matching rules.

`data_fetcher/sector_context.yaml` lists the sectors that produced compounders
in the NTD era and the ones the Dec 2025 Wealth Creation Study rules out —
answering "is this a sector worth owning?". `sector_applicability.yaml`, beside
this module, declares which metrics measure nothing useful for which kind of
company — answering "does this reading mean anything here?". This module is the
single reader of both, used by the scored metric, the LLM prompt context, and
the report's expansion decision.

They are deliberately kept apart. A tailwind bucket is a claim about
compounding history and says nothing about whether a ratio is measuring what it
claims to; keying applicability on the bucket would have asserted that every
strong-tailwind sector shares a balance sheet, which is false the moment
"Finance" and "Healthcare" sit in the same bucket. What the two share is the
*name matching* — `_matches` below — so a company in a known sector inherits
both kinds of rule with no new entry anywhere (KTD6).
"""

import logging
import re
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_PATH = (
    Path(__file__).parent.parent / "data_fetcher" / "sector_context.yaml"
)
DEFAULT_APPLICABILITY_PATH = Path(__file__).parent / "sector_applicability.yaml"

STRONG = "strong_tailwind"
MODERATE = "moderate_tailwind"
NON_CONSIDERATION = "non_consideration"
UNKNOWN = "unknown"

# The applicability verdict vocabulary — three-valued, like every other
# evaluator in this system. `INDETERMINATE` is not a formality here: it is the
# answer for every sector nobody has reviewed, which is most of them.
APPLIES = "applies"
DOES_NOT_APPLY = "does_not_apply"
INDETERMINATE = "indeterminate"
APPLICABILITY_VERDICTS = (APPLIES, DOES_NOT_APPLY, INDETERMINATE)

# Keys a sector entry may carry. An allowlist rather than a "check the ones we
# know" pass, because the failure it prevents is silent: a typo'd
# `not_applicible:` would leave the sector marked reviewed with nothing
# excluded, which reads as "every metric applies here" — the exact wrong answer
# this table exists to prevent, arrived at by a spelling mistake.
_SECTOR_ENTRY_KEYS = frozenset({"label", "not_applicable"})

# Keys an individual exclusion may carry when written in its long form. The
# short form — `metric_id: "reason"` — is equivalent to `{reason: "..."}` with
# flags suppressed. Same allowlist reasoning as `_SECTOR_ENTRY_KEYS`: a
# misspelled `keep_flag:` would silently suppress a warning somebody meant to
# keep, and nothing at runtime would say so.
_EXCLUSION_KEYS = frozenset({"reason", "keep_flags"})


@lru_cache(maxsize=4)
def load_sector_context(path: str | None = None) -> dict:
    """Load the sector lists. Returns empty buckets if the file is unreadable."""
    target = Path(path) if path else DEFAULT_CONTEXT_PATH
    empty = {STRONG: [], MODERATE: [], NON_CONSIDERATION: [], "raw": {}}

    try:
        raw = yaml.safe_load(target.read_text()) or {}
    except (OSError, yaml.YAMLError) as exc:
        logger.warning(f"Could not read sector context at {target}: {exc}")
        return empty

    buckets = raw.get("mtd_consideration_sectors", {}) or {}
    return {
        STRONG: list(buckets.get("strong_tailwind", []) or []),
        MODERATE: list(buckets.get("moderate_tailwind", []) or []),
        NON_CONSIDERATION: list(buckets.get("non_consideration", []) or []),
        "raw": raw,
    }


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _matches(sector: str, listed: str) -> bool:
    """Whole-phrase match, tolerant of a trailing plural 's'.

    Word boundaries stop a short code like 'IT' hitting 'Securities'; the
    optional plural lets the study's 'Capital Market' match Screener's actual
    breadcrumb label 'Capital Markets'.
    """
    sector_n, listed_n = _normalise(sector), _normalise(listed)
    if sector_n == listed_n:
        return True
    return re.search(rf"(?<!\w){re.escape(listed_n)}s?(?!\w)", sector_n) is not None


def classify_sector(sector: str | None, context: dict | None = None) -> str:
    """Map a sector name onto its study bucket, or `unknown` when unlisted."""
    if not sector or not str(sector).strip():
        return UNKNOWN

    ctx = context or load_sector_context()
    for bucket in (STRONG, MODERATE, NON_CONSIDERATION):
        if any(_matches(str(sector), listed) for listed in ctx.get(bucket, [])):
            return bucket
    return UNKNOWN


# ── Group structure ───────────────────────────────────────────────────────
#
# Screener's `sector_industry` breadcrumb, already fetched and stored on every
# metadata.json, names a holding company outright. Nothing read it, so a
# conglomerate's consolidated ratios were interpreted as one business's.
#
# The reading this most distorts is growth. EDELWEISS consolidates a lending
# book being deliberately run down (₹18,000 Cr to ₹600 Cr) with fee businesses
# compounding 27-63%; blended, that is 2.6% revenue CAGR, and the model called
# it a Growth Trap with negative real growth. Both halves of the blend are
# real and the average describes neither.
#
# Matched as substrings on a lowercased breadcrumb rather than through
# `_matches` above: that function answers "is this company in sector X", where
# whole-phrase matching stops `IT` hitting `Securities`. This asks a different
# question — does the industry label contain a structural word — and
# `Holding Company` appears inside longer labels.
_STRUCTURE_MARKERS = (
    ("holding company", "a holding company"),
    ("conglomerate", "a conglomerate"),
    ("diversified", "a diversified group"),
    ("investment company", "an investment company"),
)


def group_structure(metadata: dict | None) -> dict:
    """Whether this company's reported figures blend distinct businesses.

    Returns `{"is_group": bool, "label": str, "industry": str | None}`.
    False for an unlabelled company, which is a statement about the label
    rather than about the company — plenty of conglomerates are filed under an
    operating industry, and this cannot see those.
    """
    meta = metadata or {}
    industry = meta.get("sector_industry")
    outcome = {"is_group": False, "label": "", "industry": industry}
    if not industry or not str(industry).strip():
        return outcome

    lowered = _normalise(str(industry))
    for marker, label in _STRUCTURE_MARKERS:
        if marker in lowered:
            outcome["is_group"] = True
            outcome["label"] = label
            return outcome
    return outcome


def applicability_labels(metadata: dict | None) -> tuple[str, ...]:
    """The labels an applicability lookup should be made against.

    The broad `sector` breadcrumb and, when present, the narrower
    `sector_industry`. Both describe the same company, so a rule written about
    either applies — which is what lets an entry be keyed on "Investment
    Company" when every such company's *sector* reads "Finance". One place
    rather than at each call site, because the scorer and the reading layer
    must ask the identical question or the report will describe a metric the
    composite did not withhold.
    """
    meta = metadata or {}
    labels = [meta.get("sector"), meta.get("sector_industry")]
    return tuple(str(x) for x in labels if x is not None and str(x).strip())


def structure_caveat(metadata: dict | None) -> str:
    """The sentence a reader needs before trusting a consolidated ratio.

    Empty when the company is not labelled as a group, so callers can append
    it unconditionally.
    """
    structure = group_structure(metadata)
    if not structure["is_group"]:
        return ""
    return (
        f"GROUP STRUCTURE: Screener files this company under "
        f"'{structure['industry']}', so it is {structure['label']} and every "
        f"figure below is a consolidation of businesses that may be growing "
        f"at very different rates and earning very different returns. A "
        f"blended revenue or margin here describes no individual business, "
        f"and a segment being wound down deliberately will read as "
        f"stagnation. Treat consolidated growth and quality readings as a "
        f"starting question rather than a finding, and look for the segment "
        f"disclosures before concluding the company is not growing."
    )


def study_findings(context: dict | None = None) -> dict:
    """The business-type and leadership findings that sit alongside the lists."""
    raw = (context or load_sector_context()).get("raw", {})
    return {
        "business_type": raw.get("business_type_preference", {}) or {},
        "leadership": raw.get("market_leadership", {}) or {},
    }


# ── Sector-by-metric applicability ────────────────────────────────────────


@lru_cache(maxsize=4)
def load_sector_applicability(path: str | None = None) -> dict:
    """Read the declared applicability table. Returns the `sectors` mapping.

    Returned exactly as written, not normalised, so
    `validate_sector_applicability` can name a malformed entry rather than
    quietly smoothing it into something that loads.

    An unreadable file degrades to `{}`, the same posture
    `load_sector_context` takes above — and here the degradation is safe in a
    way worth stating: with no sector reviewed, every lookup reads
    indeterminate (R4). That costs the expansion trigger its signal, loudly in
    the log, but it never asserts that a metric fits a company when nobody
    said so. The opposite default would fail silently and in the dangerous
    direction.
    """
    target = Path(path) if path else DEFAULT_APPLICABILITY_PATH

    try:
        raw = yaml.safe_load(target.read_text()) or {}
    except (OSError, yaml.YAMLError) as exc:
        logger.warning(f"Could not read sector applicability at {target}: {exc}")
        return {}

    return raw.get("sectors", {}) or {}


def validate_sector_applicability(
    table: dict, known_metric_ids: Iterable[str]
) -> list[str]:
    """Return a list of table errors — empty when the table is sound.

    Startup validation exists because every failure it catches is invisible at
    runtime. A metric id the registry does not define would sit in the table
    forever without ever matching a computed metric, and a rule that can never
    fire looks exactly like a rule whose condition is never met. A reason left
    blank renders as an expanded section that says a metric does not apply and
    never says why, which is R7's failure rather than R6's — but it reaches the
    reader as the same shrug.
    """
    known = set(known_metric_ids)
    errors: list[str] = []

    if not isinstance(table, dict):
        return [f"sectors must be a mapping, got {type(table).__name__}"]

    for sector, entry in table.items():
        if not isinstance(sector, str) or not sector.strip():
            errors.append(f"sector key {sector!r} must be a non-empty string")
            continue
        if not isinstance(entry, dict):
            errors.append(f"{sector}: entry must be a mapping")
            continue

        unknown_keys = sorted(set(entry) - _SECTOR_ENTRY_KEYS)
        if unknown_keys:
            errors.append(
                f"{sector}: unknown key(s) {', '.join(unknown_keys)} — "
                f"expected {', '.join(sorted(_SECTOR_ENTRY_KEYS))}"
            )

        excluded = entry.get("not_applicable")
        if excluded is None:
            continue
        if not isinstance(excluded, dict):
            errors.append(
                f"{sector}.not_applicable must be a mapping of metric id to "
                f"reason, got {type(excluded).__name__}"
            )
            continue

        for metric_id, declared in excluded.items():
            where = f"{sector}.not_applicable.{metric_id}"
            if metric_id not in known:
                errors.append(f"{where}: unknown metric id {metric_id!r}")

            # Two declaration forms. A bare string is the reason and suppresses
            # the metric's flags; a mapping carries the same reason and may add
            # `keep_flags: true`. Both are validated for the reason, because
            # that string is what R7 puts in front of the reader either way.
            if isinstance(declared, dict):
                unknown_keys = sorted(set(declared) - _EXCLUSION_KEYS)
                if unknown_keys:
                    errors.append(
                        f"{where}: unknown key(s) {', '.join(unknown_keys)} — "
                        f"expected {', '.join(sorted(_EXCLUSION_KEYS))}"
                    )
                if "keep_flags" in declared and not isinstance(
                    declared["keep_flags"], bool
                ):
                    errors.append(
                        f"{where}.keep_flags must be true or false, not "
                        f"{declared['keep_flags']!r}"
                    )
                reason = declared.get("reason")
            else:
                reason = declared

            if not isinstance(reason, str) or not reason.strip():
                errors.append(
                    f"{where}: needs a reason a reader can act on, not "
                    f"{reason!r}"
                )

    return errors


class SectorApplicability:
    """Answers "does this metric mean anything for a company of this kind?".

    The fourth three-valued evaluator in the system, and it inherits the rule
    that carries the weight in the other three: **indeterminate, never a silent
    pass.** `EligibilityEvaluator` refuses to call a gate met when it could not
    read the metric; this refuses to call a metric applicable when nobody has
    reviewed the sector. The asymmetry matters more here than the symmetry —
    "applies" is the answer that lets a lender be marked down for lending, so
    it is the one that has to be earned.

    `known_metric_ids` is **required**, and positional, on purpose. Its sibling
    `LaneGateEvaluator` takes the same argument optionally, and CLAUDE.md
    carries the resulting warning: construct it without and the unknown-metric
    check silently never runs. There is no reason to repeat that here — every
    real caller holds a `ComputeEngine` and can pass `engine.metrics`. Making
    it required is what puts the check somewhere it can see both halves: the
    table is loaded in `compute_engine/sector.py`, which the metrics import and
    which therefore can never import the engine, so the registry has to arrive
    from the caller rather than be fetched from here.
    """

    def __init__(self, known_metric_ids: Iterable[str], table: dict | None = None):
        self.known_metric_ids = frozenset(known_metric_ids)
        self.table = load_sector_applicability() if table is None else table

        errors = validate_sector_applicability(self.table, self.known_metric_ids)
        if errors:
            for error in errors:
                logger.error(f"  SECTOR APPLICABILITY ERROR: {error}")
            # The errors travel in the message as well as the log, matching
            # `LaneGateEvaluator`: this is also constructed from caller-supplied
            # tables in tests, where a bare count leaves nothing to act on.
            raise ValueError(
                f"Sector applicability validation failed: {len(errors)} errors — "
                f"{'; '.join(errors)}"
            )

        # A company's sector does not change while its report renders, but
        # `evaluate` is asked once per metric — 114 matches for one company
        # against a table this instance cannot mutate. Memoised per instance
        # rather than with `lru_cache`, which would keep `self` alive in a
        # module-level cache; keyed on the sector string, so two companies can
        # never share an answer that was not already identical by definition.
        self._matches_cache: dict[tuple[str, ...], list[str]] = {}

    @staticmethod
    def _labels(sector) -> tuple[str, ...]:
        """Normalise one label or several into the tuple the matcher walks.

        Callers pass a company's `sector` breadcrumb and, where it has one, its
        narrower `sector_industry`. Both describe the same company and both may
        carry rules, so both are matched and their exclusions merge — that is
        what lets an entry be written about "Investment Company" when every
        such company's *sector* reads "Finance".
        """
        if sector is None:
            return ()
        if isinstance(sector, str):
            candidates = [sector]
        else:
            candidates = list(sector)
        return tuple(
            str(label) for label in candidates
            if label is not None and str(label).strip()
        )

    def matching_sectors(self, sector) -> list[str]:
        """Every declared sector key this company falls under.

        Shortest key first. Two keys can legitimately match one company — a
        future "Housing Finance" entry beside "Finance" is true of the same
        housing financier, and "Investment Company" is true of a lender that is
        also a holding vehicle — and both claims hold, so their exclusions
        merge. Ordering by length means the narrower entry's wording wins where
        both name the same metric, since it was written about this kind of
        company specifically.
        """
        labels = self._labels(sector)
        if not labels:
            return []

        if labels not in self._matches_cache:
            matched = {
                declared
                for declared in self.table
                for label in labels
                if _matches(label, declared)
            }
            self._matches_cache[labels] = sorted(
                matched,
                key=lambda declared: (len(_normalise(declared)), declared),
            )
        # A copy: the list is handed to callers and reaches `matched_sectors`
        # on every outcome, and a shared mutable would let one reader's edit
        # rewrite what the next one is told.
        return list(self._matches_cache[labels])

    @staticmethod
    def _entry_reason(declared) -> str:
        """The reason out of either declaration form. See `_SECTOR_ENTRY_KEYS`."""
        if isinstance(declared, dict):
            return str(declared.get("reason", ""))
        return str(declared)

    @staticmethod
    def _entry_keeps_flags(declared) -> bool:
        """Whether this metric's flags survive being declared inapplicable.

        **Suppression is the default, and the asymmetry is the point.** A
        metric that measures nothing here says nothing here — so
        `cash_conversion`, withdrawn from a lender's score, must not go on
        emitting `cash_cow` into the report's Strengths list. JIOFIN carried
        exactly that: "Cash Cow — Strong Cash Conversion", top of the
        strengths, on a company whose operating cash flow was -₹15,439 Cr.

        The exception is a metric whose *reading* is sound and whose
        *threshold* is not. Debt/equity of 4x is 4x however you score it, and
        `debt_risk` is a real warning that a lender's owner still needs; only
        the mark out of ten was calibrated for a manufacturer. Those entries
        say `keep_flags: true` and have to argue for it.
        """
        return bool(declared.get("keep_flags")) if isinstance(declared, dict) else False

    def _declarations(self, sector: str | None,
                      matched: list[str] | None = None) -> dict:
        merged: dict = {}
        for key in (self.matching_sectors(sector) if matched is None else matched):
            merged.update((self.table[key] or {}).get("not_applicable") or {})
        return merged

    def not_applicable_metrics(self, sector: str | None,
                               matched: list[str] | None = None) -> dict[str, str]:
        """The metrics declared inapplicable to this sector, with their reasons.

        Empty for a sector nobody has reviewed — which is not the same claim as
        "everything applies here". Ask `evaluate` when the difference matters.

        `matched` lets a caller that has already resolved the sector pass it in
        rather than have it resolved twice in one call.
        """
        return {
            metric_id: self._entry_reason(declared)
            for metric_id, declared in self._declarations(sector, matched).items()
        }

    def flag_suppressed_metrics(self, sector: str | None,
                                matched: list[str] | None = None) -> set[str]:
        """Inapplicable metrics whose flags must not reach a reader either."""
        return {
            metric_id
            for metric_id, declared in self._declarations(sector, matched).items()
            if not self._entry_keeps_flags(declared)
        }

    def evaluate(self, metric_id: str, sector: str | None) -> dict:
        """Whether `metric_id` measures anything for a company in `sector`.

        Returns `applies` as a True/False/None tri-state beside the verdict
        word, mirroring `EligibilityEvaluator`'s `eligible`, plus the reason —
        which for a declared exclusion is the table's own sentence, shown to
        the reader verbatim under R7.
        """
        outcome = {
            "metric": metric_id,
            "sector": sector,
            "applies": None,
            "verdict": INDETERMINATE,
            "reason": "",
            "matched_sectors": [],
        }

        if metric_id not in self.known_metric_ids:
            # A caller asking about something the registry does not define. Not
            # a reader-facing path — but answering "applies" would be a
            # guess dressed as a finding, so it reads indeterminate like
            # everything else that could not be checked.
            outcome["reason"] = (
                f"'{metric_id}' is not a metric this registry computes, so its "
                f"applicability could not be judged"
            )
            return outcome

        if not self._labels(sector):
            outcome["reason"] = (
                "No sector is recorded for this company, so whether its metrics "
                "fit could not be judged — refetch to pick up the sector "
                "breadcrumb"
            )
            return outcome

        matched = self.matching_sectors(sector)
        outcome["matched_sectors"] = matched
        if not matched:
            outcome["reason"] = (
                f"'{', '.join(self._labels(sector))}' has not been reviewed "
                f"against the metric set, so whether this metric measures "
                f"anything here is unknown"
            )
            return outcome

        excluded = self.not_applicable_metrics(sector, matched)
        if metric_id in excluded:
            outcome["applies"] = False
            outcome["verdict"] = DOES_NOT_APPLY
            outcome["reason"] = excluded[metric_id]
            return outcome

        # Reviewed, and not excluded. A sector key present in the table is a
        # claim that somebody read the metric set against this kind of company,
        # so silence about a metric is a positive finding rather than an
        # absence of one. That is exactly why an *unlisted* sector cannot
        # reach here: there, silence means nobody looked.
        outcome["applies"] = True
        outcome["verdict"] = APPLIES
        outcome["reason"] = (
            f"{matched[-1]} has been reviewed against the metric set and this "
            f"metric was not excluded from it"
        )
        return outcome
