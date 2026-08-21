"""MetricResult — the universal return type for all metric functions."""

from dataclasses import dataclass, field

# Flags whose presence means the value, though computed, is not evidence.
#
# **A number can be arithmetically correct and still not be a reading.** A CAGR
# from a base of ₹45 Cr to ₹3,521 Cr is 328%/yr and is not a growth rate; it is
# the arithmetic of a company that did not exist at the start of the window.
# Left alone, JIOFIN scored a perfect 1.0 on six such metrics — 47% of its
# Growth element — and its trailing PEG of 0.29x, 38% of all scored Price
# weight, came off the same base. The report then printed "the golden rule for
# 100-baggers... the valuation appears justified and attractive."
#
# The value is kept and shown, because "revenue went from 45 to 3,521" is worth
# knowing. What is withdrawn is its vote: the scorer waives it, and
# `EligibilityEvaluator` refuses to gate on it. **Both, or neither** — a figure
# too unreliable to score is too unreliable to admit a company through a 100x
# gate, and the two layers reading one list is what keeps them from drifting.
#
# This lives here because `base.py` is the leaf both layers already import;
# `compute_engine/eligibility.py` and `compute_engine/scorer.py` may not
# import each other.
UNSCORABLE_FLAGS = frozenset({
    "cagr_off_negligible_base",
})


def is_scorable(result) -> bool:
    """Whether a computed result may vote — in a score or in a gate."""
    flags = getattr(result, "flags", None) or ()
    return not any(flag in UNSCORABLE_FLAGS for flag in flags)


@dataclass
class MetricResult:
    """Every compute function returns this.

    Attributes:
        value: The computed number (None if data unavailable).
        raw_series: Optional yearly/quarterly values for trend display.
        flags: Qualitative flags for LLM context (e.g., "consistently_high_roce").
        metadata: Debug info, years used, intermediate calculations.
        error: Error message if computation failed.
    """

    value: float | str | None = None
    raw_series: list[float] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    error: str | None = None

    @property
    def ok(self) -> bool:
        """True if computation succeeded and produced a value."""
        return self.value is not None and self.error is None
