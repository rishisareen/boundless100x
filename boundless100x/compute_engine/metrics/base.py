"""MetricResult — the universal return type for all metric functions."""

from dataclasses import dataclass, field


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
