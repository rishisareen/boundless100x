"""The chart layer's one contract: a builder that cannot draw returns `""`.

Both halves of that are tested here, because a CAPLIPOINT run on 2026-08-12
proved they are different guarantees. The builders honour it for the gaps they
*anticipate* — a missing column, an empty frame — and `pe_band_chart_historical`
still raised `NotImplementedError` out of pandas on a fiscal-year-transition
stub, taking down a report whose JSON exports, copied annual reports and paid
two-pass LLM analysis were already on disk.

So: the parsing bug that raised, and the fence that stops the next one from
costing the document.
"""

import re

import pytest

from boundless100x.output import report_charts
from boundless100x.output.report_charts import pe_band_chart_historical, render_charts
from tests.conftest import make_result, year_labels

# The label a company that moved its fiscal year end actually gets. Caplin
# Point went from a June to a March year end, and Screener renders the 9-month
# stub as `Mar 20169m` — which starts with "Mar", so the annual filter admits
# it exactly where it correctly drops `TTM`, and then `%b %Y` parses nothing.
TRANSITION_STUB = "Mar 20169m"

# Which key in `render_charts`'s dict each builder is responsible for.
PANEL_OF = {
    "roce_trend_chart": "roce_trend",
    "pe_band_chart": "pe_band",
    "growth_chart": "growth",
    "dcf_visualization": "dcf_gauge",
    "cashflow_quality_chart": "cashflow_quality",
    "pe_band_chart_historical": "pe_band_historical",
}


def fingerprint(html: str) -> str:
    """A rendered figure with Plotly's per-render uuids normalised away.

    The payload runs to hundreds of KB and every render stamps fresh ids, so
    this is what lets two figures be compared for "drew the same thing" without
    asserting on Plotly's serialisation.
    """
    return re.sub(r"[0-9a-f]{8}-[0-9a-f-]{27}", "", html)


def explode(*args, **kwargs):
    raise NotImplementedError("pandas said no")


class TestFiscalYearTransitionStub:
    """`Mar 20169m` reaching `interpolate(method="time")` as a `NaT`."""

    def test_the_chart_still_draws(self):
        """The regression, at its narrowest: this raised."""
        labels = year_labels(10)
        labels[1] = TRANSITION_STUB

        assert pe_band_chart_historical(make_result(financials={"year": labels}))

    def test_the_stub_row_is_dropped_not_coerced_to_a_nearby_date(self):
        """The stubbed frame must draw *the frame without that row* — one
        unparseable label may cost its own year and must not move the bands the
        remaining years sit between.

        Both sides fix `eps` explicitly so the only difference between them is
        the stub row itself, rather than `make_financials`' compounding series
        being regenerated at a different length.
        """
        eps = [float(v) for v in range(10, 20)]
        stubbed_labels = year_labels(10)
        stubbed_labels[1] = TRANSITION_STUB

        stubbed = pe_band_chart_historical(
            make_result(financials={"year": stubbed_labels, "eps": eps})
        )
        # The same data with the stub row genuinely absent.
        clean = pe_band_chart_historical(
            make_result(
                n=9,
                financials={
                    "year": year_labels(10)[:1] + year_labels(10)[2:],
                    "eps": eps[:1] + eps[2:],
                },
            )
        )

        assert stubbed and clean
        assert fingerprint(stubbed) == fingerprint(clean)

    def test_a_frame_that_is_all_stub_returns_empty_not_an_exception(self):
        """Fewer than three parseable year ends is a data gap, and a data gap
        is the empty string. The count is re-checked *after* the drop — before
        it, ten stub rows look like ten usable ones."""
        result = make_result(financials={"year": [TRANSITION_STUB] * 10})

        assert pe_band_chart_historical(result) == ""

    def test_ttm_was_never_the_problem(self):
        """`TTM` does not start with "Mar", so the annual filter already
        removed it. Pinned so a rewrite of that filter cannot reintroduce the
        `NaT` from the other end — the corpus has `TTM` on most tickers."""
        labels = year_labels(10) + ["TTM"]

        assert pe_band_chart_historical(
            make_result(n=11, financials={"year": labels})
        )


class TestOneBuilderCannotCostTheDocument:
    """The fence: every builder is called through `_drawn`."""

    @pytest.mark.parametrize("builder", sorted(PANEL_OF))
    def test_a_raising_builder_becomes_a_missing_panel(self, builder, monkeypatch):
        monkeypatch.setattr(report_charts, builder, explode)

        charts = render_charts(make_result())

        # Not raising is the point; the panel it owns is absent or empty.
        assert not charts.get(PANEL_OF[builder])

    @pytest.mark.parametrize("builder", sorted(PANEL_OF))
    def test_every_other_panel_still_draws(self, builder, monkeypatch):
        """A degraded report is still a report. Compared against the panels
        that were actually drawing on this fixture rather than against a count,
        since several builders legitimately return `""` on it."""
        healthy = {k for k, v in render_charts(make_result()).items() if v}

        monkeypatch.setattr(report_charts, builder, explode)
        degraded = {k for k, v in render_charts(make_result()).items() if v}

        assert degraded == healthy - {PANEL_OF[builder]}

    def test_the_failure_is_logged_not_swallowed(self, monkeypatch, caplog):
        """A chart that silently stops drawing is a defect that hides."""
        monkeypatch.setattr(report_charts, "pe_band_chart_historical", explode)

        with caplog.at_level("ERROR"):
            render_charts(make_result())

        assert "pe_band_historical" in caplog.text
        assert "pandas said no" in caplog.text
