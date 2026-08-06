"""The four forward-growth sub-metrics.

Three read the extraction pass's output; the fourth is fully offline from the
quarterly series. All four carry zero weight in an element deliberately absent
from `element_weights`, so most of what follows is about the two ways they can
lie: reading a value where the source section was never usable (R5), and
moving a composite they must not touch (R7).

The assertion doing the most work is the steady-growth one. A single YoY figure
is a growth *level* — it answers "is this growing", not "is growth speeding
up" — and a company compounding a constant 20% would report +20 momentum under
that reading while accelerating not at all.
"""

import shutil

import pandas as pd
import pytest
import yaml

from boundless100x import forward_growth_schema as schema
from boundless100x.compute_engine.engine import ComputeEngine
from boundless100x.compute_engine.metrics.builtin import forward_growth as fgm
from boundless100x.compute_engine.scorer import SQGLPScorer
from tests.conftest import make_data, make_financials, make_quarterly

SUB_METRICS = (
    "promises_kept_ratio", "capex_pipeline", "tam_runway", "quarterly_momentum",
)
TEXT_DERIVED = ("promises_kept_ratio", "capex_pipeline", "tam_runway")

FUNCTIONS = {
    "promises_kept_ratio": fgm.compute_promises_kept,
    "capex_pipeline": fgm.compute_capex_pipeline,
    "tam_runway": fgm.compute_tam_runway,
}


def promise(metric="revenue", target_value=300.0, target_period="FY2025",
            section="mdna", **extra):
    entry = {
        "metric": metric,
        "target_value": target_value,
        "target_period": target_period,
        "source_sentence": f"We target {target_value} in {target_period}.",
        "section": section,
    }
    entry.update(extra)
    return entry


def capex(amount=500.0, year="FY2027", section="mdna"):
    return {
        "amount_inr_cr": amount,
        "commissioning_year": year,
        "source_sentence": f"A new plant of Rs {amount} crore commissions in {year}.",
        "section": section,
    }


def tam(size=50000.0, section="mdna"):
    return {
        "market_size_inr_cr": size,
        "source_sentence": f"The addressable market is Rs {size} crore.",
        "section": section,
    }


def year_payload(sections=None, guidance=(), capex_entries=(), tam_entries=()):
    return {
        "sections": sections if sections is not None else {
            "mdna": schema.FOUND, "chairman": schema.FOUND,
        },
        "guidance": list(guidance),
        "capex": list(capex_entries),
        "tam": list(tam_entries),
    }


def fg_data(forward_growth=None, revenue=None, n=3, **kwargs):
    """A data dict carrying extraction output and a short, explicit P&L."""
    revenue = revenue if revenue is not None else [100.0, 200.0, 300.0]
    data = make_data(n=n, **kwargs)
    data["financials"] = make_financials(n, revenue=revenue)
    if forward_growth is not None:
        data["forward_growth"] = forward_growth
    return data


# ── R5: a section that was never usable yields no value ────────────────────


class TestProvenanceGatesEverySubMetric:
    @pytest.mark.parametrize("metric_id", TEXT_DERIVED)
    def test_a_fallback_required_section_reads_indeterminate(self, metric_id):
        data = fg_data({"2025": year_payload(
            sections={"mdna": schema.FALLBACK, "chairman": schema.FALLBACK},
            guidance=[promise()], capex_entries=[capex()], tam_entries=[tam()],
        )})
        result = FUNCTIONS[metric_id](data, {})

        assert not result.ok
        assert "fallback" in result.error or "not usable" in result.error

    @pytest.mark.parametrize("metric_id", TEXT_DERIVED)
    def test_a_suspect_required_section_reads_indeterminate(self, metric_id):
        """`suspect` is treated exactly as `fallback` (KTD9)."""
        data = fg_data({"2025": year_payload(
            sections={"mdna": schema.SUSPECT, "chairman": schema.SUSPECT},
            guidance=[promise()], capex_entries=[capex()], tam_entries=[tam()],
        )})
        assert not FUNCTIONS[metric_id](data, {}).ok

    def test_mixed_provenance_still_blocks_promises_kept(self):
        """The case in 10 of 29 real report-years; a year-keyed check misses it."""
        data = fg_data({"2025": year_payload(
            sections={"mdna": schema.FALLBACK, "chairman": schema.FOUND},
            guidance=[promise()],
        )})
        assert not fgm.compute_promises_kept(data, {}).ok

    def test_tam_runway_accepts_the_chairman_section_as_a_ranked_fallback(self):
        """The one sub-metric whose claim is qualitative anyway (KTD4)."""
        data = fg_data({"2025": year_payload(
            sections={"mdna": schema.FALLBACK, "chairman": schema.FOUND},
            tam_entries=[tam(section="chairman")],
        )})
        assert fgm.compute_tam_runway(data, {}).ok

    def test_an_entry_from_an_unusable_section_is_not_counted(self):
        """Provenance is per section, and so is the filter."""
        data = fg_data({"2025": year_payload(
            sections={"mdna": schema.FOUND, "chairman": schema.FALLBACK},
            tam_entries=[tam(size=99999.0, section="chairman"), tam(size=50000.0)],
        )})
        result = fgm.compute_tam_runway(data, {})

        assert result.metadata["tam_inr_cr"] == 50000.0


class TestAbsentAndEmptyInput:
    @pytest.mark.parametrize("metric_id", TEXT_DERIVED)
    def test_an_absent_key_errors_rather_than_delegating_to_the_engine(self, metric_id):
        assert not FUNCTIONS[metric_id](fg_data(), {}).ok

    @pytest.mark.parametrize("metric_id", TEXT_DERIVED)
    def test_a_present_but_empty_dict_errors(self, metric_id):
        """The engine's input check passes a present-but-empty dict straight through."""
        assert not FUNCTIONS[metric_id](fg_data({}), {}).ok

    @pytest.mark.parametrize("metric_id", TEXT_DERIVED)
    def test_a_year_with_no_entries_of_this_kind_errors(self, metric_id):
        data = fg_data({"2025": year_payload()})
        assert not FUNCTIONS[metric_id](data, {}).ok


# ── promises_kept_ratio ────────────────────────────────────────────────────


class TestPromisesKept:
    def two_years(self, guidance_2024, guidance_2025):
        return fg_data({
            "2024": year_payload(guidance=guidance_2024),
            "2025": year_payload(guidance=guidance_2025),
        })

    def test_guidance_met_yields_a_high_ratio(self):
        data = self.two_years(
            [promise(target_value=200.0, target_period="FY2024")],
            [promise(target_value=300.0, target_period="FY2025")],
        )
        result = fgm.compute_promises_kept(data, {})

        assert result.ok
        assert result.value == 100.0
        assert result.metadata["due"] == 2
        assert result.metadata["kept"] == 2

    def test_guidance_missed_yields_a_low_ratio(self):
        data = self.two_years(
            [promise(target_value=400.0, target_period="FY2024")],
            [promise(target_value=600.0, target_period="FY2025")],
        )
        result = fgm.compute_promises_kept(data, {})

        assert result.value == 0.0
        assert result.metadata["due"] == 2

    def test_a_rounded_miss_inside_tolerance_still_counts_as_kept(self):
        """Indian guidance is often a rounded target; exact matching would read
        rounding as broken credibility."""
        data = self.two_years(
            [promise(target_value=205.0, target_period="FY2024")],   # 200/205 = 0.976
            [promise(target_value=330.0, target_period="FY2025")],   # 300/330 = 0.909
        )
        result = fgm.compute_promises_kept(data, {})

        assert result.metadata["kept"] == 1
        assert result.value == 50.0

    def test_the_lower_bound_of_a_range_is_the_promise(self):
        data = self.two_years(
            [promise(target_value=180.0, target_value_high=260.0, target_period="FY2024")],
            [promise(target_value=290.0, target_value_high=400.0, target_period="FY2025")],
        )
        assert fgm.compute_promises_kept(data, {}).value == 100.0

    def test_a_promise_whose_period_has_not_arrived_is_pending_not_missed(self):
        """Pending enters neither numerator nor denominator."""
        data = self.two_years(
            [promise(target_value=200.0, target_period="FY2024")],
            [promise(target_value=9999.0, target_period="FY2031")],
        )
        result = fgm.compute_promises_kept(data, {})

        assert result.metadata["due"] == 1
        assert result.metadata["pending"] == 1
        assert result.value == 100.0

    def test_an_unresolvable_target_period_is_discarded_not_guessed(self):
        data = self.two_years(
            [promise(target_value=200.0, target_period="FY2024")],
            [promise(target_value=300.0, target_period="the medium term")],
        )
        result = fgm.compute_promises_kept(data, {})

        assert result.metadata["discarded"] == 1
        assert result.metadata["due"] == 1

    def test_one_report_year_yields_indeterminate_not_a_perfect_score(self):
        """One year of guidance is not a credibility record (A5)."""
        data = fg_data({
            "2025": year_payload(guidance=[promise(target_value=300.0,
                                                   target_period="FY2025")]),
        })
        result = fgm.compute_promises_kept(data, {})

        assert not result.ok
        assert "report year" in result.error

    def test_no_due_promises_at_all_reads_indeterminate(self):
        """Zero kept out of zero due is not credibility; it is silence."""
        data = self.two_years(
            [promise(target_period="FY2030")],
            [promise(target_period="FY2031")],
        )
        assert not fgm.compute_promises_kept(data, {}).ok

    def test_a_margin_promise_settles_against_the_margin_column(self):
        data = self.two_years(
            [promise(metric="operating_margin_pct", target_value=24.0,
                     target_period="FY2024")],
            [promise(metric="operating_margin_pct", target_value=30.0,
                     target_period="FY2025")],
        )
        result = fgm.compute_promises_kept(data, {})

        # make_financials sets opm_pct to 25 throughout: 25/24 kept, 25/30 missed.
        assert result.metadata["kept"] == 1

    def test_every_settlement_records_what_it_was_settled_against(self):
        data = self.two_years(
            [promise(target_value=200.0, target_period="FY2024")],
            [promise(target_value=300.0, target_period="FY2025")],
        )
        settled = fgm.compute_promises_kept(data, {}).metadata["settled"]

        assert all(s["settled_against"] == "financials.revenue" for s in settled)
        assert all(s["source_sentence"] for s in settled)


# ── capex_pipeline ─────────────────────────────────────────────────────────


class TestCapexPipeline:
    def test_announced_forward_capex_is_expressed_against_revenue(self):
        data = fg_data({"2025": year_payload(capex_entries=[capex(amount=150.0)])})
        result = fgm.compute_capex_pipeline(data, {})

        assert result.ok
        assert result.value == pytest.approx(50.0)  # 150 against 300 of revenue

    def test_already_commissioned_capex_is_not_forward_runway(self):
        data = fg_data({"2025": year_payload(
            capex_entries=[capex(amount=150.0, year="FY2020")]
        )})
        assert not fgm.compute_capex_pipeline(data, {}).ok

    def test_the_same_project_announced_twice_is_not_counted_twice(self):
        """Consecutive reports repeat a live project; double-counting it would
        make a company look like it was building twice as much."""
        data = fg_data({
            "2024": year_payload(capex_entries=[capex(amount=150.0, year="FY2027")]),
            "2025": year_payload(capex_entries=[capex(amount=150.0, year="FY2027")]),
        })
        result = fgm.compute_capex_pipeline(data, {})

        assert result.metadata["announced_inr_cr"] == 150.0

    def test_distinct_projects_are_summed(self):
        data = fg_data({"2025": year_payload(capex_entries=[
            capex(amount=150.0, year="FY2027"), capex(amount=90.0, year="FY2028"),
        ])})
        assert fgm.compute_capex_pipeline(data, {}).metadata["announced_inr_cr"] == 240.0

    def test_the_projects_travel_with_the_number(self):
        data = fg_data({"2025": year_payload(capex_entries=[capex(amount=150.0)])})
        projects = fgm.compute_capex_pipeline(data, {}).metadata["projects"]

        assert len(projects) == 1
        assert projects[0]["source_sentence"]


# ── tam_runway ─────────────────────────────────────────────────────────────


class TestTamRunway:
    def test_runway_is_the_years_of_current_growth_before_revenue_meets_the_market(self):
        data = fg_data(
            {"2025": year_payload(tam_entries=[tam(size=1200.0)])},
            revenue=[100.0, 200.0, 300.0],
        )
        result = fgm.compute_tam_runway(data, {})

        assert result.ok
        assert result.value > 0
        assert result.metadata["tam_inr_cr"] == 1200.0

    def test_a_bigger_market_leaves_a_longer_runway(self):
        small = fgm.compute_tam_runway(
            fg_data({"2025": year_payload(tam_entries=[tam(size=600.0)])}), {}
        )
        large = fgm.compute_tam_runway(
            fg_data({"2025": year_payload(tam_entries=[tam(size=6000.0)])}), {}
        )
        assert large.value > small.value

    def test_revenue_already_at_the_stated_market_leaves_no_runway(self):
        data = fg_data({"2025": year_payload(tam_entries=[tam(size=250.0)])})
        result = fgm.compute_tam_runway(data, {})

        assert result.value == 0.0

    def test_a_non_positive_growth_rate_reads_indeterminate(self):
        data = fg_data(
            {"2025": year_payload(tam_entries=[tam(size=5000.0)])},
            revenue=[300.0, 200.0, 100.0],
        )
        assert not fgm.compute_tam_runway(data, {}).ok

    def test_an_enormous_runway_saturates_rather_than_reporting_a_fiction(self):
        data = fg_data(
            {"2025": year_payload(tam_entries=[tam(size=10_000_000.0)])},
            revenue=[299.0, 299.5, 300.0],
        )
        result = fgm.compute_tam_runway(data, {})

        assert result.value == result.metadata["cap_years"]
        assert result.metadata["saturated"] is True

    def test_the_largest_stated_market_in_the_newest_report_is_used(self):
        data = fg_data({
            "2024": year_payload(tam_entries=[tam(size=99999.0)]),
            "2025": year_payload(tam_entries=[tam(size=1200.0), tam(size=2000.0)]),
        })
        assert fgm.compute_tam_runway(data, {}).metadata["tam_inr_cr"] == 2000.0


# ── quarterly_momentum ─────────────────────────────────────────────────────


def quarterly_with_yoy(base: list[float], yoys: list[float]) -> pd.DataFrame:
    """A revenue series whose YoY change at each period is exactly `yoys[i]`."""
    values = list(base)
    for index, rate in enumerate(yoys):
        values.append(values[index] * (1 + rate))
    return make_quarterly(periods=len(values), revenue=values)


class TestQuarterlyMomentum:
    def momentum(self, frame, **params):
        return fgm.compute_quarterly_momentum({"quarterly": frame}, params)

    def test_steady_growth_yields_momentum_near_zero(self):
        """The assertion that separates a second difference from a growth rate.

        A company compounding a constant 20% is not accelerating. A single-YoY
        implementation reports +20 here and calls it momentum.
        """
        frame = quarterly_with_yoy([100.0] * 4, [0.20] * 8)
        result = self.momentum(frame)

        assert result.ok
        assert result.value == pytest.approx(0.0, abs=1e-6)

    def test_decelerating_growth_is_negative_even_while_growth_stays_positive(self):
        frame = quarterly_with_yoy([100.0] * 4, [0.30] * 5 + [0.30, 0.20, 0.12])
        result = self.momentum(frame)

        assert result.value < 0
        assert all(figure > 0 for figure in result.metadata["yoy_pct"])

    def test_accelerating_growth_is_positive(self):
        frame = quarterly_with_yoy([100.0] * 4, [0.05] * 5 + [0.10, 0.20, 0.32])
        assert self.momentum(frame).value > 0

    def test_it_compares_four_quarters_back_not_the_previous_quarter(self):
        """Seasonality must not read as a trend (Phase 1's checkpoint rule)."""
        frame = quarterly_with_yoy([100.0, 40.0, 160.0, 70.0], [0.10] * 8)
        result = self.momentum(frame)

        assert result.value == pytest.approx(0.0, abs=1e-6)
        assert all(f == pytest.approx(10.0) for f in result.metadata["yoy_pct"])

    def test_fewer_than_six_periods_reads_indeterminate(self):
        """Two YoY figures cannot be formed, so no second difference exists."""
        result = self.momentum(make_quarterly(periods=5))
        assert not result.ok
        assert "6" in result.error

    def test_six_periods_is_enough_for_one_second_difference(self):
        assert self.momentum(make_quarterly(periods=6)).ok

    def test_an_absent_quarterly_frame_errors_rather_than_reading_zero(self):
        assert not fgm.compute_quarterly_momentum({}, {}).ok
        assert not fgm.compute_quarterly_momentum({"quarterly": pd.DataFrame()}, {}).ok

    def test_a_zero_base_period_is_skipped_rather_than_dividing_by_zero(self):
        frame = make_quarterly(periods=10, revenue=[0.0] + [100.0] * 9)
        assert self.momentum(frame).ok

    def test_the_field_is_configurable(self):
        frame = quarterly_with_yoy([100.0] * 4, [0.20] * 8)
        assert self.momentum(frame, field="pat").ok

    def test_a_missing_column_errors(self):
        frame = make_quarterly(periods=10).drop(columns=["revenue"])
        assert not self.momentum(frame).ok


# ── R7 and registration ────────────────────────────────────────────────────


class TestRegistration:
    def test_all_four_are_registered_at_zero_weight(self):
        engine = ComputeEngine()
        for metric_id in SUB_METRICS:
            config = engine.metrics[metric_id]
            assert config["element"] == "forward_growth"
            assert config["scoring"]["weight"] == 0.0

    def test_the_forward_growth_element_carries_no_weight_in_the_registry(self):
        """Belt and braces: zero weight *and* an element nothing can score."""
        assert "forward_growth" not in ComputeEngine().element_weights

    def test_the_element_does_not_appear_in_the_element_scores(self):
        engine = ComputeEngine()
        scorer = SQGLPScorer(engine.metrics, engine.element_weights)
        scores = scorer.score(engine.run_all(make_data()))

        assert "forward_growth" not in scores["elements"]

    def test_each_text_derived_metric_declares_forward_growth_as_an_input(self):
        engine = ComputeEngine()
        for metric_id in TEXT_DERIVED:
            assert "forward_growth" in engine.metrics[metric_id]["inputs"]

    def test_all_four_appear_in_details_unscored(self):
        engine = ComputeEngine()
        scorer = SQGLPScorer(engine.metrics, engine.element_weights)
        details = scorer.score(engine.run_all(make_data()))["details"]

        for metric_id in SUB_METRICS:
            assert details[metric_id]["weight"] == 0
            assert details[metric_id]["score"] is None


class TestNonRegression:
    def test_registering_all_four_leaves_every_score_identical(self, tmp_path):
        shipped = ComputeEngine().registry_dir
        without = tmp_path / "without"
        shutil.copytree(shipped, without)
        (without / "elements" / "forward_growth.yaml").unlink()

        data = make_data()

        def score(engine):
            scorer = SQGLPScorer(
                engine.metrics, engine.element_weights,
                history_waiver_mcap=engine.master.get("history_waiver_mcap"),
            )
            return scorer.score(engine.run_all(data))

        before = score(ComputeEngine(registry_dir=str(without)))
        after = score(ComputeEngine())

        assert after["composite"] == before["composite"]
        assert after["elements"] == before["elements"]
        assert after["coverage"] == before["coverage"]

    def test_they_never_join_the_series_safe_allowlist(self):
        """KTD6: no Phase 2 metric earns a persist_years rule in this phase."""
        from boundless100x.lifecycle.evaluator import SERIES_SAFE_METRICS

        assert not set(SUB_METRICS) & SERIES_SAFE_METRICS
        assert "rerating_headroom" not in SERIES_SAFE_METRICS
