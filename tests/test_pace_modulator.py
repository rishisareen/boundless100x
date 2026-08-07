"""Deployment pace: deploy more cautiously when the whole corpus is expensive.

The failure mode that matters is macro leaking into per-name decisions. §11
allows the market's valuation to slow *entry* and nothing else — a company must
never be blocked from a kill-switch, an exit review, or an eligibility verdict
because the market is dear. So the assertions that carry this unit are the
negative ones: kill-switches fire identically under a compressed spread, gates
read identically, and only `→ probe` thresholds move.

The second is that this is a *regime* reading. The roadmap originally named
`earnings_yield_vs_gsec` as the input, but that metric is per-company — using
it would tighten entry when the company is expensive, which is the inverse of
the purpose and a second valuation test on a trigger that already tests
valuation (KTD7). One expensive name must not modulate the run.
"""

import json

import pytest

from boundless100x.compute_engine.engine import ComputeEngine
from boundless100x.lifecycle import pace
from boundless100x.lifecycle.advance import advance
from boundless100x.lifecycle.evaluator import TriggerEvaluator, load_triggers
from boundless100x.service import AnalysisResult
from boundless100x.watchlist import WatchlistManager
from tests.conftest import make_financials
from tests.test_lifecycle_advance import (
    StubService,
    fast_lane_entry,
    fast_lane_metrics,
    healthy_metrics,
    metric,
)


def reading(median=3.0, contributors=12):
    return {
        "median_pp": median,
        "contributors": contributors,
        "tickers": [f"T{i}" for i in range(contributors)],
    }


@pytest.fixture
def wm(tmp_path):
    return WatchlistManager(path=str(tmp_path / "watchlist.json"))


def watched(wm, ticker="ASTRAL", state="watch"):
    wm.add(ticker)
    wm.transition(ticker, state, "seed", evidence="test seed")
    return wm


# ── The corpus reading ─────────────────────────────────────────────────────


def corpus(tmp_path, pes: dict) -> str:
    """A raw_data directory with one metadata.json + financials.csv per ticker."""
    for ticker, pe in pes.items():
        directory = tmp_path / ticker
        directory.mkdir(parents=True, exist_ok=True)
        make_financials(5).to_csv(directory / "financials.csv", index=False)
        payload = {"name": ticker} if pe is None else {"name": ticker, "Stock P/E": pe}
        (directory / "metadata.json").write_text(json.dumps(payload))
    return str(tmp_path)


class TestCorpusSpread:
    def test_the_median_is_taken_over_every_cached_ticker(self, tmp_path):
        # Earnings yield 100/PE minus a 7% G-Sec: 20x -> -2, 10x -> +3, 25x -> -3.
        path = corpus(tmp_path, {"A": 20.0, "B": 10.0, "C": 25.0})
        result = pace.corpus_spread(path, macro={"gsec_yield_pct": 7.0})

        assert result["contributors"] == 3
        assert result["median_pp"] == pytest.approx(-2.0)

    def test_a_ticker_without_a_usable_multiple_does_not_contribute(self, tmp_path):
        path = corpus(tmp_path, {"A": 20.0, "B": None, "C": 10.0})
        result = pace.corpus_spread(path, macro={"gsec_yield_pct": 7.0})

        assert result["contributors"] == 2
        assert "B" not in result["tickers"]

    def test_a_directory_without_financials_is_not_a_ticker(self, tmp_path):
        path = corpus(tmp_path, {"A": 20.0})
        (tmp_path / "500001").mkdir()  # a BSE-code directory of annual reports
        (tmp_path / "500001" / "metadata.json").write_text(json.dumps({"Stock P/E": 5}))

        assert pace.corpus_spread(path, macro={})["contributors"] == 1

    def test_an_absent_corpus_reads_as_no_contributors(self, tmp_path):
        result = pace.corpus_spread(str(tmp_path / "nothing"), macro={})
        assert result["contributors"] == 0
        assert result["median_pp"] is None

    @pytest.mark.parametrize("payload", ["[]", "null", "42", '"text"'])
    def test_a_wrong_shaped_metadata_file_costs_one_contributor_not_the_run(
        self, tmp_path, payload
    ):
        """This runs once before advance()'s per-ticker isolation.

        Valid JSON of the wrong shape parses fine and then raises inside the
        metric, which would end the run for every tracked company — breaking
        the one guarantee advance() makes.
        """
        path = corpus(tmp_path, {"A": 20.0, "B": 10.0, "C": 25.0})
        broken = tmp_path / "BROKEN"
        broken.mkdir()
        make_financials(5).to_csv(broken / "financials.csv", index=False)
        (broken / "metadata.json").write_text(payload)

        result = pace.corpus_spread(path, macro={"gsec_yield_pct": 7.0})

        assert result["contributors"] == 3
        assert "BROKEN" not in result["tickers"]

    def test_an_unreadable_corpus_leaves_advance_running(self, tmp_path, wm):
        """Defence in depth: pace must cost the modulation, never the advance."""
        watched(wm)
        service = TestThroughAdvance().service(metrics=healthy_metrics())
        service.suite = type(
            "S", (), {"raw_data_dir": object()}  # explodes on Path()
        )()

        out = advance(service, wm)

        assert out["pace"]["applied"] is False
        assert "could not be resolved" in out["pace"]["reason"]
        assert len(out["outcomes"]) == 1

    def test_one_expensive_name_does_not_move_a_wide_median(self, tmp_path):
        """The assertion that separates a regime reading from a per-name test."""
        path = corpus(tmp_path, {
            "EXPENSIVE": 200.0, "A": 10.0, "B": 10.0, "C": 10.0, "D": 10.0,
        })
        assert pace.corpus_spread(path, macro={"gsec_yield_pct": 7.0})["median_pp"] > 0


# ── Modulation ─────────────────────────────────────────────────────────────


class TestModulation:
    def triggers(self):
        return load_triggers()

    def buy_zone(self, triggers):
        return triggers["valuation_buy_zone"]

    def thresholds(self, triggers):
        return [
            condition["threshold"]
            for condition in self.buy_zone(triggers)["conditions"]
            if "metric" in condition
        ]

    def test_a_wide_spread_leaves_thresholds_untouched(self):
        before = self.thresholds(self.triggers())
        modulated, decision = pace.modulate(self.triggers(), reading(median=3.0))

        assert decision["applied"] is False
        assert self.thresholds(modulated) == before

    def test_a_compressed_spread_tightens_entry_thresholds(self):
        before = self.thresholds(self.triggers())
        modulated, decision = pace.modulate(self.triggers(), reading(median=-4.0))

        assert decision["applied"] is True
        assert all(a < b for a, b in zip(self.thresholds(modulated), before))

    def test_the_declared_triggers_are_not_mutated_in_place(self):
        """A derived copy, or one run's caution would leak into the next."""
        triggers = self.triggers()
        before = self.thresholds(triggers)
        pace.modulate(triggers, reading(median=-4.0))

        assert self.thresholds(triggers) == before

    def test_too_few_contributors_leaves_thresholds_unmodified(self):
        """An unknown macro reading must not tighten entry any more than loosen it."""
        before = self.thresholds(self.triggers())
        modulated, decision = pace.modulate(
            self.triggers(), reading(median=-9.0, contributors=2)
        )

        assert decision["applied"] is False
        assert "contributors" in decision["reason"]
        assert self.thresholds(modulated) == before

    def test_no_reading_at_all_leaves_thresholds_unmodified(self):
        before = self.thresholds(self.triggers())
        modulated, decision = pace.modulate(
            self.triggers(), {"median_pp": None, "contributors": 0, "tickers": []}
        )

        assert decision["applied"] is False
        assert self.thresholds(modulated) == before

    def test_kill_switches_are_never_touched(self):
        triggers = self.triggers()
        modulated, _ = pace.modulate(triggers, reading(median=-4.0))

        for trigger_id, spec in triggers.items():
            if spec["to"] != "probe":
                assert modulated[trigger_id] == spec

    def test_only_entry_transitions_are_adjusted(self):
        _, decision = pace.modulate(self.triggers(), reading(median=-4.0))

        adjusted = set(decision["adjusted"])
        assert adjusted == {"valuation_buy_zone"}

    def test_flag_conditions_are_left_alone(self):
        modulated, _ = pace.modulate(self.triggers(), reading(median=-4.0))
        conditions = modulated["valuation_buy_zone"]["conditions"]

        flag_conditions = [c for c in conditions if "flag_absent" in c]
        assert flag_conditions == [
            c for c in self.triggers()["valuation_buy_zone"]["conditions"]
            if "flag_absent" in c
        ]

    def test_a_gte_threshold_is_tightened_upwards_not_downwards(self):
        """Tightening means harder to satisfy, whichever way the comparator points."""
        triggers = {
            "entry": {
                "label": "E", "to": "probe", "from": ["watch"], "mode": "all",
                "conditions": [{"metric": "roiic", "comparator": "gte", "threshold": 20.0}],
            }
        }
        modulated, _ = pace.modulate(triggers, reading(median=-4.0), factor=0.8)

        assert modulated["entry"]["conditions"][0]["threshold"] > 20.0

    @pytest.mark.parametrize("threshold,comparator", [
        (-2.0, "lte"), (-2.0, "lt"), (-2.0, "gte"), (-2.0, "gt"),
    ])
    def test_a_negative_threshold_is_tightened_not_loosened(self, threshold, comparator):
        """Multiplying a negative `lte` threshold moves it toward zero — looser.

        This file's own domain is signed (its corpus reading today is -4.33pp),
        so a negative entry threshold is a foreseeable addition, and the
        multiplicative form would have done the exact opposite of R6 silently.
        """
        tightened = pace._tighten(threshold, comparator, 0.85)

        if comparator in ("lt", "lte"):
            assert tightened < threshold
        else:
            assert tightened > threshold

    @pytest.mark.parametrize("factor", [1.5, 0.0, -0.5, 2, None, "0.85", True])
    def test_a_factor_outside_zero_to_one_cannot_loosen_entry(self, factor):
        """"Tighten it more aggressively" reads as 1.5, which would loosen."""
        before = self.thresholds(self.triggers())
        modulated, _ = pace.modulate(
            self.triggers(), reading(median=-9.0), factor=factor
        )
        after = self.thresholds(modulated)

        assert all(a <= b for a, b in zip(after, before))

    def test_a_factor_above_one_falls_back_to_the_default(self):
        modulated, decision = pace.modulate(
            self.triggers(), reading(median=-9.0), factor=1.5
        )
        expected, _ = pace.modulate(self.triggers(), reading(median=-9.0))

        assert self.thresholds(modulated) == self.thresholds(expected)
        assert decision["applied"] is True

    def test_the_evidence_reports_the_direction_actually_written(self):
        _, decision = pace.modulate(self.triggers(), reading(median=-4.0))
        assert "stricter" in decision["evidence"]

    def test_the_decision_records_the_median_and_its_contributor_count(self):
        _, decision = pace.modulate(self.triggers(), reading(median=-4.0, contributors=11))

        assert decision["median_pp"] == -4.0
        assert decision["contributors"] == 11
        assert decision["floor_pp"] is not None
        assert decision["factor"] is not None

    def test_the_evidence_line_names_the_reading(self):
        _, decision = pace.modulate(self.triggers(), reading(median=-4.0, contributors=11))
        assert "-4" in decision["evidence"] and "11" in decision["evidence"]


# ── Through advance() ──────────────────────────────────────────────────────


class TestThroughAdvance:
    def service(self, **kwargs):
        """A stub that builds its own evaluator, so it needs real metric ids.

        `advance` validates the trigger registry against `engine.metrics` at
        construction — that startup check is what stops a trigger naming a
        nonexistent metric from reading indeterminate forever.
        """
        service = StubService(**kwargs)
        service.config = {}
        service.suite = type("S", (), {"raw_data_dir": "/nonexistent"})()
        service.engine = type(
            "E", (),
            {
                "registry_hash": "abc123",
                "metrics": dict(ComputeEngine().metrics),
                "macro": {},
            },
        )()
        return service

    def buyable(self):
        metrics = healthy_metrics()
        # Inside the standard buy zone (60 / 2.0) but outside a tightened one.
        metrics["pe_vs_historical"] = metric(55.0)
        metrics["trailing_peg"] = metric(1.9)
        return metrics

    def test_the_default_path_resolves_a_reading_without_one_being_injected(
        self, wm, tmp_path
    ):
        """Every other test here injects `pace_reading`, so the real resolution
        path — `advance` -> `corpus_spread` -> `modulate` — was never exercised
        end to end. A corpus it can actually read must produce a decision, and
        the reading must be the corpus's, not a default."""
        watched(wm)
        service = self.service(metrics=self.buyable())
        # Six cheap names: a real median, but under the contributor minimum.
        service.suite = type("S", (), {
            "raw_data_dir": corpus(tmp_path, {f"T{i}": 5.0 for i in range(6)})
        })()

        out = advance(service, wm)

        assert out["pace"]["applied"] is False
        assert out["pace"]["contributors"] == 6
        assert "contributors" in out["pace"]["reason"]

    def test_the_default_path_applies_when_the_real_corpus_is_expensive(
        self, wm, tmp_path
    ):
        watched(wm)
        service = self.service(metrics=self.buyable())
        # Ten expensive names (P/E 100 -> ~1% earnings yield vs a 7% G-Sec).
        service.suite = type("S", (), {
            "raw_data_dir": corpus(tmp_path, {f"T{i}": 100.0 for i in range(10)})
        })()
        service.engine = type("E", (), {
            "registry_hash": "abc123",
            "metrics": dict(ComputeEngine().metrics),
            "macro": {"gsec_yield_pct": 7.0},
        })()

        out = advance(service, wm)

        assert out["pace"]["applied"] is True
        assert out["pace"]["contributors"] == 10
        assert out["outcomes"][0]["proposal"] is None

    def test_a_wide_spread_proposes_entry(self, wm):
        watched(wm)
        out = advance(
            self.service(metrics=self.buyable()), wm, pace_reading=reading(median=3.0)
        )

        assert out["outcomes"][0]["proposal"]["to"] == "probe"
        assert out["pace"]["applied"] is False

    def test_a_compressed_spread_withholds_the_same_entry(self, wm):
        watched(wm)
        out = advance(
            self.service(metrics=self.buyable()), wm, pace_reading=reading(median=-4.0)
        )

        assert out["outcomes"][0]["proposal"] is None
        assert out["pace"]["applied"] is True

    def test_modulation_is_named_in_the_proposal_evidence(self, wm):
        """A tightened threshold must never be invisible in the decision record."""
        watched(wm)
        metrics = healthy_metrics()
        metrics["pe_vs_historical"] = metric(30.0)   # clears even the tightened bar
        metrics["trailing_peg"] = metric(1.0)

        out = advance(
            self.service(metrics=metrics), wm, pace_reading=reading(median=-4.0)
        )
        proposal = out["outcomes"][0]["proposal"]

        assert proposal["to"] == "probe"
        assert "pace" in proposal["evidence"].lower()
        assert proposal["pace"]["applied"] is True

    def test_the_median_and_contributor_count_appear_in_that_evidence(self, wm):
        watched(wm)
        metrics = healthy_metrics()
        metrics["pe_vs_historical"] = metric(30.0)
        metrics["trailing_peg"] = metric(1.0)

        out = advance(
            self.service(metrics=metrics), wm,
            pace_reading=reading(median=-4.0, contributors=11),
        )
        evidence = out["outcomes"][0]["proposal"]["evidence"]

        assert "-4" in evidence and "11" in evidence

    def test_kill_switches_fire_identically_under_a_compressed_spread(self, wm):
        """The failure that matters: macro must never reach exit logic."""
        watched(wm, state="probe")
        metrics = healthy_metrics()
        metrics["roiic"] = metric(4.0)  # below the 12% incremental-return switch

        wide = advance(self.service(metrics=metrics), wm, pace_reading=reading(median=3.0))
        assert wide["outcomes"][0]["proposal"]["to"] == "exit_review"

        # Same company, same metrics, compressed spread.
        wm2 = WatchlistManager(path=str(wm.path) + ".2")
        watched(wm2, state="probe")
        tight = advance(
            self.service(metrics=metrics), wm2, pace_reading=reading(median=-9.0)
        )

        assert tight["outcomes"][0]["proposal"]["to"] == "exit_review"
        assert tight["outcomes"][0]["proposal"]["trigger_id"] == (
            wide["outcomes"][0]["proposal"]["trigger_id"]
        )

    def test_exit_review_transitions_are_not_threshold_adjusted(self, wm):
        _, decision = pace.modulate(load_triggers(), reading(median=-9.0))
        assert all(
            load_triggers()[trigger_id]["to"] == "probe" for trigger_id in decision["adjusted"]
        )

    def test_the_eligibility_verdict_is_identical_either_way(self, wm):
        """Gates read metrics, never triggers — the modulator cannot reach them."""
        watched(wm)
        service = self.service(metrics=self.buyable())

        wide = advance(service, wm, pace_reading=reading(median=3.0))
        wm2 = WatchlistManager(path=str(wm.path) + ".2")
        watched(wm2)
        tight = advance(service, wm2, pace_reading=reading(median=-9.0))

        assert wide["outcomes"][0]["verdict"] == tight["outcomes"][0]["verdict"]

    def test_an_injected_evaluator_is_used_as_supplied(self, wm):
        """Injection is the existing seam; the modulator must not override it."""
        watched(wm)
        out = advance(
            self.service(metrics=self.buyable()), wm,
            evaluator=TriggerEvaluator(load_triggers()),
            pace_reading=reading(median=-9.0),
        )

        assert out["outcomes"][0]["proposal"]["to"] == "probe"
        assert out["pace"]["applied"] is False
        assert "caller" in out["pace"]["reason"]

    def test_adding_a_watchlist_entry_does_not_shift_the_reading(self, tmp_path, wm):
        """The median is a property of the corpus, not of what is being tracked."""
        path = corpus(tmp_path / "raw", {"A": 20.0, "B": 10.0, "C": 25.0})
        before = pace.corpus_spread(path, macro={"gsec_yield_pct": 7.0})

        watched(wm, "ASTRAL")
        watched(wm, "CDSL")
        after = pace.corpus_spread(path, macro={"gsec_yield_pct": 7.0})

        assert before == after


class TestThePaceClauseIsAttachedByTrigger:
    """The note may only appear on a proposal a threshold actually moved.

    `modulate` renders its own evidence from the values it wrote, because that
    line "must never be able to claim a tightening that did not happen". The
    same rule has to hold one layer up, where the line is *recorded*:
    `watchlist.transition` writes a proposal's evidence into an append-only
    history, so a clause attached to the wrong proposal is permanent.

    Two triggers now propose `probe`, and only one of them is tightenable.
    `valuation_buy_zone` carries `metric` conditions with thresholds a factor
    can move; `fast_lane_buy_zone`'s single condition is
    `lane_verdict: qualifies` and holds no threshold anywhere. Keyed by
    destination state — which is how this was written when exactly one trigger
    targeted `probe` — the clause reached both. `decision["adjusted"]` is keyed
    by trigger id, which is the granularity the question actually has.
    """

    def both_lanes(self, wm):
        """A core entry and a fast-lane entry, both one run from `probe`."""
        watched(wm, "ASTRAL")
        fast_lane_entry(wm, ticker="ZENSAR", state="watch")

    def run(self, wm, apply=False):
        # `fast_lane_metrics` clears all six lane gates *and* stays inside the
        # tightened core buy zone, so both lanes propose an entry in one run —
        # which is the only arrangement that can tell the two apart.
        service = TestThroughAdvance().service(
            metrics=fast_lane_metrics(), composite=6.5
        )
        out = advance(service, wm, apply=apply, pace_reading=reading(median=-4.0))
        assert out["pace"]["applied"] is True
        return {o["ticker"]: o["proposal"] for o in out["outcomes"]}

    def test_the_core_proposal_carries_the_clause(self, wm):
        self.both_lanes(wm)
        core = self.run(wm)["ASTRAL"]

        assert core["trigger_id"] == "valuation_buy_zone"
        assert core["to"] == "probe"
        assert "deployment pace" in core["evidence"].lower()
        assert core["pace"]["applied"] is True

    def test_the_fast_lane_proposal_does_not(self, wm):
        """Nothing about `fast_lane_buy_zone` was tightened, so nothing may say
        it was."""
        self.both_lanes(wm)
        fast = self.run(wm)["ZENSAR"]

        assert fast["trigger_id"] == "fast_lane_buy_zone"
        assert fast["to"] == "probe"
        assert "pace" not in fast["evidence"].lower()
        assert "pace" not in fast

    def test_the_recorded_history_line_carries_no_pace_claim_either(self, wm):
        """The damage this does: the evidence is written, once, forever."""
        self.both_lanes(wm)
        self.run(wm, apply=True)

        record = wm.get("ZENSAR")["state_history"][-1]
        assert record["to"] == "probe"
        assert "pace" not in record["evidence"].lower()

        # The core lane's record is the control: the same run, the same
        # destination, and there the claim is true.
        assert "pace" in wm.get("ASTRAL")["state_history"][-1]["evidence"].lower()

    def test_the_attached_record_lists_only_that_triggers_own_changes(self, wm):
        """And the string recorded is the one the attached payload states —
        two renderings of a tightening are two things that can disagree."""
        self.both_lanes(wm)
        core = self.run(wm)["ASTRAL"]

        assert set(core["pace"]["adjusted"]) == {"valuation_buy_zone"}

        clause = core["evidence"].split("[deployment pace: ", 1)[1].rstrip("]")
        assert clause == core["pace"]["evidence"]
        assert "pe_vs_historical" in clause and "trailing_peg" in clause

    def test_a_fast_lane_entry_alone_still_reports_the_run_level_reading(self, wm):
        """`adjusted_states` remains the run's display aggregate — the CLI line
        that says the corpus is expensive is unchanged, only the per-proposal
        attachment moved."""
        fast_lane_entry(wm, ticker="ZENSAR", state="watch")
        service = TestThroughAdvance().service(
            metrics=fast_lane_metrics(), composite=6.5
        )

        out = advance(service, wm, pace_reading=reading(median=-4.0))

        assert out["pace"]["applied"] is True
        assert out["pace"]["adjusted_states"] == ("probe",)
        assert out["outcomes"][0]["proposal"]["trigger_id"] == "fast_lane_buy_zone"


class TestNonFiniteReadingsFailClosed:
    """An unknown macro reading must not tighten entry any more than loosen it.

    Every comparison against NaN is False, so an unguarded non-finite median
    skipped the at-or-above-floor branch and landed in the tightening one —
    inverting this module's single invariant on a signal nobody could read.
    """

    @pytest.mark.parametrize("median", [float("nan"), float("inf"), float("-inf")])
    def test_a_non_finite_spread_leaves_thresholds_alone(self, median):
        triggers, decision = pace.modulate(
            load_triggers(), {"median_pp": median, "contributors": 20}
        )

        assert decision["applied"] is False
        assert "not a finite number" in decision["reason"]
        assert triggers == load_triggers()

    def test_a_non_finite_floor_leaves_thresholds_alone(self):
        _, decision = pace.modulate(
            load_triggers(), {"median_pp": -5.0, "contributors": 20},
            floor_pp=float("nan"),
        )

        assert decision["applied"] is False
        assert "floor" in decision["reason"]

    def test_a_non_finite_company_reading_is_not_a_contributor(self, tmp_path):
        """A NaN is a float; it would pass isinstance and poison the median."""
        import json
        import pandas as pd
        from boundless100x.compute_engine.metrics.base import MetricResult

        for name in ("A", "B"):
            d = tmp_path / name
            d.mkdir()
            (d / "metadata.json").write_text(json.dumps({"Stock P/E": 20.0}))
            pd.DataFrame({"year": ["Mar 2025"], "revenue": [100.0]}).to_csv(
                d / "financials.csv", index=False
            )

        import boundless100x.lifecycle.pace as pace_module
        original = pace_module.compute_earnings_yield_spread
        try:
            pace_module.compute_earnings_yield_spread = (
                lambda data, params: MetricResult(value=float("nan"))
            )
            reading = pace.corpus_spread(tmp_path)
        finally:
            pace_module.compute_earnings_yield_spread = original

        assert reading["contributors"] == 0
        assert reading["median_pp"] is None
