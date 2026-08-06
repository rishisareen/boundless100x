"""Re-rating headroom: valuation measured constructively rather than as a veto.

Every other price metric answers "is this too expensive?". This one answers
"how much multiple expansion would the company's own fundamentals justify?" —
the accelerator the 100x roadmap is built around.

Two properties carry the unit. The band must respond to *fundamentals*, not
only to price, or it is a rescaled P/E wearing a new name; and it must move
nothing, because it enters at zero weight and R7 is this phase's hardest
constraint.
"""

import shutil

import pandas as pd
import pytest
import yaml

from boundless100x.compute_engine.engine import ComputeEngine
from boundless100x.compute_engine.metrics.builtin.valuation import (
    compute_rerating_headroom,
)
from boundless100x.compute_engine.scorer import SQGLPScorer
from tests.conftest import make_data, make_financials


def headroom_data(roce: float = 28.0, pat_growth: float = 0.28,
                  current_multiple: float = 30.0, n: int = 10) -> dict:
    """A company whose latest close over its latest annual EPS is `current_multiple`."""
    eps = float(make_financials(n, pat_growth=pat_growth)["eps"].iloc[-1])
    return make_data(
        n=n,
        financials={"pat_growth": pat_growth},
        ratios={"roce": roce},
        price={"start_close": 10.0, "end_close": eps * current_multiple},
    )


def headroom(**kwargs):
    return compute_rerating_headroom(headroom_data(**kwargs), {})


class TestDirectionAndSign:
    def test_trading_below_the_justified_multiple_yields_positive_headroom(self):
        """Sign convention: positive means room to re-rate up, as the name says."""
        result = headroom(roce=28.0, pat_growth=0.28, current_multiple=15.0)

        assert result.ok
        assert result.value > 0
        assert result.metadata["justified_multiple"] > result.metadata["current_multiple"]

    def test_trading_above_the_justified_multiple_yields_negative_headroom(self):
        result = headroom(roce=28.0, pat_growth=0.28, current_multiple=120.0)

        assert result.ok
        assert result.value < 0

    def test_the_value_is_a_ratio_expressed_as_a_percentage(self):
        probe = headroom(current_multiple=20.0)
        justified = probe.metadata["justified_multiple"]

        # Priced at exactly half the justified multiple => +100%.
        result = headroom(current_multiple=justified / 2)
        assert result.value == pytest.approx(100.0, abs=0.5)


class TestTheBandRespondsToFundamentals:
    def test_a_low_quality_company_earns_a_lower_justified_multiple(self):
        rich = headroom(roce=28.0, pat_growth=0.28, current_multiple=30.0)
        plain = headroom(roce=10.0, pat_growth=0.05, current_multiple=30.0)

        assert plain.metadata["justified_multiple"] < rich.metadata["justified_multiple"]

    def test_the_same_traded_multiple_produces_less_headroom_for_the_weaker_company(self):
        """If price alone drove this, both would report the same number."""
        rich = headroom(roce=28.0, pat_growth=0.28, current_multiple=30.0)
        plain = headroom(roce=10.0, pat_growth=0.05, current_multiple=30.0)

        assert plain.value < rich.value

    def test_consistency_lifts_the_justified_multiple(self):
        """Longevity is a multiplier on the RoCE x growth band, not decoration."""
        consistent = headroom(roce=22.0, pat_growth=0.18, current_multiple=30.0)
        erratic = compute_rerating_headroom(
            _with_roce_series(headroom_data(roce=22.0, pat_growth=0.18),
                              [22.0] * 2 + [8.0] * 8),
            {},
        )
        assert erratic.metadata["justified_multiple"] < consistent.metadata["justified_multiple"]

    def test_the_percentage_form_is_scale_free(self):
        """Two very different absolute multiples, one ratio, one headroom figure."""
        def at_ratio(roce, growth, ratio):
            probe = headroom(roce=roce, pat_growth=growth, current_multiple=20.0)
            justified = probe.metadata["justified_multiple"]
            return headroom(roce=roce, pat_growth=growth,
                            current_multiple=justified / ratio)

        rich = at_ratio(28.0, 0.28, 1.5)
        plain = at_ratio(13.0, 0.10, 1.5)

        assert rich.metadata["current_multiple"] != pytest.approx(
            plain.metadata["current_multiple"]
        )
        assert rich.value == pytest.approx(plain.value, abs=0.5)
        assert rich.value == pytest.approx(50.0, abs=0.5)


def _with_roce_series(data: dict, series: list[float]) -> dict:
    data["ratios"] = data["ratios"].assign(roce=series)
    return data


class TestErrorsRatherThanDefaults:
    def test_a_missing_roce_input_errors_rather_than_taking_a_middle_band(self):
        """An unknown quality profile must not silently receive a default multiple."""
        data = headroom_data()
        data["ratios"] = data["ratios"].drop(columns=["roce"])

        result = compute_rerating_headroom(data, {})
        assert not result.ok
        assert "roce" in result.error.lower()

    def test_a_missing_growth_input_errors(self):
        data = headroom_data()
        data["financials"] = data["financials"].drop(columns=["pat"])

        result = compute_rerating_headroom(data, {})
        assert not result.ok

    def test_a_missing_price_errors(self):
        data = headroom_data()
        data["price"] = pd.DataFrame()

        result = compute_rerating_headroom(data, {})
        assert not result.ok
        assert "price" in result.error.lower()

    def test_zero_or_negative_earnings_error_rather_than_a_nonsensical_ratio(self):
        for eps in (0.0, -4.0):
            data = headroom_data()
            data["financials"] = data["financials"].assign(
                eps=[eps] * len(data["financials"])
            )
            result = compute_rerating_headroom(data, {})
            assert not result.ok
            assert "eps" in result.error.lower()

    def test_a_malformed_band_table_errors_rather_than_guessing(self):
        result = compute_rerating_headroom(
            headroom_data(), {"roce_bands": [10, 20], "justified_multiple": [[1, 2]]}
        )
        assert not result.ok


class TestContractsWithTheRestOfTheEngine:
    def test_the_metric_emits_no_raw_series(self):
        """KTD6: a multiples series behind a ratio value is a unit mismatch."""
        assert headroom().raw_series == []

    def test_the_price_basis_is_recorded(self):
        assert headroom().metadata["price_basis"] == "legacy_close_unknown_adjustment"

        data = headroom_data()
        data["price"] = data["price"].assign(adj_close=data["price"]["close"])
        assert compute_rerating_headroom(data, {}).metadata["price_basis"] == "raw_close"

    def test_the_interpretation_band_travels_with_the_value(self):
        """R8: a bare number cannot be read as favourable without one."""
        probe = headroom(current_multiple=20.0)
        justified = probe.metadata["justified_multiple"]

        assert headroom(current_multiple=justified / 2).metadata["band"] == "favourable"
        assert headroom(current_multiple=justified).metadata["band"] == "fair"
        assert headroom(current_multiple=justified * 2).metadata["band"] == "stretched"

    def test_flags_name_the_band_and_nothing_else(self):
        probe = headroom(current_multiple=20.0)
        justified = probe.metadata["justified_multiple"]

        assert headroom(current_multiple=justified / 2).flags == [
            "rerating_headroom_favourable"
        ]
        assert headroom(current_multiple=justified * 2).flags == [
            "rerating_headroom_stretched"
        ]
        assert headroom(current_multiple=justified).flags == []

    def test_it_reads_only_truncatable_frames(self):
        """A2: this must compute inside the backtest, which withholds metadata."""
        data = headroom_data()
        data["metadata"] = {}

        assert compute_rerating_headroom(data, {}).ok


class TestNonRegression:
    def test_registering_the_metric_leaves_every_score_identical(self, tmp_path):
        """R7 is the phase's hardest constraint; this is its per-unit proof."""
        shipped = ComputeEngine().registry_dir
        without = tmp_path / "without"
        shutil.copytree(shipped, without)

        price_yaml = without / "elements" / "price.yaml"
        config = yaml.safe_load(price_yaml.read_text())
        assert "rerating_headroom" in config["metrics"]
        del config["metrics"]["rerating_headroom"]
        # sort_keys=False on purpose. The scorer accumulates each element's
        # weighted score in registry iteration order, and re-alphabetising the
        # metrics reorders that float summation — 6.5 became 6.499999999999999
        # with nothing about the scoring having changed. A comparison that
        # loose would hide the regression this test exists to catch.
        price_yaml.write_text(yaml.safe_dump(config, sort_keys=False))

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

    def test_it_appears_in_details_unscored_and_unweighted(self):
        engine = ComputeEngine()
        scorer = SQGLPScorer(engine.metrics, engine.element_weights)
        scores = scorer.score(engine.run_all(make_data()))

        entry = scores["details"]["rerating_headroom"]
        assert entry["weight"] == 0
        assert entry["score"] is None
        assert entry["value"] is not None

    def test_it_is_declared_at_zero_weight_in_the_price_element(self):
        engine = ComputeEngine()
        config = engine.metrics["rerating_headroom"]

        assert config["element"] == "price"
        assert config["scoring"]["weight"] == 0.0
