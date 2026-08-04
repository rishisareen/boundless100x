"""Promoter pledge must fail closed.

An unpledged promoter and an unknown one look identical if absence scores as
0.0 — the best possible outcome for a risk metric. Only a verified BSE
observation may report a value; otherwise the metric errors out and the
scorer excludes it, same as any other unavailable input.
"""

import pandas as pd

from boundless100x.compute_engine.metrics.builtin.size import compute_promoter_pledge


class TestComputePromoterPledge:
    def test_no_bse_data_reports_unavailable_not_zero(self):
        result = compute_promoter_pledge({"shareholding": pd.DataFrame()}, {})

        assert not result.ok
        assert result.error is not None
        assert result.value != 0.0

    def test_bse_data_without_pledge_column_reports_unavailable(self):
        data = {
            "shareholding": pd.DataFrame(),
            "shareholding_bse": pd.DataFrame({"quarter": ["Q1 2025"]}),
        }
        result = compute_promoter_pledge(data, {})

        assert not result.ok

    def test_bse_pledge_value_is_used_when_present(self):
        data = {
            "shareholding": pd.DataFrame(),
            "shareholding_bse": pd.DataFrame({
                "quarter": ["Q1 2025", "Q2 2025"],
                "promoter_pledge_pct": [12.0, 15.0],
            }),
        }
        result = compute_promoter_pledge(data, {})

        assert result.ok
        assert result.value == 15.0
        assert "promoter_pledge_red_flag" in result.flags

    def test_bse_pledge_below_threshold_has_no_red_flag(self):
        data = {
            "shareholding": pd.DataFrame(),
            "shareholding_bse": pd.DataFrame({
                "quarter": ["Q1 2025"],
                "promoter_pledge_pct": [3.0],
            }),
        }
        result = compute_promoter_pledge(data, {})

        assert result.ok
        assert result.flags == []
