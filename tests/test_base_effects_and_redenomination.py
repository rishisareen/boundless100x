"""Readings that are arithmetically correct and are not evidence.

Two distinct failures, both found on JIOFIN — a company demerged from Reliance
in 2023 whose reported history begins mid-window.

**The base effect.** Revenue of ₹45 Cr growing to ₹3,521 Cr is a 328% "CAGR"
and is a fact about the demerger date. Six such metrics scored a flat 1.0,
carrying 47% of the Growth element, and the trailing PEG built on the same base
was 38% of all scored Price weight. The report then printed "0.29x is below 1.0
— the golden rule for 100-baggers... the valuation appears justified and
attractive" about a share on 78x earnings.

**The re-denomination.** A bonus issue moves reserves into equity capital: net
worth unchanged, share count up, every holder compensated in shares. Book value
per share therefore falls by arithmetic while nothing has happened to the owner.
`book_value_cagr_5yr` charged companies for it — 13 of the 22 cached tickers
carry such a break, and four read as catastrophic book destruction.
"""

import pandas as pd
import pytest

from boundless100x.compute_engine.metrics.base import UNSCORABLE_FLAGS, is_scorable
from boundless100x.compute_engine.metrics.builtin.growth import (
    NEGLIGIBLE_BASE_FLAG,
    compute_book_value_cagr,
    compute_cagr,
)
from boundless100x.compute_engine.metrics.builtin.valuation import compute_trailing_peg


def financials(pat_series, revenue=None):
    n = len(pat_series)
    return pd.DataFrame({
        "year": [f"Mar 20{15 + i}" for i in range(n)],
        "pat": pat_series,
        "revenue": revenue if revenue is not None else pat_series,
    })


def balance_sheet(equity_capital, reserves):
    n = len(equity_capital)
    return pd.DataFrame({
        "year": [f"Mar 20{15 + i}" for i in range(n)],
        "equity_capital": equity_capital,
        "reserves": reserves,
    })


class TestTheBaseEffectIsNotAGrowthRate:
    def test_a_cagr_from_a_negligible_base_is_flagged_unscorable(self):
        """JIOFIN's revenue line: ₹45 Cr to ₹3,521 Cr across the window."""
        data = {"financials": financials([31, 1605, 1613, 1561],
                                         revenue=[45, 1855, 2043, 3521])}

        result = compute_cagr(data, {"field": "revenue", "years": 5})

        assert result.ok
        assert NEGLIGIBLE_BASE_FLAG in result.flags
        assert not is_scorable(result)

    def test_the_value_is_kept_rather_than_refused(self):
        """"Revenue went from 45 to 3,521" is worth knowing. What the flag
        withdraws is its vote, not the observation."""
        data = {"financials": financials([31, 1605, 1613, 1561],
                                         revenue=[45, 1855, 2043, 3521])}

        result = compute_cagr(data, {"field": "revenue", "years": 5})

        assert result.value > 100
        assert "45" in result.metadata["base_effect_reason"]
        assert "3,521" in result.metadata["base_effect_reason"]

    def test_ordinary_growth_is_untouched(self):
        """The guard must not fire on a company that merely grew well."""
        data = {"financials": financials([100, 125, 150, 180, 220, 260])}

        result = compute_cagr(data, {"field": "pat", "years": 5})

        assert result.ok
        assert NEGLIGIBLE_BASE_FLAG not in result.flags
        assert is_scorable(result)

    def test_a_trailing_peg_off_that_base_is_refused_too(self):
        """The PEG computes its own CAGR, so it needs its own guard — and it
        was the metric that carried JIOFIN's entry-price gate."""
        data = {
            "financials": financials([31, 1605, 1613, 1561]),
            "metadata": {"Stock P/E": 78.0},
        }

        result = compute_trailing_peg(data, {"cagr_years": 3})

        assert result.ok and result.value < 1.0
        assert NEGLIGIBLE_BASE_FLAG in result.flags
        assert not is_scorable(result)

    def test_the_flag_is_registered_as_unscorable(self):
        """The two layers that must agree read one list."""
        assert NEGLIGIBLE_BASE_FLAG in UNSCORABLE_FLAGS


class TestABonusIssueIsNotBookValueDestruction:
    def test_a_bonus_issue_leaves_book_value_per_share_growth_intact(self):
        """1:1 bonus in year 3: equity capital doubles out of reserves, net
        worth unchanged. The holder now owns twice as many shares of the same
        company and has lost nothing."""
        data = {
            "balance_sheet": balance_sheet(
                equity_capital=[10, 10, 20, 20, 20, 20],
                reserves=[90, 110, 120, 145, 175, 210],
            ),
            "metadata": {"Face Value": 10.0},
        }

        result = compute_book_value_cagr(data, {"years": 5})

        assert result.ok
        assert result.value > 0, "a bonus issue read as book value shrinking"
        assert "share_count_restated" in result.flags
        assert "book_value_eroding" not in result.flags

    def test_a_genuine_issuance_still_lowers_book_value_per_share(self):
        """The other half, and the reason this is not simply "ignore share
        count jumps": cash arriving raises net worth, so the dilution is real
        and must go on showing.

        Priced at ₹200 against a book of ₹200/share — 1.5 Cr new shares on 1 Cr
        existing, raising ₹300 Cr. Net worth rises in step with the count, the
        ratio test reads it as a raise, and book value per share is unchanged
        rather than restated away.
        """
        data = {
            "balance_sheet": balance_sheet(
                equity_capital=[10, 10, 25, 25, 25, 25],
                reserves=[190, 190, 475, 475, 475, 475],
            ),
            "metadata": {"Face Value": 10.0},
        }

        result = compute_book_value_cagr(data, {"years": 5})

        assert result.ok
        assert "share_count_restated" not in result.flags
        # Issued at book: the holder is neither better nor worse off per share.
        assert result.value == pytest.approx(0.0, abs=0.01)

    def test_a_deficit_year_refuses_rather_than_reporting_a_collapse(self):
        """IGIL's shape: positive at both ends, negative in the middle. There
        is no compound rate through a deficit, and reporting -87.9%/yr
        described a restructuring as a shrinking book."""
        data = {
            "balance_sheet": balance_sheet(
                equity_capital=[0.4, 0.4, 86, 86],
                reserves=[339, -605, 976, 1402],
            ),
            "metadata": {"Face Value": 2.0},
        }

        result = compute_book_value_cagr(data, {"years": 5})

        assert not result.ok
        assert "negative" in result.error

    @pytest.mark.parametrize("ticker", ["JIOFIN", "IGIL", "TARSONS", "IXIGO"])
    def test_the_four_corpus_regressions_are_gone(self, ticker):
        """Read off the real cached data, because the fixtures above are the
        shapes I believed were there — these are the ones that actually were."""
        from pathlib import Path
        import json

        root = (
            Path(__file__).parent.parent
            / "boundless100x" / "data_fetcher" / "raw_data" / ticker
        )
        if not (root / "balance_sheet.csv").exists():
            pytest.skip(f"{ticker} not in this checkout's corpus")

        result = compute_book_value_cagr(
            {
                "balance_sheet": pd.read_csv(root / "balance_sheet.csv"),
                "metadata": json.loads((root / "metadata.json").read_text()),
            },
            {"years": 5},
        )

        if result.ok:
            assert -30 < result.value < 80, (
                f"{ticker} still reads {result.value:.1f}%/yr"
            )
        else:
            assert result.error.strip()
