"""Cash parked is not cash spent, and a label is not a second opinion.

Found reviewing CAPLIPOINT — a debt-free compounder holding ₹2,875 Cr of
liquid assets against a ₹19,460 Cr market capitalisation.

**Treasury.** Free cash flow here is `CFO + CFI`, and CFI carries money moved
into deposits and mutual funds as well as money spent on plant. Caplin grew
financial investments by ₹848 Cr over five years against ₹495 Cr of growth in
plant, so most of its "negative free cash flow" was saving. Average FCF read
₹10.8 Cr, the DCF returned an intrinsic value of ₹43 against a ₹2,561 price,
the reverse DCF pinned at its +50% ceiling, and the veto that fired off it
FAILED an eligibility gate whose own conditions had passed.

**Labels.** Every metric declares its name and unit in the registry, and
`checklist.py` kept a second copy for the prompt. It drifted three times, each
time putting a confident wrong sentence in front of the model.
"""

import pandas as pd
import pytest

from boundless100x.compute_engine.metrics.builtin._helpers import (
    TREASURY_ADJUSTED_FLAG,
    operating_free_cash_flow,
    treasury_flows,
)
from boundless100x.compute_engine.metrics.builtin.profitability import (
    _get_annual_rows,
    stub_period_labels,
)


def frames(cfo, cfi, investments, start=2020):
    years = [f"Mar {start + i}" for i in range(len(cfo))]
    return (
        pd.DataFrame({"year": years, "cfo": cfo, "cfi": cfi}),
        pd.DataFrame({"year": years, "investments": investments}),
    )


class TestTreasuryIsNotCapex:
    def test_money_moved_into_investments_is_added_back(self):
        cf, bs = frames(cfo=[100, 100, 100], cfi=[-150, -150, -150],
                        investments=[0, 100, 200])

        reported, operating, detail = operating_free_cash_flow(cf, bs)

        assert list(reported) == [-50, -50, -50]
        # Years two and three each parked 100, which was never spent.
        assert list(operating) == [-50, 50, 50]
        assert detail["adjusted"] is True

    def test_real_capex_is_untouched(self):
        """The control. A company building plant must still show the outflow."""
        cf, bs = frames(cfo=[100, 100, 100], cfi=[-150, -150, -150],
                        investments=[50, 50, 50])

        reported, operating, detail = operating_free_cash_flow(cf, bs)

        assert list(reported) == list(operating)
        assert detail["adjusted"] is False

    def test_investments_being_sold_is_not_a_deduction(self):
        """Only increases are added back. A company liquidating its treasury to
        cover an operating shortfall is the case a reader most needs to see, and
        netting the inflow off would hide it."""
        cf, bs = frames(cfo=[10, 10], cfi=[100, 100], investments=[200, 100])

        _, operating, _ = operating_free_cash_flow(cf, bs)

        assert list(operating) == [110, 110]

    def test_alignment_is_on_the_period_label_not_row_position(self):
        """The two frames are filtered independently; a balance sheet with an
        extra period would otherwise pair each year against its neighbour."""
        cf = pd.DataFrame({"year": ["Mar 2021", "Mar 2022"],
                           "cfo": [100, 100], "cfi": [-150, -150]})
        bs = pd.DataFrame({"year": ["Mar 2020", "Mar 2021", "Mar 2022"],
                           "investments": [0, 100, 100]})

        _, operating, _ = operating_free_cash_flow(cf, bs)

        # 2021 parked 100; 2022 parked nothing.
        assert list(operating) == [50, -50]

    def test_no_investments_column_leaves_the_reading_alone(self):
        """Nothing can be inferred with no stock to difference, and inventing
        an adjustment would be worse than declining to make one."""
        cf = pd.DataFrame({"year": ["Mar 2021"], "cfo": [100], "cfi": [-150]})

        reported, operating, detail = operating_free_cash_flow(cf, pd.DataFrame())

        assert list(reported) == list(operating) == [-50]
        assert detail["adjusted"] is False

    def test_the_corpus_case_reverses_the_sign(self):
        """CAPLIPOINT, off the real cached data."""
        root = "boundless100x/data_fetcher/raw_data/CAPLIPOINT"
        try:
            cf = _get_annual_rows(pd.read_csv(f"{root}/cashflow.csv"), 10)
            bs = _get_annual_rows(pd.read_csv(f"{root}/balance_sheet.csv"), 11)
        except FileNotFoundError:
            pytest.skip("CAPLIPOINT not in this checkout's corpus")

        reported, operating, detail = operating_free_cash_flow(cf, bs)

        assert reported.mean() < 40, "reported FCF was not the depressed figure"
        assert operating.mean() > 100
        assert detail["adjusted"] is True


class TestTheAdjustmentIsStated:
    def test_a_metric_that_used_it_says_so(self):
        """An estimate that silently replaced a reported figure would be worse
        than the figure it replaced. `treasury_flows` infers cash movement from
        a year-end stock, and a rise can also be a mark-to-market gain."""
        from boundless100x.compute_engine.engine import ComputeEngine

        engine = ComputeEngine()
        root = "boundless100x/data_fetcher/raw_data/CAPLIPOINT"
        try:
            data = {
                "financials": pd.read_csv(f"{root}/financials.csv"),
                "cashflow": pd.read_csv(f"{root}/cashflow.csv"),
                "balance_sheet": pd.read_csv(f"{root}/balance_sheet.csv"),
                "metadata": __import__("json").loads(
                    open(f"{root}/metadata.json").read()
                ),
                "price": pd.DataFrame(),
            }
        except FileNotFoundError:
            pytest.skip("CAPLIPOINT not in this checkout's corpus")

        result = engine._run_metric(
            "fcf_consistency", engine.metrics["fcf_consistency"], data
        )

        assert result.ok
        assert TREASURY_ADJUSTED_FLAG in result.flags

    def test_the_flag_is_registered_for_rendering(self):
        from boundless100x.output.report_vocabulary import (
            FLAG_ELEMENT_MAP,
            FLAG_LABELS,
        )

        assert TREASURY_ADJUSTED_FLAG in FLAG_LABELS
        assert TREASURY_ADJUSTED_FLAG in FLAG_ELEMENT_MAP


class TestATransitionStubIsNotAYear:
    def test_a_shortened_period_is_dropped(self):
        """A company changing its year end files one short period whose label
        starts with the new month, so every month-based filter keeps it."""
        df = pd.DataFrame({
            "year": ["Jun 2015", "Mar 20169m", "Mar 2017", "Mar 2018"],
            "revenue": [252, 239, 402, 540],
        })

        kept = _get_annual_rows(df, 10)

        assert list(kept["year"]) == ["Mar 2017", "Mar 2018"]
        assert stub_period_labels(df) == ["Mar 20169m"]

    def test_ordinary_labels_are_untouched(self):
        df = pd.DataFrame({"year": ["Mar 2024", "Mar 2025"], "revenue": [1, 2]})

        assert list(_get_annual_rows(df, 10)["year"]) == ["Mar 2024", "Mar 2025"]
        assert stub_period_labels(df) == []


class TestPromptLabelsComeFromTheRegistry:
    def test_the_declared_name_and_unit_are_used(self):
        """The three drifts this replaces: a level labelled as a change, a
        ratio labelled as the wrong ratio, and a multiple labelled a percent."""
        from boundless100x.llm_layer.checklist import _label_for

        assert _label_for("cash_conversion") == ("Cash Conversion (OCF/EBITDA)", "%")
        assert _label_for("reinvestment_rate") == (
            "Reinvestment Rate (Capex/Depreciation)", "x"
        )
        assert _label_for("promoter_holding_trend") == (
            "Promoter Holding Trend (5yr)", "%"
        )

    def test_every_id_the_prompt_lists_exists_in_the_registry(self):
        """A list of ids can still name something the registry dropped; that
        would render as a title-cased guess with no unit."""
        import inspect

        from boundless100x.compute_engine.engine import ComputeEngine
        from boundless100x.llm_layer import checklist

        known = set(ComputeEngine().metrics)
        source = inspect.getsource(checklist)
        for block in ("metric_ids = [", "key_metric_ids = ["):
            listed = source.split(block, 1)[1].split("]", 1)[0]
            ids = [x.strip().strip('",') for x in listed.split(",") if x.strip()]
            unknown = [i for i in ids if i and not i.startswith("#") and i not in known]
            assert unknown == [], f"{block} names unknown metric(s): {unknown}"

    def test_an_unknown_id_degrades_readably(self):
        from boundless100x.llm_layer.checklist import _label_for

        assert _label_for("not_a_metric") == ("Not A Metric", "")


class TestSectorIsClassifiedOnEveryLabel:
    def test_a_pharma_breadcrumb_finds_the_studys_bucket(self):
        """"Pharmaceuticals & Biotechnology" matches neither "Healthcare" nor
        "Pharma" by whole phrase, and scored `unknown` for want of asking."""
        from boundless100x.compute_engine.sector import classify_sector, study_labels

        metadata = {
            "sector_industry": "Pharmaceuticals",
            "sector": "Pharmaceuticals & Biotechnology",
            "sector_broad": "Healthcare",
        }

        assert classify_sector(study_labels(metadata)) == "moderate_tailwind"

    def test_the_narrower_label_wins(self):
        """A pharma company is pharma before it is healthcare, so it takes the
        moderate bucket rather than the broad group's strong one."""
        from boundless100x.compute_engine.sector import classify_sector

        assert classify_sector(("Pharmaceuticals", "Healthcare")) == "moderate_tailwind"
        assert classify_sector(("Healthcare",)) == "strong_tailwind"

    def test_a_single_string_still_works(self):
        from boundless100x.compute_engine.sector import classify_sector

        assert classify_sector("Finance") == "strong_tailwind"
        assert classify_sector("Sugar") == "non_consideration"
        assert classify_sector(None) == "unknown"


class TestProductApprovalsAreNotStakeChanges:
    def test_a_usfda_approval_is_its_own_category(self):
        from boundless100x.data_fetcher.fetch_announcements import (
            classify_announcement,
        )

        assert classify_announcement(
            "Announcement under Regulation 30 (LODR)-Press Release / Media Release",
            "Press release regarding receipt of final approval by our subsidiary",
        ) == "product_approval"

    def test_a_real_stake_change_still_classifies(self):
        from boundless100x.data_fetcher.fetch_announcements import (
            classify_announcement,
        )

        assert classify_announcement(
            "Announcement under Regulation 30 (LODR)-Diversification / Disinvestment",
            "Edelweiss to bring in capital",
        ) == "stake_change"

    def test_naming_a_subsidiary_is_not_by_itself_a_stake_change(self):
        """Companies name a subsidiary in every announcement about one."""
        from boundless100x.data_fetcher.fetch_announcements import (
            classify_announcement,
        )

        assert classify_announcement(
            "Intimation regarding change of registered office of subsidiary", ""
        ) != "stake_change"
