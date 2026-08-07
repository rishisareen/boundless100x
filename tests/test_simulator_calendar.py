"""`simulator.calendar` — replay dates from the corpus's own fiscal
calendar (KTD7), plus the per-lane battery-complete reading.

Every fixture here is written to a temp directory via `write_ticker_dir`
(`tests/conftest.py`) — never `data_fetcher/raw_data/` — mirroring the
CSV/JSON shape `universe.load_ticker_data` reads.
"""

import pytest

from boundless100x.simulator import calendar as calendar_module
from tests.conftest import write_ticker_dir


def test_dates_stay_within_start_and_last_priced_date(tmp_path):
    root = tmp_path / "raw_data"
    write_ticker_dir(root, "AAA", years=10, quarters=13, shareholding_quarters=12, price_days=3200)
    write_ticker_dir(root, "BBB", years=10, quarters=13, shareholding_quarters=12, price_days=2600)

    cal = calendar_module.compute_calendar(root)

    assert cal.dates, "expected at least one replay date from a healthy fixture"
    assert min(cal.dates) >= cal.start == calendar_module.REPLAY_START
    assert max(cal.dates) <= cal.end

    # `end` must be the *minimum* of the two tickers' last priced dates
    # (BBB's shorter series), not the maximum — every replay date must be
    # markable for every discovered ticker.
    import boundless100x.simulator.universe as universe_module
    last_priced = {
        t.name: universe_module.load_ticker_data(t)["price"]["date"].max()
        for t in universe_module.discover_candidates(root)
    }
    assert cal.end == min(last_priced.values())
    assert cal.end < max(last_priced.values())


def test_dominant_fiscal_month_derived_not_hardcoded(tmp_path):
    """`make_financials`'s `year` labels are all `Mar YYYY` — March should
    fall out of the corpus scan, not merely match a hardcoded default."""
    root = tmp_path / "raw_data"
    write_ticker_dir(root, "AAA", years=10, quarters=13, shareholding_quarters=12, price_days=3200)

    cal = calendar_module.compute_calendar(root)

    assert cal.dominant_fiscal_month == 3
    # Every returned date is a March-grid quarter-end (month in
    # {3, 6, 9, 12}) pushed forward by the 2-month lag_months — the whole
    # point of deriving the grid.
    assert {d.month for d in cal.dates} <= {2, 5, 8, 11}


def test_battery_complete_recorded_and_rerating_lands_after_core(tmp_path):
    """Built so the asymmetry is true by construction: annual history goes
    back to ~2010 (financials `years=15`), while `quarterly`/`shareholding`
    are recent and shallow (8 / 4 rows) — exactly the corpus's own shape
    (deep annual history, shallow quarterly-grain frames) per KTD7.

    With `quarters=8` (the exact depth `growth_intact` needs) the frame's
    own 8 rows only become fully visible — and thus only clear the gate's
    contiguous-8 requirement — once every row is truncated in, which is
    later than `institutional_accumulation`'s 3-row requirement over
    `shareholding_quarters=4`. `core`'s battery is complete almost
    immediately (financials already exceed `MIN_TOTAL_YEARS` well before
    the 2023 replay start).
    """
    root = tmp_path / "raw_data"
    write_ticker_dir(
        root, "AAA",
        years=15, quarters=8, shareholding_quarters=4, price_days=3200,
    )

    cal = calendar_module.compute_calendar(root)

    core_date = cal.battery_complete[calendar_module.CORE_LANE]
    rerating_date = cal.battery_complete[calendar_module.RERATING_LANE]

    assert core_date is not None
    assert rerating_date is not None
    assert rerating_date > core_date
    # 15 years of annual history clears MIN_TOTAL_YEARS well before the
    # very first replay date — core's battery is complete immediately.
    assert core_date == cal.dates[0]

    # Detail is recorded, not just the bare date (U6 will want to attribute
    # a battery-complete reading to the ticker that supplied it).
    assert cal.battery_detail[calendar_module.CORE_LANE]["binding_ticker"] == "AAA"
    assert cal.battery_detail[calendar_module.RERATING_LANE]["binding_ticker"] == "AAA"


def test_battery_complete_is_none_when_never_reached(tmp_path):
    """A corpus with no `quarterly`/`shareholding` frames at all can never
    complete the fast lane's battery — `None`, not a crash or a fabricated
    date, is what `_battery_complete_rerating` must report."""
    root = tmp_path / "raw_data"
    write_ticker_dir(
        root, "AAA",
        years=10, quarters=None, shareholding_quarters=None, price_days=3200,
    )

    cal = calendar_module.compute_calendar(root)

    assert cal.battery_complete[calendar_module.RERATING_LANE] is None
    assert "reason" in cal.battery_detail[calendar_module.RERATING_LANE]
    # Core is unaffected — it never reads the quarterly-grain frames.
    assert cal.battery_complete[calendar_module.CORE_LANE] is not None


def test_no_tickers_raises_rather_than_returning_an_empty_calendar(tmp_path):
    root = tmp_path / "raw_data"
    root.mkdir()
    with pytest.raises(ValueError):
        calendar_module.compute_calendar(root)


def test_longest_trailing_contiguous_run():
    run = calendar_module._longest_trailing_contiguous_run
    assert run([]) == 0
    assert run([5]) == 1
    assert run([1, 2, 3, 4]) == 4
    # A gap before the trailing run does not extend it.
    assert run([1, 5, 6, 7]) == 3
    # Out-of-order / duplicate input is tolerated (de-duplicated then sorted).
    assert run([7, 6, 6, 5]) == 3


def test_replay_calendar_as_dict_is_json_friendly(tmp_path):
    root = tmp_path / "raw_data"
    write_ticker_dir(root, "AAA", years=10, quarters=13, shareholding_quarters=12, price_days=3200)
    cal = calendar_module.compute_calendar(root)

    rendered = cal.as_dict()
    assert isinstance(rendered["dates"], list)
    assert all(isinstance(d, str) for d in rendered["dates"])
    assert rendered["battery_complete"][calendar_module.CORE_LANE] is None or isinstance(
        rendered["battery_complete"][calendar_module.CORE_LANE], str
    )
