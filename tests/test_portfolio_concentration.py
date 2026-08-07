"""Concentration guardrails, counted in names rather than rupees (§8.1, §14.1).

KTD8 is the rule every assertion here rests on: the watchlist has never
recorded an invested amount, a tranche size, or a cost basis, and this phase
does not add one. So "no more than 10% of the sleeve in one name" has no
denominator to check against, and the guardrail that *can* be checked is a
count — how many positioned names a lane holds, and how many of them sit in the
same sector. A percentage would look more precise and mean nothing.

Two failure modes carry the unit.

The first is a position disappearing from a cap check because its analysis
failed. Counts are seeded from the **watchlist**, not from the run's successful
outcomes, so a fetch that broke for one holding costs that holding's *sector*
reading and nothing else — the position itself still counts against its lane.
A guardrail that silently stops seeing a position on the day its data went
missing is worse than no guardrail, because it reads as headroom.

The second is an unknown sector becoming a sector. A ticker fetched before the
breadcrumb fix carries no `metadata.sector`; grouping those together would
invent a "None" sector and either flag unrelated companies as correlated or,
worse, let a genuinely correlated pair hide inside it. Unknown is excluded from
grouping, listed, and logged — never treated as a sector of its own.
"""

import logging

import pytest

from boundless100x.lifecycle import portfolio, states
from boundless100x.lifecycle.advance import advance, advance_ticker
from boundless100x.lifecycle.evaluator import TriggerEvaluator, load_triggers
from boundless100x.service import AnalysisResult
from boundless100x.watchlist import WatchlistManager
from tests.test_lifecycle_advance import StubService, healthy_metrics

# A tight fast lane, so the plan's own verification case — three positioned
# re-rating names against a cap of two — is one fixture away in every test.
CONFIG = {
    "portfolio": {
        "max_positioned_per_lane": {"core": 8, "rerating": 2},
        "max_positioned_per_sector": 3,
    }
}


def entry(ticker, lane="core", state=states.PROBE, sector="Chemicals") -> dict:
    """One row in the list `advance()` builds for the reading."""
    return {"ticker": ticker, "lane": lane, "state": state, "sector": sector}


@pytest.fixture
def wm(tmp_path):
    return WatchlistManager(path=str(tmp_path / "watchlist.json"))


@pytest.fixture
def evaluator():
    return TriggerEvaluator(load_triggers())


def lane_of(reading, lane) -> dict:
    return reading["lanes"][lane]


def sector_named(reading, name) -> dict | None:
    for group in reading["sectors"]:
        if group["sector"].strip().lower() == name.strip().lower():
            return group
    return None


# ── Positioned counts ──────────────────────────────────────────────────────


class TestPositionedCounts:
    def test_only_positioned_states_count_toward_a_lane(self):
        """A candidate being watched is not capital at risk."""
        reading = portfolio.check_concentration([
            entry("A", state=states.SCREEN),
            entry("B", state=states.QUALIFY),
            entry("C", state=states.WATCH),
            entry("D", state=states.PROBE),
            entry("E", state=states.SCALE),
        ], CONFIG)

        assert lane_of(reading, "core")["positioned"] == 2
        assert lane_of(reading, "core")["tickers"] == ["D", "E"]

    @pytest.mark.parametrize("state", sorted(states.POSITIONED))
    def test_every_state_the_lifecycle_calls_positioned_counts(self, state):
        """Read off `states.POSITIONED` rather than a hand-listed pair.

        A state added to `POSITIONED` later must not have to be remembered here
        too — the failure would be a live position that no cap check can see.
        """
        reading = portfolio.check_concentration([entry("A", state=state)], CONFIG)
        assert lane_of(reading, "core")["positioned"] == 1

    def test_a_dropped_or_exited_name_holds_no_capital(self):
        reading = portfolio.check_concentration([
            entry("A", state=states.DROPPED),
            entry("B", state=states.EXITED),
        ], CONFIG)

        assert lane_of(reading, "core")["positioned"] == 0

    def test_positioned_means_what_the_lifecycle_says_it_means(self):
        """`exit_review` is outside `states.POSITIONED`, so it is outside this count.

        Arguably capital is still deployed while an exit is under review, and a
        cap check could reasonably say so. It does not, because `POSITIONED` is
        the lifecycle's single definition of a state holding capital and a
        second, quietly different one here would make two parts of the system
        disagree about how many positions exist. Changing the answer means
        changing `states.POSITIONED`, where every consumer sees it.
        """
        reading = portfolio.check_concentration([
            entry("A", state=states.EXIT_REVIEW),
        ], CONFIG)

        assert states.EXIT_REVIEW not in states.POSITIONED
        assert lane_of(reading, "core")["positioned"] == 0

    def test_each_lane_is_counted_separately(self):
        reading = portfolio.check_concentration([
            entry("A", lane="core"),
            entry("B", lane="rerating"),
            entry("C", lane="rerating"),
        ], CONFIG)

        assert lane_of(reading, "core")["positioned"] == 1
        assert lane_of(reading, "rerating")["positioned"] == 2

    def test_a_lane_with_a_configured_cap_appears_even_when_empty(self):
        """Headroom is a reading too — an absent lane looks like a missing check."""
        reading = portfolio.check_concentration([entry("A", lane="core")], CONFIG)

        assert lane_of(reading, "rerating")["positioned"] == 0
        assert lane_of(reading, "rerating")["breach"] is False


# ── Lane caps ──────────────────────────────────────────────────────────────


class TestLaneCaps:
    def test_a_lane_over_its_cap_is_a_breach(self):
        """The plan's verification case: three fast-lane names against a cap of two."""
        reading = portfolio.check_concentration([
            entry("A", lane="rerating"),
            entry("B", lane="rerating"),
            entry("C", lane="rerating"),
        ], CONFIG)

        assert lane_of(reading, "rerating")["breach"] is True
        assert lane_of(reading, "rerating")["max"] == 2
        assert any("rerating" in line for line in reading["breaches"])

    def test_a_lane_exactly_at_its_cap_is_not_a_breach(self):
        reading = portfolio.check_concentration([
            entry("A", lane="rerating"),
            entry("B", lane="rerating"),
        ], CONFIG)

        assert lane_of(reading, "rerating")["breach"] is False
        assert reading["breaches"] == []

    def test_a_lane_under_its_cap_is_not_a_breach(self):
        reading = portfolio.check_concentration([entry("A", lane="rerating")], CONFIG)
        assert lane_of(reading, "rerating")["breach"] is False

    def test_a_lane_with_no_configured_cap_says_so_rather_than_passing(self):
        reading = portfolio.check_concentration(
            [entry("A", lane="rerating")],
            {"portfolio": {"max_positioned_per_lane": {"core": 8}}},
        )

        assert lane_of(reading, "rerating")["max"] is None
        assert lane_of(reading, "rerating")["breach"] is False
        assert "no cap" in lane_of(reading, "rerating")["note"].lower()

    def test_the_shipped_config_supplies_the_caps_when_none_is_passed(self):
        """`check_concentration(entries)` must not silently mean "uncapped"."""
        reading = portfolio.check_concentration([entry("A", lane="core")])

        assert (
            lane_of(reading, "core")["max"]
            == portfolio.DEFAULT_MAX_POSITIONED_PER_LANE["core"]
        )

    def test_the_reading_states_that_it_counts_names(self):
        """KTD8, made unmissable: nothing here is a share of capital."""
        reading = portfolio.check_concentration([entry("A")], CONFIG)

        assert reading["basis"] == portfolio.BASIS_COUNTS
        assert "count" in portfolio.describe(reading).lower()


# ── Sector repetition ──────────────────────────────────────────────────────


class TestSectorGroups:
    def test_two_positioned_names_in_one_sector_are_grouped(self):
        """The plan's second verification case: the same-sector correlation note."""
        reading = portfolio.check_concentration([
            entry("A", sector="Chemicals"),
            entry("B", sector="Chemicals"),
        ], CONFIG)

        group = sector_named(reading, "Chemicals")
        assert group is not None
        assert group["count"] == 2
        assert group["tickers"] == ["A", "B"]
        assert any("Chemicals" in note for note in reading["notes"])

    def test_a_single_name_in_a_sector_is_not_flagged(self):
        reading = portfolio.check_concentration([
            entry("A", sector="Chemicals"),
            entry("B", sector="Software"),
        ], CONFIG)

        assert reading["sectors"] == []
        assert reading["breaches"] == []

    def test_a_group_within_the_sector_cap_is_reported_but_is_not_a_breach(self):
        """Correlation is worth saying out loud before it is worth stopping."""
        reading = portfolio.check_concentration([
            entry("A", sector="Chemicals"),
            entry("B", sector="Chemicals"),
        ], CONFIG)

        assert sector_named(reading, "Chemicals")["breach"] is False
        assert reading["breaches"] == []

    def test_a_group_over_the_sector_cap_is_a_breach(self):
        reading = portfolio.check_concentration([
            entry("A", sector="Chemicals"),
            entry("B", sector="Chemicals"),
            entry("C", sector="Chemicals"),
            entry("D", sector="Chemicals"),
        ], CONFIG)

        assert sector_named(reading, "Chemicals")["breach"] is True
        assert any("Chemicals" in line for line in reading["breaches"])

    def test_sectors_are_grouped_across_lanes(self):
        """Correlation does not respect the sleeve it was bought into."""
        reading = portfolio.check_concentration([
            entry("A", lane="core", sector="Chemicals"),
            entry("B", lane="rerating", sector="Chemicals"),
        ], CONFIG)

        assert sector_named(reading, "Chemicals")["count"] == 2

    def test_an_unpositioned_name_does_not_join_a_sector_group(self):
        reading = portfolio.check_concentration([
            entry("A", sector="Chemicals"),
            entry("B", state=states.WATCH, sector="Chemicals"),
        ], CONFIG)

        assert reading["sectors"] == []

    def test_the_same_sector_written_differently_is_one_group(self):
        """Screener's breadcrumb is free text; casing and spacing drift."""
        reading = portfolio.check_concentration([
            entry("A", sector="Chemicals"),
            entry("B", sector=" chemicals "),
        ], CONFIG)

        assert sector_named(reading, "Chemicals")["count"] == 2


class TestUnknownSector:
    def test_a_name_with_no_sector_is_excluded_from_grouping_and_logged(self, caplog):
        """A pre-breadcrumb-fix fetch carries no `metadata.sector`."""
        with caplog.at_level(logging.INFO):
            reading = portfolio.check_concentration([
                entry("A", sector=None),
                entry("B", sector=None),
            ], CONFIG)

        assert reading["sectors"] == []
        assert reading["unknown_sector"] == ["A", "B"]
        assert "sector" in caplog.text.lower()

    def test_an_unknown_sector_still_counts_toward_its_lane(self):
        """Not knowing what a company does is no reason to stop counting it."""
        reading = portfolio.check_concentration([
            entry("A", lane="rerating", sector=None),
            entry("B", lane="rerating", sector=None),
            entry("C", lane="rerating", sector=None),
        ], CONFIG)

        assert lane_of(reading, "rerating")["positioned"] == 3
        assert lane_of(reading, "rerating")["breach"] is True

    def test_an_unknown_sector_never_hides_a_real_group(self):
        reading = portfolio.check_concentration([
            entry("A", sector=None),
            entry("B", sector="Chemicals"),
            entry("C", sector="Chemicals"),
        ], CONFIG)

        assert sector_named(reading, "Chemicals")["count"] == 2
        assert "A" not in sector_named(reading, "Chemicals")["tickers"]

    def test_a_blank_sector_string_reads_as_unknown(self):
        reading = portfolio.check_concentration([
            entry("A", sector="  "),
            entry("B", sector=""),
        ], CONFIG)

        assert reading["sectors"] == []
        assert reading["unknown_sector"] == ["A", "B"]


# ── Through advance() ──────────────────────────────────────────────────────


class SectorService(StubService):
    """A stub whose analysis carries a per-ticker `metadata.sector`.

    `advance_ticker` reads the sector off the analysis result rather than the
    watchlist, per KTD8 — the reading is resolved once per run, exactly as the
    deployment-pace modulator resolves its corpus median, instead of becoming a
    stored field that can go stale between fetches.
    """

    def __init__(self, sectors=None, **kwargs):
        super().__init__(**kwargs)
        self._sectors = sectors or {}

    def analyze(self, ticker, **kwargs):
        result = super().analyze(ticker, **kwargs)
        sector = self._sectors.get(ticker)
        result.data = {"metadata": {"sector": sector}} if sector else {}
        return result


def positioned(wm, ticker, lane="core", state=states.PROBE):
    wm.add(ticker, lane=lane)
    wm.transition(ticker, state, "seed", evidence="test seed")
    return ticker


class TestThroughAdvance:
    def service(self, sectors=None, config=None):
        service = SectorService(sectors=sectors, metrics=healthy_metrics())
        service.config = config if config is not None else CONFIG
        return service

    def test_advance_returns_a_concentration_reading_beside_pace(self, wm, evaluator):
        positioned(wm, "ASTRAL")
        out = advance(self.service({"ASTRAL": "Chemicals"}), wm, evaluator=evaluator)

        assert out["concentration"]["available"] is True
        assert lane_of(out["concentration"], "core")["positioned"] == 1

    def test_advance_ticker_reports_the_sector_from_the_analysis_metadata(
        self, wm, evaluator
    ):
        positioned(wm, "ASTRAL")
        outcome = advance_ticker(
            self.service({"ASTRAL": "Chemicals"}), wm, "ASTRAL", evaluator
        )

        assert outcome["sector"] == "Chemicals"

    def test_a_ticker_whose_analysis_carries_no_metadata_reports_no_sector(
        self, wm, evaluator
    ):
        positioned(wm, "ASTRAL")
        outcome = advance_ticker(self.service(), wm, "ASTRAL", evaluator)

        assert outcome["sector"] is None

    def test_a_positioned_ticker_whose_analysis_errored_still_counts(
        self, wm, evaluator
    ):
        """The seeding rule, stated as a test.

        Counts come from the watchlist, so a failed fetch costs the sector
        reading and nothing else. Seeded from the run's outcomes instead, a
        broken fetch would have quietly opened up headroom in a lane that is
        already full.
        """
        positioned(wm, "GOOD")
        positioned(wm, "BAD")

        class Flaky(SectorService):
            def analyze(self, ticker, **kwargs):
                if ticker == "BAD":
                    raise RuntimeError("fetch failed")
                return super().analyze(ticker, **kwargs)

        service = Flaky(sectors={"GOOD": "Chemicals"}, metrics=healthy_metrics())
        service.config = CONFIG

        out = advance(service, wm, evaluator=evaluator)
        reading = out["concentration"]

        assert out["errors"] == [("BAD", "fetch failed")]
        assert lane_of(reading, "core")["positioned"] == 2
        assert "BAD" in lane_of(reading, "core")["tickers"]
        # Its sector is the only thing the failure cost.
        assert reading["unknown_sector"] == ["BAD"]
        assert reading["sectors"] == []

    def test_a_fast_lane_breach_is_reported_through_advance(self, wm, evaluator):
        """The plan's verification case, end to end."""
        for ticker in ("ONE", "TWO", "THREE"):
            positioned(wm, ticker, lane="rerating")

        out = advance(self.service(), wm, evaluator=evaluator)

        assert lane_of(out["concentration"], "rerating")["breach"] is True
        assert out["concentration"]["breaches"]

    def test_the_same_sector_note_is_reported_through_advance(self, wm, evaluator):
        positioned(wm, "ASTRAL")
        positioned(wm, "SUPREME")
        service = self.service({"ASTRAL": "Chemicals", "SUPREME": "Chemicals"})

        out = advance(service, wm, evaluator=evaluator)

        assert sector_named(out["concentration"], "Chemicals")["count"] == 2

    def test_a_quarterly_run_still_counts_the_names_it_did_not_advance(
        self, wm, evaluator
    ):
        """`--quarterly` advances a stale subset; the portfolio is not a subset.

        Seeding from the watchlist gives this for free, which is the point —
        the same property that survives a failed fetch survives a partial run.
        """
        for ticker in ("ONE", "TWO", "THREE"):
            positioned(wm, ticker, lane="rerating")
            # A never-scored entry reads stale, so freshness has to be recorded
            # for `--quarterly` to skip anything at all.
            wm.record_snapshot(ticker, AnalysisResult(ticker=ticker), "abc123")

        out = advance(self.service(), wm, evaluator=evaluator, quarterly=True)

        assert out["outcomes"] == []
        assert lane_of(out["concentration"], "rerating")["positioned"] == 3
        assert lane_of(out["concentration"], "rerating")["breach"] is True

    def test_a_failed_reading_costs_the_reading_not_the_run(
        self, wm, evaluator, monkeypatch
    ):
        """Computed after every ticker has been advanced — it must not throw that away."""
        positioned(wm, "ASTRAL")

        def boom(*args, **kwargs):
            raise RuntimeError("counting broke")

        monkeypatch.setattr(portfolio, "check_concentration", boom)

        out = advance(self.service(), wm, evaluator=evaluator)

        assert out["outcomes"][0]["ticker"] == "ASTRAL"
        assert out["concentration"]["available"] is False
        assert "counting broke" in out["concentration"]["reason"]

    def test_an_empty_watchlist_reads_as_no_positions_rather_than_a_gap(
        self, wm, evaluator
    ):
        out = advance(self.service(), wm, evaluator=evaluator)

        assert out["concentration"]["available"] is True
        assert out["concentration"]["positioned"] == 0
        assert out["concentration"]["breaches"] == []


# ── What the owner actually reads ──────────────────────────────────────────


class TestCli:
    """The rendered line, captured the way `_print_exit_friction` is tested."""

    def render(self, reading) -> str:
        from boundless100x import cli

        with cli.console.capture() as captured:
            cli._print_concentration(reading)
        return captured.get()

    def test_the_line_names_its_basis_so_it_cannot_read_as_a_percentage(self):
        reading = portfolio.check_concentration([entry("A")], CONFIG)
        text = self.render(reading)

        assert "count" in text.lower()
        assert "%" not in text

    def test_a_breach_is_shown_with_the_names_that_caused_it(self):
        reading = portfolio.check_concentration([
            entry("ONE", lane="rerating"),
            entry("TWO", lane="rerating"),
            entry("THREE", lane="rerating"),
        ], CONFIG)
        text = self.render(reading)

        assert "rerating" in text
        assert "THREE" in text

    def test_the_same_sector_note_reaches_the_owner(self):
        reading = portfolio.check_concentration([
            entry("A", sector="Chemicals"),
            entry("B", sector="Chemicals"),
        ], CONFIG)

        assert "Chemicals" in self.render(reading)

    def test_a_bracketed_sector_name_survives_the_markup_parser(self):
        """Sector text is scraped, so it can contain anything rich reads as a tag."""
        reading = portfolio.check_concentration([
            entry("A", sector="Chemicals [dim]"),
            entry("B", sector="Chemicals [dim]"),
        ], CONFIG)

        assert "Chemicals [dim]" in self.render(reading)

    def test_an_unavailable_reading_is_shown_rather_than_skipped(self):
        text = self.render(portfolio.unavailable("counting broke"))

        assert "unavailable" in text.lower()
        assert "counting broke" in text

    def test_nothing_is_printed_when_nothing_is_positioned(self):
        """A watchlist of candidates has no concentration to report."""
        reading = portfolio.check_concentration([entry("A", state=states.WATCH)], CONFIG)

        assert self.render(reading).strip() == ""
