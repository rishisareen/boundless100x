"""What happens to a metric the sector table withdraws — beyond its score.

Wiring the applicability table into the scorer removed the false marks out of
ten. Three consequences of that were left unhandled and all three surfaced on
JIOFIN, a lending group filed under "Investment Company" whose balance sheet is
81% other companies' equity:

  * a withdrawn metric that ALSO errored was counted as missing evidence,
    because the error branch ran first — and for a lender the two arrive
    together as a rule, since `reverse_dcf_growth` errors on every lender
    growing its book;
  * its FLAGS went on rendering, so "Cash Cow — Strong Cash Conversion" led the
    Strengths list of a company with -₹15,439 Cr of operating cash flow;
  * the eligibility GATES read it anyway, because they consult metric results
    rather than scores, leaving the entry-price gate permanently indeterminate.
"""

import pytest

from boundless100x.compute_engine.eligibility import EligibilityEvaluator
from boundless100x.compute_engine.metrics.base import MetricResult
from boundless100x.compute_engine.scorer import SQGLPScorer
from boundless100x.compute_engine.sector import SectorApplicability

CONFIGS = {
    "solid": {"element": "price", "name": "Solid",
              "scoring": {"weight": 0.5, "thresholds": [1, 2, 3, 4, 5, 6],
                          "direction": "higher_is_better"}},
    "broken_here": {"element": "price", "name": "Broken Here",
                    "scoring": {"weight": 0.5, "thresholds": [1, 2, 3, 4, 5, 6],
                                "direction": "higher_is_better"}},
}
WEIGHTS = {"price": 1.0}

TABLE = {
    "Lenders": {
        "label": "Lenders",
        "not_applicable": {
            "broken_here": "Measures the loan book's direction, not a return.",
        },
    },
    "Keeps": {
        "label": "Keeps its warnings",
        "not_applicable": {
            "broken_here": {
                "keep_flags": True,
                "reason": "Miscalibrated here, but the reading itself is real.",
            },
        },
    },
}


def scorer(table=TABLE):
    return SQGLPScorer(
        CONFIGS, WEIGHTS,
        applicability=SectorApplicability(set(CONFIGS), table),
    )


class TestAWithdrawnMetricThatAlsoErrored:
    def test_it_leaves_the_coverage_denominator_rather_than_reading_as_a_gap(self):
        """The ordering bug. `reverse_dcf_growth` is declared meaningless for a
        lender *because* it cannot compute for one — so counting its failure as
        missing evidence penalises the company twice for one fact."""
        results = {
            "solid": MetricResult(value=5.0),
            "broken_here": MetricResult(error="Negative average FCF"),
        }

        scores = scorer().score(results, sector="Lenders")

        assert scores["coverage"]["elements"]["price"] == 1.0
        assert scores["coverage"]["unscored"] == []
        assert "broken_here" in scores["not_applicable"]

    def test_the_error_still_travels_for_a_reader(self):
        """Withdrawn is not the same as unexamined; the row still says what
        happened when someone tried."""
        results = {
            "solid": MetricResult(value=5.0),
            "broken_here": MetricResult(error="Negative average FCF"),
        }

        detail = scorer().score(results, sector="Lenders")["details"]["broken_here"]

        assert detail["not_applicable"]
        assert detail["error"] == "Negative average FCF"

    def test_a_metric_that_merely_errored_is_still_a_gap(self):
        """The control. Nothing here forgives ordinary missing data."""
        results = {
            "solid": MetricResult(value=5.0),
            "broken_here": MetricResult(error="No data"),
        }

        scores = scorer().score(results, sector="Unreviewed Sector")

        assert scores["coverage"]["elements"]["price"] == 0.5
        assert scores["coverage"]["unscored"] == ["broken_here"]


class TestFlagsOfAWithdrawnMetric:
    def test_they_are_suppressed_by_default(self):
        """A metric that measures nothing here says nothing here."""
        results = {
            "solid": MetricResult(value=5.0),
            "broken_here": MetricResult(value=981.0, flags=["cash_cow"]),
        }

        scores = scorer().score(results, sector="Lenders")

        assert scores["details"]["broken_here"]["flags"] == []
        assert scores["flags_suppressed"] == ["broken_here"]

    def test_an_entry_may_keep_them_and_must_say_so(self):
        """Debt/equity's case: four times leverage is four times leverage
        whatever we score it out of ten, and the warning is the only one a
        lender's owner gets."""
        results = {
            "solid": MetricResult(value=5.0),
            "broken_here": MetricResult(value=4.0, flags=["debt_risk"]),
        }

        scores = scorer().score(results, sector="Keeps")

        assert scores["details"]["broken_here"]["flags"] == ["debt_risk"]
        assert scores["flags_suppressed"] == []

    def test_the_shipped_table_keeps_exactly_the_leverage_warnings(self):
        """Read off the real file: those two argue for it in their entries,
        and nothing else should be quietly keeping its flags."""
        from boundless100x.compute_engine.engine import ComputeEngine

        metrics = ComputeEngine().metrics
        applicability = SectorApplicability(set(metrics))
        excluded = set(applicability.not_applicable_metrics("Finance"))
        suppressed = applicability.flag_suppressed_metrics("Finance")

        assert excluded - suppressed == {"debt_equity", "interest_coverage"}


class TestGatesRespectTheSameWithdrawal:
    GATE = {
        "price": {
            "label": "Entry price sanity",
            "mode": "any",
            "conditions": [{"metric": "solid", "comparator": "lt", "threshold": 2.0}],
            "veto_flags": ["overpriced"],
            "veto_sources": ["broken_here"],
        }
    }

    def test_an_unavailable_veto_source_still_blocks_when_it_applies(self):
        """Unchanged behaviour for everyone else: absence of a veto proves
        nothing when the metric that would raise it never ran."""
        metrics = {
            "solid": MetricResult(value=1.0),
            "broken_here": MetricResult(error="Negative average FCF"),
        }

        outcome = EligibilityEvaluator(self.GATE).evaluate(metrics)

        assert outcome["verdict"] == "indeterminate"

    def test_a_withdrawn_veto_source_does_not_make_the_gate_indeterminate(self):
        """JIOFIN's case. A veto that cannot fire meaningfully for this kind of
        company must not refuse it forever either."""
        metrics = {
            "solid": MetricResult(value=1.0),
            "broken_here": MetricResult(error="Negative average FCF"),
        }

        outcome = EligibilityEvaluator(self.GATE).evaluate(
            metrics, not_applicable={"broken_here"}
        )

        assert outcome["verdict"] == "eligible"

    def test_a_veto_flag_from_a_withdrawn_metric_does_not_disqualify(self):
        """The other direction, and the more dangerous one: a flag off a
        meaningless reading must not refuse a company outright."""
        metrics = {
            "solid": MetricResult(value=1.0),
            "broken_here": MetricResult(value=-10.0, flags=["overpriced"]),
        }

        assert EligibilityEvaluator(self.GATE).evaluate(metrics)["verdict"] == "not_eligible"
        assert EligibilityEvaluator(self.GATE).evaluate(
            metrics, not_applicable={"broken_here"}
        )["verdict"] == "eligible"

    def test_a_gate_will_not_read_an_unscorable_value(self):
        """Both layers, one list. A figure the scorer refuses to score must not
        admit a company through a gate — JIOFIN's 0.29x trailing PEG, off a
        post-demerger base, cleared the entry-price gate on its own."""
        gate = {
            "price": {
                "label": "Entry price sanity",
                "conditions": [
                    {"metric": "solid", "comparator": "lt", "threshold": 2.0}
                ],
            }
        }
        metrics = {
            "solid": MetricResult(value=0.29, flags=["cagr_off_negligible_base"])
        }

        outcome = EligibilityEvaluator(gate).evaluate(metrics)

        assert outcome["verdict"] == "indeterminate"
        assert "not a usable reading" in outcome["gates"]["price"]["reason"]


class TestTheTableMayBeKeyedOnIndustry:
    def test_both_labels_are_matched_and_their_exclusions_merge(self):
        """JIOFIN's sector reads "Finance" and its industry "Investment
        Company", and rules exist for both. Matching only the sector would
        leave the narrower entry unreachable for every company it describes."""
        table = {
            "Finance": {"label": "F", "not_applicable": {"solid": "Lender reason."}},
            "Investment Company": {
                "label": "I", "not_applicable": {"broken_here": "Holdco reason."}
            },
        }
        applicable = SectorApplicability(set(CONFIGS), table)

        merged = applicable.not_applicable_metrics(("Finance", "Investment Company"))

        assert set(merged) == {"solid", "broken_here"}

    def test_one_label_still_works_unchanged(self):
        applicable = SectorApplicability(set(CONFIGS), TABLE)

        assert set(applicable.not_applicable_metrics("Lenders")) == {"broken_here"}

    def test_the_shipped_table_reaches_a_holdco_through_its_industry(self):
        from boundless100x.compute_engine.engine import ComputeEngine

        metrics = ComputeEngine().metrics
        applicable = SectorApplicability(set(metrics))

        sector_only = set(applicable.not_applicable_metrics("Finance"))
        with_industry = set(
            applicable.not_applicable_metrics(("Finance", "Investment Company"))
        )

        assert with_industry > sector_only
        assert "operating_margin_5yr" in with_industry - sector_only


class TestFlagsDerivedFromAnArtefact:
    """A flag computed off a value we refuse to score is refused with it.

    This survived the first pass of the fix, which suppressed flags from
    sector-*withdrawn* metrics but not from *unscorable* ones. JIOFIN's
    Strengths list therefore opened with "Attractive Trailing PEG" — read off
    the 0.29x PEG that the same report had already declined to score and the
    gates had already declined to read.
    """

    def _scores(self):
        from boundless100x.output.report_generator import _flag_bearing_metrics
        return _flag_bearing_metrics({
            "flags_suppressed": ["withdrawn_one"],
            "details": {
                "artefact": {"waived": "not_a_reading"},
                "sound": {"score": 0.5},
            },
        })

    def test_the_two_kinds_are_reported_separately(self):
        suppressed, unscorable = self._scores()

        assert suppressed == {"withdrawn_one"}
        assert unscorable == {"artefact"}

    def test_the_report_drops_a_favourable_flag_off_an_artefact(self):
        from boundless100x.output.report_generator import ReportGenerator

        flags = ReportGenerator(output_dir=".")._collect_flags(
            {"artefact": MetricResult(
                value=0.29,
                flags=["attractive_trailing_peg", "cagr_off_negligible_base"],
            )},
            {"details": {"artefact": {"waived": "not_a_reading"}}},
        )

        raw = {f["raw"] for f in flags}
        assert "attractive_trailing_peg" not in raw
        # The flag that explains the refusal is exactly what to keep.
        assert "cagr_off_negligible_base" in raw

    def test_the_llm_context_applies_the_identical_rule(self):
        """Model and reader must be shown the same evidence."""
        from boundless100x.llm_layer.checklist import build_flags_context

        rendered = build_flags_context(
            {"artefact": MetricResult(
                value=0.29,
                flags=["attractive_trailing_peg", "cagr_off_negligible_base"],
            )},
            {"details": {"artefact": {"waived": "not_a_reading"}}},
        )

        assert "attractive_trailing_peg" not in rendered
        assert "cagr_off_negligible_base" in rendered

    def test_a_sound_metrics_flags_are_untouched(self):
        from boundless100x.output.report_generator import ReportGenerator

        flags = ReportGenerator(output_dir=".")._collect_flags(
            {"sound": MetricResult(value=5.0, flags=["debt_risk"])},
            {"details": {"sound": {"score": 0.5}}},
        )

        assert {f["raw"] for f in flags} == {"debt_risk"}
