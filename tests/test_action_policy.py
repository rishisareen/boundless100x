"""The displayed action must respect the deterministic 100x verdict.

Pass 2 returns a `suggested_action` having never been shown the eligibility
verdict, and the report renders that action directly beside the verdict badge.
Nothing stopped a report showing "Not a 100x Candidate" and "STRONG BUY"
together. The guard is code, not prompt compliance.

Capping is not overriding: failing a gate makes a company an unlikely
hundred-bagger, not a bad investment, so a constrained action is lowered to
`watchlist` rather than flipped to `avoid`, and the model's own action is
preserved beside it.
"""

import pytest

from boundless100x.action_policy import (
    CAP_CEILING,
    resolve_final_action,
)


def eligibility(verdict: str, **kwargs) -> dict:
    base = {
        "eligible": {"eligible": True, "verdict": "eligible", "gates": {}, "failed": [], "indeterminate": []},
        "not_eligible": {
            "eligible": False, "verdict": "not_eligible",
            "gates": {"size": {"passed": False, "reason": "Size headroom: market cap 95,000 exceeds 30,000"}},
            "failed": ["size"], "indeterminate": [],
        },
        "indeterminate": {
            "eligible": None, "verdict": "indeterminate",
            "gates": {"price": {"passed": None, "reason": "Entry price sanity: reverse DCF unavailable"}},
            "failed": [], "indeterminate": ["price"],
        },
    }[verdict]
    return {**base, **kwargs}


def scores(*flags, coverage: float = 0.95) -> dict:
    return {"composite": 7.0, "flags": list(flags), "coverage": {"composite": coverage}}


class TestEligibleVerdictPassesThrough:
    @pytest.mark.parametrize("action", ["strong_buy", "buy", "hold", "watchlist", "avoid"])
    def test_action_is_untouched_when_every_gate_clears(self, action):
        decision = resolve_final_action(action, eligibility("eligible"), scores())

        assert decision["action"] == action
        assert decision["capped"] is False
        assert decision["constraints"] == []


class TestFailedVerdictCapsTheAction:
    @pytest.mark.parametrize("action", ["strong_buy", "buy"])
    def test_buy_side_actions_are_capped_to_watchlist(self, action):
        decision = resolve_final_action(action, eligibility("not_eligible"), scores())

        assert decision["action"] == CAP_CEILING
        assert decision["capped"] is True
        assert decision["ceiling"] == CAP_CEILING

    def test_the_models_own_action_is_preserved_not_erased(self):
        decision = resolve_final_action("strong_buy", eligibility("not_eligible"), scores())

        assert decision["llm_action"] == "strong_buy"

    def test_the_failing_gate_reason_is_carried_as_the_explanation(self):
        decision = resolve_final_action("buy", eligibility("not_eligible"), scores())

        assert any("30,000" in reason for reason in decision["constraints"])

    @pytest.mark.parametrize("action", ["watchlist", "avoid"])
    def test_actions_already_at_or_below_the_ceiling_are_not_raised(self, action):
        """A cap lowers; it must never promote `avoid` up to `watchlist`."""
        decision = resolve_final_action(action, eligibility("not_eligible"), scores())

        assert decision["action"] == action
        assert decision["capped"] is False

    def test_a_constrained_but_uncapped_action_still_reports_why(self):
        """`avoid` needed no cap, but the reader still gets the gate reason."""
        decision = resolve_final_action("avoid", eligibility("not_eligible"), scores())

        assert decision["capped"] is False
        assert decision["constraints"]

    def test_hold_sits_above_the_ceiling_and_is_capped(self):
        decision = resolve_final_action("hold", eligibility("not_eligible"), scores())

        assert decision["action"] == CAP_CEILING
        assert decision["capped"] is True


class TestIndeterminateVerdictAlsoCaps:
    def test_unknown_eligibility_is_not_treated_as_eligible(self):
        decision = resolve_final_action("strong_buy", eligibility("indeterminate"), scores())

        assert decision["action"] == CAP_CEILING
        assert decision["capped"] is True

    def test_the_unevaluated_gate_reason_is_reported(self):
        decision = resolve_final_action("buy", eligibility("indeterminate"), scores())

        assert any("reverse DCF" in reason for reason in decision["constraints"])


class TestMissingEligibilityFailsClosed:
    """Stage 3.6 catches its own exceptions, so a missing verdict means the
    gates never ran — not that they passed."""

    @pytest.mark.parametrize("missing", [None, {}, {"gates": {}}])
    def test_absent_verdict_caps_rather_than_waves_through(self, missing):
        decision = resolve_final_action("strong_buy", missing, scores())

        assert decision["action"] == CAP_CEILING
        assert decision["capped"] is True
        assert any("not evaluated" in r for r in decision["constraints"])


class TestLowCoverageCaps:
    def test_thin_evidence_caps_an_unqualified_buy(self):
        decision = resolve_final_action(
            "strong_buy", eligibility("eligible"), scores("low_data_coverage", coverage=0.61)
        )

        assert decision["action"] == CAP_CEILING
        assert decision["capped"] is True

    def test_the_coverage_shortfall_is_quantified_for_the_reader(self):
        decision = resolve_final_action(
            "buy", eligibility("eligible"), scores("low_data_coverage", coverage=0.61)
        )

        assert any("61%" in reason for reason in decision["constraints"])

    def test_coverage_and_eligibility_constraints_both_surface(self):
        decision = resolve_final_action(
            "strong_buy", eligibility("not_eligible"), scores("low_data_coverage", coverage=0.5)
        )

        assert len(decision["constraints"]) == 2


class TestDegenerateInputs:
    def test_no_llm_action_yields_no_action(self):
        decision = resolve_final_action(None, eligibility("not_eligible"), scores())

        assert decision["action"] is None
        assert decision["capped"] is False

    def test_an_unrecognised_action_is_capped_rather_than_trusted(self):
        """A model returning something off-schema must not bypass the ceiling."""
        decision = resolve_final_action("ACCUMULATE NOW", eligibility("not_eligible"), scores())

        assert decision["action"] == CAP_CEILING
        assert decision["capped"] is True

    def test_case_and_whitespace_do_not_defeat_the_ranking(self):
        decision = resolve_final_action("  Avoid  ", eligibility("not_eligible"), scores())

        assert decision["action"] == "  Avoid  "
        assert decision["capped"] is False

    def test_missing_scores_do_not_crash_the_guard(self):
        decision = resolve_final_action("strong_buy", eligibility("eligible"), None)

        assert decision["action"] == "strong_buy"
