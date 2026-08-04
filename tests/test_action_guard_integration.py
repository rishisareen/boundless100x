"""The action guard must hold on both paths that can display an action.

`action_policy` decides correctly in isolation; these tests check the two
places a wrong answer would actually reach a reader — the service resolving
it after Pass 2, and the report generator rendering it beside the eligibility
badge. The originally-reported defect was exactly this pairing: a dashboard
showing "Not a 100x Candidate" and "STRONG BUY" at once.
"""

import pytest

from boundless100x.output.report_generator import ReportGenerator
from boundless100x.service import Boundless100xService
from tests.conftest import make_result, make_scores


def llm_analysis(action: str = "strong_buy", **p2_extra) -> dict:
    return {
        "pass1": {"skipped": True},
        "pass2": {
            "thesis": "A quality compounder.",
            "conviction_level": "high",
            "suggested_action": action,
            "target_holding_period": "10yr+",
            **p2_extra,
        },
        "usage": {"total_tokens": 100},
    }


def failed_eligibility() -> dict:
    return {
        "eligible": False,
        "verdict": "not_eligible",
        "gates": {"size": {"passed": False, "reason": "Size headroom: market cap too large"}},
        "failed": ["size"],
        "indeterminate": [],
    }


def clean_eligibility() -> dict:
    return {
        "eligible": True, "verdict": "eligible",
        "gates": {"size": {"passed": True, "reason": "Size headroom: clears"}},
        "failed": [], "indeterminate": [],
    }


def result_with(action: str, eligibility: dict | None, scores: dict | None = None):
    result = make_result(scores=scores if scores is not None else make_scores())
    result.llm_analysis = llm_analysis(action)
    result.eligibility = eligibility
    return result


class TestServiceResolvesTheAction:
    def test_failed_gate_caps_a_strong_buy(self):
        decision = Boundless100xService.resolve_action(
            result_with("strong_buy", failed_eligibility())
        )

        assert decision["action"] == "watchlist"
        assert decision["capped"] is True
        assert decision["llm_action"] == "strong_buy"

    def test_clean_verdict_leaves_a_strong_buy_alone(self):
        decision = Boundless100xService.resolve_action(
            result_with("strong_buy", clean_eligibility())
        )

        assert decision["action"] == "strong_buy"
        assert decision["capped"] is False

    def test_missing_eligibility_caps(self):
        decision = Boundless100xService.resolve_action(result_with("buy", None))

        assert decision["action"] == "watchlist"
        assert decision["capped"] is True

    def test_no_llm_analysis_yields_no_decision(self):
        result = make_result()
        result.eligibility = clean_eligibility()

        assert Boundless100xService.resolve_action(result) is None

    def test_errored_pass2_yields_no_decision(self):
        result = make_result()
        result.eligibility = clean_eligibility()
        result.llm_analysis = {"pass2": {"error": "API timeout"}}

        assert Boundless100xService.resolve_action(result) is None

    def test_skipped_llm_yields_no_decision(self):
        result = make_result()
        result.llm_analysis = {"skipped": True, "reason": "LLM disabled"}

        assert Boundless100xService.resolve_action(result) is None


class TestReportNeverShowsAnUnguardedAction:
    """The defect as originally reported: verdict badge and action badge
    rendered from different sources, free to contradict each other."""

    def build(self, tmp_path, result) -> dict:
        return ReportGenerator(output_dir=str(tmp_path))._build_executive_summary(result)

    def test_failed_verdict_and_strong_buy_cannot_appear_together(self, tmp_path):
        summary = self.build(tmp_path, result_with("strong_buy", failed_eligibility()))

        assert summary["eligibility"]["verdict"] == "not_eligible"
        assert summary["suggested_action"] == "watchlist"
        assert summary["suggested_action"] != "strong_buy"

    def test_indeterminate_verdict_also_caps(self, tmp_path):
        indeterminate = {
            "eligible": None, "verdict": "indeterminate",
            "gates": {"price": {"passed": None, "reason": "Entry price sanity: unavailable"}},
            "failed": [], "indeterminate": ["price"],
        }

        summary = self.build(tmp_path, result_with("buy", indeterminate))

        assert summary["eligibility"]["verdict"] == "indeterminate"
        assert summary["suggested_action"] == "watchlist"

    def test_eligible_verdict_renders_the_models_action_unchanged(self, tmp_path):
        summary = self.build(tmp_path, result_with("strong_buy", clean_eligibility()))

        assert summary["suggested_action"] == "strong_buy"
        assert summary["action_constraint"]["capped"] is False

    def test_the_reason_travels_with_the_capped_action(self, tmp_path):
        summary = self.build(tmp_path, result_with("strong_buy", failed_eligibility()))

        constraint = summary["action_constraint"]
        assert constraint["llm_action"] == "strong_buy"
        assert any("Size headroom" in r for r in constraint["constraints"])

    def test_guard_holds_when_the_service_never_resolved_one(self, tmp_path):
        """A hand-built AnalysisResult (final_action unset) must not slip an
        unchecked action past the badge — the generator recomputes it."""
        result = result_with("strong_buy", failed_eligibility())
        assert result.final_action is None

        summary = self.build(tmp_path, result)

        assert summary["suggested_action"] == "watchlist"

    def test_a_stored_decision_contradicting_the_verdict_is_not_trusted(self, tmp_path):
        """`final_action` is an output of the policy, never an input to it.

        Trusting a pre-populated decision would make the guard only as strong
        as whoever set the field — a stale one left by a rescore, or one on a
        hand-built result, would render straight through beside the verdict it
        contradicts.
        """
        result = result_with("strong_buy", failed_eligibility())
        result.final_action = {
            "action": "strong_buy", "llm_action": "strong_buy",
            "capped": False, "ceiling": None, "constraints": [],
        }

        summary = self.build(tmp_path, result)

        assert summary["eligibility"]["verdict"] == "not_eligible"
        assert summary["suggested_action"] == "watchlist"

    def test_a_stale_decision_is_replaced_by_the_current_one(self, tmp_path):
        """Eligibility changed after Stage 4.5 ran; the render must follow the
        current inputs, not the decision recorded against the old ones."""
        result = result_with("strong_buy", clean_eligibility())
        result.final_action = {
            "action": "strong_buy", "llm_action": "strong_buy",
            "capped": False, "ceiling": None, "constraints": [],
        }
        result.eligibility = failed_eligibility()   # rescored afterwards

        summary = self.build(tmp_path, result)

        assert summary["suggested_action"] == "watchlist"
        assert summary["action_constraint"]["capped"] is True

    def test_a_disagreeing_stored_decision_is_logged_not_swallowed(self, tmp_path, caplog):
        result = result_with("strong_buy", failed_eligibility())
        result.final_action = {
            "action": "strong_buy", "llm_action": "strong_buy",
            "capped": False, "ceiling": None, "constraints": [],
        }

        with caplog.at_level("WARNING"):
            self.build(tmp_path, result)

        assert any("disagrees" in r.message for r in caplog.records)

    def test_an_agreeing_stored_decision_logs_nothing(self, tmp_path, caplog):
        result = result_with("strong_buy", clean_eligibility())
        result.final_action = Boundless100xService.resolve_action(result)

        with caplog.at_level("WARNING"):
            summary = self.build(tmp_path, result)

        assert summary["suggested_action"] == "strong_buy"
        assert not [r for r in caplog.records if "disagrees" in r.message]

    def test_low_coverage_caps_even_on_a_clean_verdict(self, tmp_path):
        thin = make_scores()
        thin["flags"] = ["low_data_coverage"]
        thin["coverage"] = {"composite": 0.55}

        summary = self.build(
            tmp_path, result_with("strong_buy", clean_eligibility(), scores=thin)
        )

        assert summary["suggested_action"] == "watchlist"
        assert any("55%" in r for r in summary["action_constraint"]["constraints"])


class TestConsoleOutputIsGuardedToo:
    """The CLI prints the eligibility gates immediately above the action, so
    it is a decision surface with the same contradiction risk as the report."""

    def printed(self, result) -> str:
        from unittest.mock import patch

        from boundless100x import cli

        captured = []
        with patch.object(cli.console, "print", lambda *a, **k: captured.append(str(a[0]) if a else "")):
            cli._print_llm_summary(result)
        return "\n".join(captured)

    def test_absent_final_action_does_not_fall_back_to_the_raw_model_action(self):
        result = result_with("strong_buy", failed_eligibility())
        assert result.final_action is None

        output = self.printed(result)

        assert "watchlist" in output
        assert "Action: [bold]strong_buy" not in output

    def test_stale_final_action_does_not_reach_the_console(self):
        result = result_with("strong_buy", failed_eligibility())
        result.final_action = {
            "action": "strong_buy", "llm_action": "strong_buy",
            "capped": False, "ceiling": None, "constraints": [],
        }

        output = self.printed(result)

        assert "watchlist" in output

    def test_clean_verdict_still_prints_the_models_action(self):
        result = result_with("strong_buy", clean_eligibility())

        output = self.printed(result)

        assert "strong_buy" in output


class TestRenderedReportIsConsistent:
    def test_markdown_never_prints_the_capped_away_action_as_the_verdict(self, tmp_path):
        result = result_with("strong_buy", failed_eligibility())
        gen = ReportGenerator(output_dir=str(tmp_path))

        rendered = gen._render_markdown(
            result,
            executive_summary=gen._build_executive_summary(result),
        )

        assert "Not a 100x Candidate" in rendered
        assert "| **Action** | WATCHLIST |" in rendered

        # The model's original may appear, but only ever labelled as capped or
        # as its suggestion — never standing alone as the recommendation.
        unlabelled = [
            line for line in rendered.splitlines()
            if "STRONG BUY" in line.upper()
            and "capped" not in line.lower()
            and "model suggested" not in line.lower()
        ]
        assert unlabelled == []

    def test_html_headline_action_is_the_capped_one(self, tmp_path):
        result = result_with("strong_buy", failed_eligibility())
        gen = ReportGenerator(output_dir=str(tmp_path))

        rendered = gen._render_html(
            result, charts={},
            executive_summary=gen._build_executive_summary(result),
        )

        assert 'class="action-badge action-watchlist"' in rendered
        assert 'class="action-badge action-strong_buy"' not in rendered
