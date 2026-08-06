"""The annual-report budget Pass 1 reads is config-driven, not a literal.

The fetcher caps each extracted section separately (`annual_reports.sections`).
A hard-coded truncation in the orchestrator sits downstream of those caps and
silently overrules them: raising a section cap to get more MD&A into the
prompt would change nothing, and the reason would be invisible from config.
"""

import pytest

from boundless100x.llm_layer.orchestrator import LLMOrchestrator


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")


def orchestrator(config: dict | None = None) -> LLMOrchestrator:
    return LLMOrchestrator(config or {})


def captured_pass1_text(orch, monkeypatch, ar_text: str) -> str:
    """Run Pass 1 against stubs and return the AR text that reached the prompt."""
    monkeypatch.setattr(orch, "_load_template", lambda name: "AR>>>{annual_report_text}")
    seen = {}
    monkeypatch.setattr(
        orch,
        "_call_api",
        lambda model, prompt, label: seen.update(prompt=prompt) or {},
    )

    orch._run_pass1(
        ticker="T",
        company_name="T Ltd",
        sector="Sector",
        market_cap=1000.0,
        metrics={},
        scores={},
        annual_report_text=ar_text,
        sector_context="ctx",
    )
    return seen["prompt"].split("AR>>>", 1)[1]


class TestBudgetIsConfigurable:
    def test_default_preserves_the_previous_literal(self, api_key):
        assert orchestrator().pass1_ar_char_budget == 3000

    def test_config_overrides_the_default(self, api_key):
        orch = orchestrator({"llm": {"pass1_ar_char_budget": 25000}})
        assert orch.pass1_ar_char_budget == 25000

    def test_the_configured_budget_is_what_truncates_the_prompt(
        self, api_key, monkeypatch
    ):
        orch = orchestrator({"llm": {"pass1_ar_char_budget": 120}})
        assert len(captured_pass1_text(orch, monkeypatch, "x" * 5000)) == 120

    def test_a_raised_budget_actually_lets_more_text_through(
        self, api_key, monkeypatch
    ):
        """The point of the change: section caps stop being overruled."""
        orch = orchestrator({"llm": {"pass1_ar_char_budget": 8000}})
        assert len(captured_pass1_text(orch, monkeypatch, "x" * 12000)) == 8000

    def test_text_under_budget_is_not_padded_or_cut(self, api_key, monkeypatch):
        orch = orchestrator({"llm": {"pass1_ar_char_budget": 3000}})
        assert captured_pass1_text(orch, monkeypatch, "short report") == "short report"


class TestNoResidualLiteral:
    def test_the_orchestrator_carries_no_hard_coded_truncation(self):
        """Guards the regression this unit exists to prevent."""
        from pathlib import Path

        import boundless100x.llm_layer.orchestrator as module

        source = Path(module.__file__).read_text()
        assert "[:3000]" not in source
