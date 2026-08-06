"""Recording the checkpoints Pass 2 proposes.

`_parse_json_response` performs no schema validation of any kind, so whatever
the model returns — malformed, truncated, or from an older prompt — arrives
here unchecked. Every shape must degrade rather than raise, and anything the
evaluator could not later find must be demoted to prose rather than stored as
a promise nobody can check.
"""

import pytest

from boundless100x.lifecycle.checkpoints import (
    load_vocabulary,
    record_from_pass2,
    vocabulary_prompt_block,
)


def monitorable(**overrides) -> dict:
    base = {
        "metric_id": "quarterly_opm_pct",
        "comparator": "gte",
        "threshold": 20.0,
        "due_date": "2026-11-15",
    }
    base.update(overrides)
    return base


def pass2(*items) -> dict:
    return {"thesis": "…", "structured_monitorables": list(items)}


class TestHappyPath:
    def test_a_valid_monitorable_becomes_a_checkpoint(self):
        result = record_from_pass2(pass2(monitorable()))

        assert len(result["checkpoints"]) == 1
        assert result["checkpoints"][0]["metric_id"] == "quarterly_opm_pct"
        assert result["demoted"] == []

    def test_recorded_checkpoints_are_marked_as_llm_sourced(self):
        checkpoint = record_from_pass2(pass2(monitorable()))["checkpoints"][0]
        assert checkpoint["source"] == "llm"

    def test_thresholds_are_coerced_to_float(self):
        checkpoint = record_from_pass2(pass2(monitorable(threshold=20)))["checkpoints"][0]
        assert isinstance(checkpoint["threshold"], float)

    def test_several_monitorables_are_all_kept(self):
        result = record_from_pass2(pass2(
            monitorable(),
            monitorable(metric_id="quarterly_revenue_yoy_pct", threshold=15.0),
            monitorable(metric_id="promoter_holding_pct", comparator="gte", threshold=50.0),
        ))
        assert len(result["checkpoints"]) == 3


class TestDemotion:
    def test_a_hallucinated_metric_id_is_demoted_not_stored(self):
        """The failure this unit exists to prevent."""
        result = record_from_pass2(pass2(
            monitorable(),
            monitorable(metric_id="roce_next_year"),
        ))

        assert len(result["checkpoints"]) == 1
        assert len(result["demoted"]) == 1
        assert "not in the checkpoint vocabulary" in result["demoted"][0]["reasons"][0]

    def test_an_annual_metric_id_is_demoted(self):
        """Registry metrics are real ids but cannot come due quarterly."""
        result = record_from_pass2(pass2(monitorable(metric_id="roce_5yr_avg")))
        assert result["checkpoints"] == []

    def test_a_bad_comparator_is_demoted(self):
        assert record_from_pass2(
            pass2(monitorable(comparator="roughly"))
        )["checkpoints"] == []

    def test_a_units_bearing_threshold_is_demoted(self):
        assert record_from_pass2(
            pass2(monitorable(threshold="20%"))
        )["checkpoints"] == []

    def test_a_vague_due_date_is_demoted(self):
        assert record_from_pass2(
            pass2(monitorable(due_date="next quarter"))
        )["checkpoints"] == []

    def test_the_demotion_records_what_was_proposed(self):
        """So a reader can see what the model wanted and why it was refused."""
        demoted = record_from_pass2(pass2(monitorable(metric_id="invented")))["demoted"][0]
        assert demoted["proposed"]["metric_id"] == "invented"
        assert demoted["reasons"]


class TestMalformedResponses:
    """No schema validation upstream — every shape must degrade, never raise."""

    def test_an_absent_key_yields_no_checkpoints(self):
        assert record_from_pass2({"thesis": "…"})["checkpoints"] == []

    def test_a_none_response_yields_no_checkpoints(self):
        assert record_from_pass2(None)["checkpoints"] == []

    def test_a_parse_error_response_yields_no_checkpoints(self):
        assert record_from_pass2(
            {"raw_response": "…", "parse_error": True}
        )["checkpoints"] == []

    def test_a_string_instead_of_a_list_is_demoted_not_raised(self):
        result = record_from_pass2({"structured_monitorables": "watch the margins"})
        assert result["checkpoints"] == []
        assert result["demoted"][0]["reasons"] == ["not a list"]

    def test_a_list_of_strings_is_demoted_not_raised(self):
        result = record_from_pass2(pass2("watch margins", "watch pledge"))
        assert result["checkpoints"] == []
        assert len(result["demoted"]) == 2

    def test_a_partial_monitorable_is_demoted_not_raised(self):
        result = record_from_pass2(pass2({"metric_id": "quarterly_opm_pct"}))
        assert result["checkpoints"] == []

    def test_a_null_entry_is_demoted_not_raised(self):
        assert record_from_pass2(pass2(None))["checkpoints"] == []


class TestPromptWiring:
    def test_the_vocabulary_block_is_injected_into_the_prompt(self):
        from pathlib import Path

        import boundless100x.llm_layer.orchestrator as module

        template = (
            Path(module.__file__).parent / "prompts" / "pass2_synthesis.txt"
        ).read_text()
        assert "{checkpoint_vocabulary}" in template
        assert "structured_monitorables" in template

    def test_the_block_lists_ids_the_recorder_will_accept(self):
        """Prompt and validator must not drift apart."""
        block = vocabulary_prompt_block()
        for metric_id in load_vocabulary():
            assert metric_id in block
            assert record_from_pass2(
                pass2(monitorable(metric_id=metric_id))
            )["checkpoints"], f"{metric_id} advertised but refused"

    def test_prose_monitorables_are_untouched(self):
        """The human-readable list keeps working exactly as before."""
        response = {"key_monitorables": ["Watch quarterly margins"],
                    "structured_monitorables": [monitorable()]}
        record_from_pass2(response)
        assert response["key_monitorables"] == ["Watch quarterly margins"]
