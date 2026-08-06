"""The forward-growth extraction pass: content gate, boundary validator, caching.

The failure modes are the unit. Upstream, `_parse_json_response` validates
nothing — a malformed, truncated or simply older response arrives unchecked —
and downstream the extractor's output becomes a forward signal a reader will
act on. So most of what follows is about what the pass *refuses*: text that
only claims to be MD&A, entries whose quoted sentence was never in the
document, and every degenerate response shape a model can produce.
"""

import json

import pytest

from boundless100x import forward_growth_schema as schema
from boundless100x.llm_layer import forward_growth as fg
from tests.conftest import AUDIT_COMMITTEE_TEXT, make_ar_sections

MDNA_TEXT = make_ar_sections()["2025"]["mdna"]["text"]
CHAIRMAN_TEXT = make_ar_sections()["2025"]["chairman"]["text"]

GUIDANCE_SENTENCE = "We expect revenue of Rs 1,500 crore in FY2026."
TAM_SENTENCE = (
    "The addressable market for our products is estimated at Rs 40,000 crore."
)


def submission(years=("2025",), sections=("mdna",)) -> dict:
    text = {"mdna": MDNA_TEXT, "chairman": CHAIRMAN_TEXT}
    return {year: {name: text[name] for name in sections} for year in years}


def guidance_entry(**overrides) -> dict:
    entry = {
        "metric": "revenue",
        "target_value": 1500,
        "target_period": "FY2026",
        "source_sentence": GUIDANCE_SENTENCE,
        "section": "mdna",
    }
    entry.update(overrides)
    return entry


def response(**years) -> dict:
    return {"years": years}


# ── The content gate (KTD9) ────────────────────────────────────────────────


class TestContentGate:
    def test_a_genuine_mdna_slice_passes(self):
        """The gate must not reject the ones detection got right."""
        gated = fg.gate_sections(make_ar_sections(provenance="found"))
        assert gated["2025"]["mdna"] == fg.FOUND

    def test_an_auditor_or_governance_slice_is_downgraded_to_suspect(self):
        """The real ASTRAL residual: a bare heading followed by governance prose."""
        sections = make_ar_sections(
            provenance="found", sections={"mdna": AUDIT_COMMITTEE_TEXT}
        )
        gated = fg.gate_sections(sections)

        assert gated["2025"]["mdna"] == fg.SUSPECT

    def test_suspect_is_a_third_value_not_a_relabelled_fallback(self):
        """The bucket is the point — it makes a wrong `found` tag visible."""
        assert fg.SUSPECT not in (fg.FOUND, fg.FALLBACK)

    def test_fallback_provenance_is_left_alone(self):
        """A slot holding first-N-pages text was never a claim to gate."""
        gated = fg.gate_sections(make_ar_sections(provenance="fallback"))
        assert gated["2025"]["mdna"] == fg.FALLBACK

    def test_mixed_provenance_is_gated_per_section(self):
        sections = make_ar_sections(
            provenance="found",
            sections={"mdna": AUDIT_COMMITTEE_TEXT, "chairman": CHAIRMAN_TEXT},
        )
        gated = fg.gate_sections(sections)

        assert gated["2025"]["mdna"] == fg.SUSPECT
        assert gated["2025"]["chairman"] == fg.FOUND

    def test_a_slice_with_neither_a_heading_nor_a_subject_marker_is_suspect(self):
        sections = make_ar_sections(
            provenance="found",
            sections={"mdna": (
                "The Board has constituted a Stakeholders Relationship "
                "Committee which met twice during the year under review."
            )},
        )
        assert fg.gate_sections(sections)["2025"]["mdna"] == fg.SUSPECT

    def test_a_genuine_slice_mentioning_the_auditors_is_not_thrown_away(self):
        """`auditor's report` names an audit opinion; it does not open one.

        A real chairman's statement in the corpus (541956) notes that the
        statutory auditors raised no qualification. Disqualifying on the words
        rather than on how an audit opinion actually opens would discard it.
        """
        sections = make_ar_sections(
            provenance="found",
            sections={"chairman": (
                "CHAIRMAN'S STATEMENT\nFurthermore, Statutory Auditors have not "
                "given any qualification or remarks in the Auditors' Report and "
                "the Comptroller & Auditor General of India has given 'Nil' "
                "comments for the FY 2024-25."
            )},
        )
        assert fg.gate_sections(sections)["2025"]["chairman"] == fg.FOUND


# Real openings from the fetched corpus, verbatim. The gate was rebuilt against
# these after a first design — two markers drawn from SEBI LODR Schedule V(B)'s
# mandated MD&A contents — rejected 11 of the corpus's 13 `found` MD&A slices,
# of which 11 were genuine. Real MD&A opens with narrative economy and industry
# prose and reaches the mandated sub-headings pages later, well past any
# workable scan window. Keeping the evidence here is what stops that recurring.
GENUINE_MDNA_OPENINGS = {
    "500405": "MANAGEMENT DISCUSSION AND ANALYSIS ECONOMIC REVIEW In 2024-25, "
              "Indian economy is estimated to have grown by 6.5%,",
    "509488": "MANAGEMENT DISCUSSION AND ANALYSIS (i) Industry's structure and "
              "developments A. Graphite and Carbon Segment Graphite Electrodes",
    "522295": "Industry Growth Drivers The FSSAI 2026 Labelling Amendment "
              "mandates detailed safety data, creating a sustained requirement",
    "532321": "Management Discussion and Analysis Global Economy Global economy "
              "continued its growth journey during the year 2023",
    "532922": "Monetary Fund (IMF) projects global growth of 3.3% for 2025, with "
              "an average of around 3.2% expected over the next five years",
    "542830": "Management Discussion and Analysis India's economic review The "
              "Indian economy exhibited strong performance and grew by 6.5%",
    "543265": "1. Economy Overview Introduction India's economy is a dynamic and "
              "rapidly expanding force, marked by resilience and innovation.",
}

WRONG_SECTION_MDNA_OPENINGS = {
    # A Corporate Governance Report under an MD&A heading.
    "500339": "Management Discussion and Analysis Company's Philosophy on Code "
              "of Governance Rain Industries Limited is committed to implement "
              "sound corporate governance practices",
    # A Board's Report pointing at an annexure, then governance prose.
    "531344": "MANAGEMENT DISCUSSION AND ANALYSIS: The detailed Management "
              "Discussion and Analysis forms a part of this report at "
              "Annexure-A. CORPORATE GOVERNANCE & GREEN INITIATIVE:",
}


class TestContentGateAgainstTheRealCorpus:
    @pytest.mark.parametrize("code", sorted(GENUINE_MDNA_OPENINGS))
    def test_a_genuine_mdna_opening_survives(self, code):
        sections = make_ar_sections(
            provenance="found", sections={"mdna": GENUINE_MDNA_OPENINGS[code]}
        )
        assert fg.gate_sections(sections)["2025"]["mdna"] == fg.FOUND

    @pytest.mark.parametrize("code", sorted(WRONG_SECTION_MDNA_OPENINGS))
    def test_a_wrong_section_opening_is_downgraded(self, code):
        sections = make_ar_sections(
            provenance="found", sections={"mdna": WRONG_SECTION_MDNA_OPENINGS[code]}
        )
        assert fg.gate_sections(sections)["2025"]["mdna"] == fg.SUSPECT

    def test_the_gate_keeps_a_large_majority_rather_than_a_handful(self):
        """A1's bar reads both ways: far below it, the gate is over-strict."""
        genuine = sum(
            fg.gate_sections(
                make_ar_sections(provenance="found", sections={"mdna": text})
            )["2025"]["mdna"] == fg.FOUND
            for text in GENUINE_MDNA_OPENINGS.values()
        )
        assert genuine == len(GENUINE_MDNA_OPENINGS)

    def test_the_gate_can_be_disabled_without_a_refetch(self):
        sections = make_ar_sections(
            provenance="found", sections={"mdna": AUDIT_COMMITTEE_TEXT}
        )
        gated = fg.gate_sections(sections, enabled=False)
        assert gated["2025"]["mdna"] == fg.FOUND

    def test_gate_reasons_name_what_was_and_was_not_matched(self):
        sections = make_ar_sections(
            provenance="found", sections={"mdna": AUDIT_COMMITTEE_TEXT}
        )
        _, reasons = fg.gate_sections_with_reasons(sections)
        assert reasons["2025"]["mdna"]


# ── What reaches the prompt ────────────────────────────────────────────────


class TestSubmissionPayload:
    def test_fallback_sections_are_not_submitted(self):
        """Assert on what was sent, not only on what came back."""
        sections = make_ar_sections(provenance="fallback")
        payload = fg.build_submission(sections, fg.gate_sections(sections))
        assert payload == {}

    def test_suspect_sections_are_not_submitted(self):
        sections = make_ar_sections(
            provenance="found",
            sections={"mdna": AUDIT_COMMITTEE_TEXT, "chairman": CHAIRMAN_TEXT},
        )
        payload = fg.build_submission(sections, fg.gate_sections(sections))

        assert set(payload["2025"]) == {"chairman"}

    def test_a_year_with_mixed_provenance_sends_only_its_usable_sections(self):
        sections = make_ar_sections(
            provenance="found", per_section_provenance={"mdna": "fallback"}
        )
        payload = fg.build_submission(sections, fg.gate_sections(sections))

        assert "mdna" not in payload["2025"]
        assert "chairman" in payload["2025"]

    def test_only_sections_a_sub_metric_can_read_are_submitted(self):
        """No tokens on a governance report that answers none of the questions."""
        sections = make_ar_sections(provenance="found")
        payload = fg.build_submission(sections, fg.gate_sections(sections))

        assert "governance" not in payload["2025"]
        assert set(payload["2025"]) <= set(fg.EXTRACTABLE_SECTIONS)

    def test_each_section_is_capped_by_the_char_budget(self):
        sections = make_ar_sections(provenance="found")
        payload = fg.build_submission(sections, fg.gate_sections(sections), char_budget=20)

        assert all(len(text) <= 20 for text in payload["2025"].values())

    def test_multiple_report_years_are_submitted_together(self):
        sections = make_ar_sections(years=["2024", "2025"], provenance="found")
        payload = fg.build_submission(sections, fg.gate_sections(sections))

        assert set(payload) == {"2024", "2025"}


# ── Boundary validation (KTD3) ─────────────────────────────────────────────


class TestWellFormedResponses:
    def test_two_report_years_yield_entries_for_both(self):
        payload = submission(years=("2024", "2025"))
        raw = response(
            **{
                "2024": {"guidance": [guidance_entry()]},
                "2025": {"guidance": [guidance_entry()]},
            }
        )
        result = fg.validate_extraction(raw, payload, fg.gate_sections(
            make_ar_sections(years=["2024", "2025"], provenance="found")
        ))

        assert len(result["years"]["2024"]["guidance"]) == 1
        assert len(result["years"]["2025"]["guidance"]) == 1

    def test_entries_retain_the_verbatim_source_sentence(self):
        result = fg.validate_extraction(
            response(**{"2025": {"guidance": [guidance_entry()]}}),
            submission(),
            fg.gate_sections(make_ar_sections(provenance="found")),
        )
        entry = result["years"]["2025"]["guidance"][0]
        assert entry["source_sentence"] == GUIDANCE_SENTENCE

    def test_every_entry_is_tagged_with_the_section_it_came_from(self):
        """KTD4: a year's sections rarely share one provenance."""
        result = fg.validate_extraction(
            response(**{"2025": {"guidance": [guidance_entry()]}}),
            submission(),
            fg.gate_sections(make_ar_sections(provenance="found")),
        )
        assert result["years"]["2025"]["guidance"][0]["section"] == "mdna"

    def test_the_gated_provenance_of_every_section_is_recorded(self):
        """So a sub-metric can say *why* it is indeterminate, not just that it is."""
        sections = make_ar_sections(
            provenance="found", sections={"mdna": AUDIT_COMMITTEE_TEXT,
                                          "chairman": CHAIRMAN_TEXT}
        )
        result = fg.validate_extraction(
            response(**{"2025": {}}),
            fg.build_submission(sections, fg.gate_sections(sections)),
            fg.gate_sections(sections),
        )
        assert result["years"]["2025"]["sections"]["mdna"] == fg.SUSPECT
        assert result["years"]["2025"]["sections"]["chairman"] == fg.FOUND

    def test_capex_and_tam_entries_validate(self):
        payload = submission(sections=("mdna", "chairman"))
        raw = response(
            **{
                "2025": {
                    "tam": [{
                        "market_size_inr_cr": 40000,
                        "source_sentence": TAM_SENTENCE,
                        "section": "chairman",
                    }],
                }
            }
        )
        result = fg.validate_extraction(
            raw, payload,
            fg.gate_sections(make_ar_sections(provenance="found")),
        )
        assert len(result["years"]["2025"]["tam"]) == 1


class TestGroundingAgainstTheSubmittedText:
    def _validate(self, raw, payload=None):
        return fg.validate_extraction(
            raw, payload or submission(),
            fg.gate_sections(make_ar_sections(provenance="found")),
        )

    def test_a_sentence_that_was_never_in_the_document_is_discarded(self):
        """Shape validation alone cannot tell a fabrication from a reading."""
        raw = response(**{"2025": {"guidance": [guidance_entry(
            source_sentence="We expect revenue of Rs 1,500 crore in FY2026 (invented)."
        )]}})
        result = self._validate(raw)

        assert result["years"]["2025"]["guidance"] == []
        assert any("source_sentence" in d["reason"] for d in result["discarded"])

    def test_a_well_typed_fabrication_is_still_discarded(self):
        """Every field is the right type; the claim is simply not in the text."""
        raw = response(**{"2025": {"guidance": [{
            "metric": "pat",
            "target_value": 999,
            "target_period": "FY2031",
            "source_sentence": "We will earn Rs 999 crore of PAT in FY2031.",
            "section": "mdna",
        }]}})
        assert self._validate(raw)["years"]["2025"]["guidance"] == []

    def test_a_value_absent_from_its_own_quoted_sentence_is_discarded(self):
        raw = response(**{"2025": {"guidance": [guidance_entry(target_value=9999)]}})
        result = self._validate(raw)

        assert result["years"]["2025"]["guidance"] == []
        assert any("target_value" in d["reason"] for d in result["discarded"])

    def test_a_sentence_the_filing_wrapped_across_lines_still_grounds(self):
        """PDF extraction keeps the printed line breaks; a quote does not.

        This is not a hypothetical: one MD&A slice in the corpus carries 320
        hard breaks, and comparing raw bytes rejected 8 of 8 genuinely-present
        statements on the first live run — catching typesetting, not
        fabrication.
        """
        wrapped = (
            "MANAGEMENT DISCUSSION AND ANALYSIS\nECONOMIC REVIEW\n"
            "We expect revenue of Rs 1,500 crore \nin FY2026."
        )
        payload = {"2025": {"mdna": wrapped}}
        raw = response(**{"2025": {"guidance": [guidance_entry()]}})

        result = fg.validate_extraction(
            raw, payload, fg.gate_sections(make_ar_sections(provenance="found"))
        )
        assert len(result["years"]["2025"]["guidance"]) == 1

    def test_a_typographic_apostrophe_does_not_break_grounding(self):
        payload = {"2025": {"mdna": "The Company’s outlook: revenue of "
                                    "Rs 1,500 crore in FY2026."}}
        raw = response(**{"2025": {"guidance": [guidance_entry(
            source_sentence="The Company's outlook: revenue of Rs 1,500 crore in FY2026."
        )]}})
        result = fg.validate_extraction(
            raw, payload, fg.gate_sections(make_ar_sections(provenance="found"))
        )
        assert len(result["years"]["2025"]["guidance"]) == 1

    def test_normalising_whitespace_does_not_weaken_the_guard(self):
        """A sentence that was never in the document still does not appear."""
        payload = {"2025": {"mdna": "We expect revenue of Rs 1,500 crore \nin FY2026."}}
        raw = response(**{"2025": {"guidance": [guidance_entry(
            source_sentence="We promise revenue of Rs 1,500 crore in FY2026."
        )]}})
        result = fg.validate_extraction(
            raw, payload, fg.gate_sections(make_ar_sections(provenance="found"))
        )
        assert result["years"]["2025"]["guidance"] == []

    def test_an_empty_source_sentence_is_discarded(self):
        raw = response(**{"2025": {"guidance": [guidance_entry(source_sentence="   ")]}})
        assert self._validate(raw)["years"]["2025"]["guidance"] == []

    def test_a_figure_in_the_wrong_unit_is_discarded(self):
        """The digits being present proves nothing about what they denominate.

        "capex of USD 500 million" genuinely contains 500, so a bare-numeral
        check grounds an `amount_inr_cr: 500` entry that is wrong by two orders
        of magnitude — and promises-kept would settle it against real INR-crore
        financials.
        """
        payload = {"2025": {"mdna": (
            "MANAGEMENT DISCUSSION AND ANALYSIS\nOUTLOOK\n"
            "We expect capex of USD 500 million by FY2027."
        )}}
        raw = response(**{"2025": {"capex": [{
            "amount_inr_cr": 500,
            "commissioning_year": "FY2027",
            "source_sentence": "We expect capex of USD 500 million by FY2027.",
            "section": "mdna",
        }]}})
        result = fg.validate_extraction(
            raw, payload, fg.gate_sections(make_ar_sections(provenance="found"))
        )

        assert result["years"]["2025"]["capex"] == []
        assert any("INR crore" in d["reason"] for d in result["discarded"])

    @pytest.mark.parametrize("phrasing", [
        "capex of Rs 500 crore by FY2027",
        "capex of 500 crore by FY2027",
        "a Rs 500 cr programme commissioning in FY2027",
    ])
    def test_a_figure_in_the_right_unit_still_grounds(self, phrasing):
        payload = {"2025": {"mdna": f"OUTLOOK\nWe expect {phrasing}."}}
        raw = response(**{"2025": {"capex": [{
            "amount_inr_cr": 500,
            "commissioning_year": "FY2027",
            "source_sentence": f"We expect {phrasing}.",
            "section": "mdna",
        }]}})
        result = fg.validate_extraction(
            raw, payload, fg.gate_sections(make_ar_sections(provenance="found"))
        )
        assert len(result["years"]["2025"]["capex"]) == 1

    def test_a_mixed_unit_sentence_grounds_on_the_right_occurrence(self):
        """One foreign figure in a sentence must not condemn a sound one."""
        sentence = "Revenue grew from USD 100 million to Rs 1,500 crore in FY2026."
        payload = {"2025": {"mdna": f"OUTLOOK\n{sentence}"}}
        raw = response(**{"2025": {"guidance": [guidance_entry(
            source_sentence=sentence
        )]}})
        result = fg.validate_extraction(
            raw, payload, fg.gate_sections(make_ar_sections(provenance="found"))
        )
        assert len(result["years"]["2025"]["guidance"]) == 1

    def test_a_percent_field_is_not_unit_checked(self):
        """A margin carries its unit in the numeral; no scale word applies."""
        sentence = "We target an operating margin of 18.5% in FY2026."
        payload = {"2025": {"mdna": f"OUTLOOK\n{sentence}"}}
        raw = response(**{"2025": {"guidance": [guidance_entry(
            metric="operating_margin_pct", target_value=18.5,
            source_sentence=sentence,
        )]}})
        result = fg.validate_extraction(
            raw, payload, fg.gate_sections(make_ar_sections(provenance="found"))
        )
        assert len(result["years"]["2025"]["guidance"]) == 1

    def test_an_oversized_free_text_field_is_discarded(self):
        """Optional prose is the one part of an entry nothing else constrains."""
        sentence = "A new plant of Rs 500 crore commissions in FY2027."
        payload = {"2025": {"mdna": f"OUTLOOK\n{sentence}"}}
        raw = response(**{"2025": {"capex": [{
            "amount_inr_cr": 500, "commissioning_year": "FY2027",
            "description": "x" * 5000,
            "source_sentence": sentence, "section": "mdna",
        }]}})
        result = fg.validate_extraction(
            raw, payload, fg.gate_sections(make_ar_sections(provenance="found"))
        )
        assert result["years"]["2025"]["capex"] == []

    def test_a_non_string_free_text_field_is_discarded(self):
        sentence = "A new plant of Rs 500 crore commissions in FY2027."
        payload = {"2025": {"mdna": f"OUTLOOK\n{sentence}"}}
        raw = response(**{"2025": {"capex": [{
            "amount_inr_cr": 500, "commissioning_year": "FY2027",
            "description": {"nested": "object"},
            "source_sentence": sentence, "section": "mdna",
        }]}})
        result = fg.validate_extraction(
            raw, payload, fg.gate_sections(make_ar_sections(provenance="found"))
        )
        assert result["years"]["2025"]["capex"] == []

    def test_a_comma_formatted_number_still_grounds(self):
        """The filing writes 1,500; the model returns 1500. Both are the same claim."""
        assert self._validate(
            response(**{"2025": {"guidance": [guidance_entry(target_value=1500.0)]}})
        )["years"]["2025"]["guidance"]

    def test_a_period_absent_from_its_own_quoted_sentence_is_discarded(self):
        raw = response(**{"2025": {"guidance": [guidance_entry(target_period="FY2099")]}})
        assert self._validate(raw)["years"]["2025"]["guidance"] == []

    def test_a_section_that_was_never_submitted_is_discarded(self):
        raw = response(**{"2025": {"guidance": [guidance_entry(section="governance")]}})
        result = self._validate(raw)

        assert result["years"]["2025"]["guidance"] == []
        assert any("section" in d["reason"] for d in result["discarded"])

    def test_a_report_year_that_was_never_submitted_is_discarded(self):
        raw = response(**{"2019": {"guidance": [guidance_entry()]}})
        result = self._validate(raw)

        assert "2019" not in result["years"]
        assert any("2019" in d["reason"] for d in result["discarded"])


class TestClosedVocabulary:
    def _validate(self, raw):
        return fg.validate_extraction(
            raw, submission(), fg.gate_sections(make_ar_sections(provenance="found"))
        )

    def test_a_guidance_metric_outside_the_closed_set_is_discarded(self):
        raw = response(**{"2025": {"guidance": [guidance_entry(metric="vibes")]}})
        result = self._validate(raw)

        assert result["years"]["2025"]["guidance"] == []
        assert any("vibes" in d["reason"] for d in result["discarded"])

    def test_an_entry_kind_outside_the_declared_set_is_dropped_with_a_reason(self):
        raw = response(**{"2025": {
            "dividends": [{"anything": 1}],
            "guidance": [guidance_entry()],
        }})
        result = self._validate(raw)

        assert "dividends" not in result["years"]["2025"]
        assert len(result["years"]["2025"]["guidance"]) == 1  # sibling survives
        assert any("dividends" in d["reason"] for d in result["discarded"])

    def test_an_unknown_field_inside_an_entry_is_stripped_and_the_entry_survives(self):
        raw = response(**{"2025": {"guidance": [guidance_entry(confidence="high")]}})
        result = self._validate(raw)

        kept = result["years"]["2025"]["guidance"]
        assert len(kept) == 1
        assert "confidence" not in kept[0]
        assert any("confidence" in d["reason"] for d in result["discarded"])

    def test_the_prompt_carries_the_closed_field_list(self):
        """Phase 1's lesson: asked for an id without a menu, a model invents one."""
        block = fg.vocabulary_prompt_block()
        for metric_id in schema.GUIDANCE_METRICS:
            assert metric_id in block
        for kind in schema.ENTRY_KINDS:
            assert kind in block


class TestDegenerateResponses:
    def _validate(self, raw):
        return fg.validate_extraction(
            raw, submission(), fg.gate_sections(make_ar_sections(provenance="found"))
        )

    def test_a_string_where_a_list_belongs_degrades_to_empty(self):
        result = self._validate(response(**{"2025": {"guidance": "lots of guidance"}}))
        assert result["years"]["2025"]["guidance"] == []

    def test_a_partial_object_is_discarded_rather_than_half_stored(self):
        raw = response(**{"2025": {"guidance": [{"metric": "revenue"}]}})
        assert self._validate(raw)["years"]["2025"]["guidance"] == []

    def test_a_null_entry_does_not_raise_and_keeps_its_valid_sibling(self):
        result = self._validate(
            response(**{"2025": {"guidance": [None, guidance_entry()]}})
        )
        assert len(result["years"]["2025"]["guidance"]) == 1

    def test_an_absent_key_degrades_to_empty(self):
        result = self._validate(response(**{"2025": {}}))
        assert result["years"]["2025"]["guidance"] == []

    def test_a_parse_error_response_yields_empty_output(self):
        """The exact shape `_parse_json_response` returns when it gives up."""
        result = self._validate({"raw_response": "sorry, no JSON", "parse_error": True})
        assert all(not year.get("guidance") for year in result["years"].values())

    def test_a_non_dict_response_yields_empty_output(self):
        for raw in (None, [], "text", 7):
            assert fg.validate_extraction(
                raw, submission(),
                fg.gate_sections(make_ar_sections(provenance="found")),
            )["years"]

    def test_years_not_a_mapping_degrades_to_empty(self):
        assert self._validate({"years": ["2025"]})["years"]["2025"]["guidance"] == []

    def test_a_boolean_is_not_a_number(self):
        raw = response(**{"2025": {"guidance": [guidance_entry(target_value=True)]}})
        assert self._validate(raw)["years"]["2025"]["guidance"] == []


# ── Sidecar caching and versioning ─────────────────────────────────────────


class TestSidecar:
    def payload(self):
        return submission()

    def test_a_valid_sidecar_round_trips(self, tmp_path):
        path = tmp_path / "fg.json"
        stored = {"2025": {"sections": {"mdna": fg.FOUND}, "guidance": []}}
        fg.write_sidecar(path, stored, self.payload(), model="m")

        assert fg.read_sidecar(path, self.payload(), model="m") == stored

    def test_a_missing_sidecar_reads_as_none(self, tmp_path):
        assert fg.read_sidecar(tmp_path / "absent.json", self.payload(), model="m") is None

    def test_a_changed_model_invalidates_the_sidecar(self, tmp_path):
        path = tmp_path / "fg.json"
        fg.write_sidecar(path, {"2025": {}}, self.payload(), model="sonnet")

        assert fg.read_sidecar(path, self.payload(), model="opus") is None

    def test_changed_source_text_invalidates_the_sidecar(self, tmp_path):
        path = tmp_path / "fg.json"
        fg.write_sidecar(path, {"2025": {}}, self.payload(), model="m")

        changed = {"2025": {"mdna": MDNA_TEXT + " and more"}}
        assert fg.read_sidecar(path, changed, model="m") is None

    def test_a_changed_field_schema_invalidates_the_sidecar(self, tmp_path, monkeypatch):
        path = tmp_path / "fg.json"
        fg.write_sidecar(path, {"2025": {}}, self.payload(), model="m")

        monkeypatch.setattr(schema, "SCHEMA_VERSION", schema.SCHEMA_VERSION + 1)
        assert fg.read_sidecar(path, self.payload(), model="m") is None

    def test_a_changed_prompt_invalidates_the_sidecar(self, tmp_path, monkeypatch):
        path = tmp_path / "fg.json"
        fg.write_sidecar(path, {"2025": {}}, self.payload(), model="m")

        monkeypatch.setattr(fg, "prompt_digest", lambda: "a-different-prompt")
        assert fg.read_sidecar(path, self.payload(), model="m") is None

    def test_an_unreadable_sidecar_reads_as_none_rather_than_raising(self, tmp_path):
        path = tmp_path / "fg.json"
        path.write_text("{not json")

        assert fg.read_sidecar(path, self.payload(), model="m") is None

    def test_the_version_block_is_written_alongside_the_years(self, tmp_path):
        path = tmp_path / "fg.json"
        fg.write_sidecar(path, {"2025": {}}, self.payload(), model="m")

        stored = json.loads(path.read_text())
        assert set(stored) == {"version", "years", "discarded"}
        assert stored["version"]["model"] == "m"

    def test_discard_reasons_survive_the_run(self, tmp_path):
        """A log line does not outlive the process; the reason for an empty
        forward-growth result has to be inspectable afterwards."""
        path = tmp_path / "fg.json"
        reasons = [{"where": "2025.guidance", "reason": "not in the document"}]
        fg.write_sidecar(path, {"2025": {}}, self.payload(), model="m",
                         discarded=reasons)

        assert fg.read_sidecar_discards(path) == reasons

    def test_discards_do_not_invalidate_an_otherwise_current_cache(self, tmp_path):
        path = tmp_path / "fg.json"
        fg.write_sidecar(path, {"2025": {}}, self.payload(), model="m",
                         discarded=[{"where": "x", "reason": "y"}])

        assert fg.read_sidecar(path, self.payload(), model="m") == {"2025": {}}

    def test_missing_or_unreadable_discards_read_as_empty(self, tmp_path):
        assert fg.read_sidecar_discards(tmp_path / "absent.json") == []
        broken = tmp_path / "broken.json"
        broken.write_text("{not json")
        assert fg.read_sidecar_discards(broken) == []


# ── The real orchestrator method ───────────────────────────────────────────


class TestPromptAssembly:
    """Exercises the real template and the real `.format()` call.

    Every Stage 1.5 test below substitutes a stub for the whole orchestrator,
    so without this the actual prompt file, its placeholders, and the
    keyword arguments passed to it are never executed by the suite — a renamed
    placeholder or a typo would surface only in production.
    """

    @pytest.fixture
    def orchestrator(self, monkeypatch):
        from boundless100x.llm_layer.orchestrator import LLMOrchestrator

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
        return LLMOrchestrator({})

    def captured(self, orchestrator, monkeypatch, payload):
        seen = {}
        monkeypatch.setattr(
            orchestrator, "_call_api",
            lambda model, prompt, label: seen.update(prompt=prompt, model=model) or {},
        )
        orchestrator.run_forward_growth_extraction("ASTRAL", "Astral Ltd", payload)
        return seen

    def test_the_real_template_renders_with_the_real_arguments(
        self, orchestrator, monkeypatch
    ):
        seen = self.captured(orchestrator, monkeypatch, submission())

        assert "ASTRAL" in seen["prompt"]
        assert "Astral Ltd" in seen["prompt"]
        assert "MANAGEMENT DISCUSSION AND ANALYSIS" in seen["prompt"]
        assert "=== REPORT YEAR 2025 ===" in seen["prompt"]
        assert "--- SECTION: mdna ---" in seen["prompt"]

    def test_the_closed_vocabulary_reaches_the_prompt(self, orchestrator, monkeypatch):
        seen = self.captured(orchestrator, monkeypatch, submission())

        for metric_id in schema.GUIDANCE_METRICS:
            assert metric_id in seen["prompt"]

    def test_the_json_schema_block_renders_with_single_braces(
        self, orchestrator, monkeypatch
    ):
        """Doubled braces in the template must survive `.format()` as literals."""
        seen = self.captured(orchestrator, monkeypatch, submission())

        assert '"years": {' in seen["prompt"]
        assert "{{" not in seen["prompt"].split("ANNUAL REPORT SECTIONS")[-1]

    def test_braces_in_filing_text_do_not_break_rendering(
        self, orchestrator, monkeypatch
    ):
        """Filing text is a substituted value, so `.format()` must not reparse it."""
        hostile = {"2025": {"mdna": "Outlook {not_a_placeholder} and {{braces}}."}}
        seen = self.captured(orchestrator, monkeypatch, hostile)

        assert "{not_a_placeholder}" in seen["prompt"]

    def test_the_configured_extraction_model_is_used(self, orchestrator, monkeypatch):
        seen = self.captured(orchestrator, monkeypatch, submission())
        assert seen["model"] == orchestrator.forward_growth_model

    def test_deep_mode_moves_the_extraction_model_and_back(self, orchestrator):
        from boundless100x.llm_layer.orchestrator import DEEP_MODEL

        configured = orchestrator.forward_growth_model
        orchestrator.use_deep_models()
        assert orchestrator.forward_growth_model == DEEP_MODEL

        orchestrator.use_configured_models()
        assert orchestrator.forward_growth_model == configured


# ── Stage 1.5 in the service ───────────────────────────────────────────────


class RecordingLLM:
    """Stands in for the orchestrator, counting calls so cost is assertable."""

    def __init__(self, response=None):
        self.forward_growth_model = "stub-model"
        self.forward_growth_char_budget = fg.DEFAULT_CHAR_BUDGET
        self.calls = []
        self._response = response if response is not None else {"years": {}}

    @property
    def _configured_model(self):
        """Whatever the test set, so a reset restores that rather than a literal."""
        return getattr(self, "_baseline_model", None) or self.forward_growth_model

    def run_forward_growth_extraction(self, ticker, company_name, submission):
        self.calls.append({"ticker": ticker, "submission": submission})
        return self._response

    def use_deep_models(self):
        self._baseline_model = self._configured_model
        self.forward_growth_model = "stub-deep-model"

    def use_configured_models(self):
        self.forward_growth_model = self._configured_model

    def run_analysis(self, **kwargs):
        # Stage 4 is not what these tests are about; keep it quiet.
        return {"pass1": {"skipped": True}, "pass2": {"skipped": True}, "usage": {}}


def service_for(monkeypatch, tmp_path, data, llm=None):
    from tests.test_source_status import service_with_stub_suite

    data["source_status"] = {"financials": "ok", "price": "ok"}
    svc = service_with_stub_suite(monkeypatch, data)
    svc.history_path = tmp_path / "h.jsonl"
    monkeypatch.setattr(svc.suite, "raw_data_dir", str(tmp_path / "raw"))
    svc._llm = llm
    return svc


def analysable(**section_kwargs):
    from tests.conftest import make_data

    data = make_data(annual_report_sections=section_kwargs)
    data["metadata"]["bse_code"] = "500001"
    return data


class TestStageOnePointFive:
    def test_a_found_report_produces_populated_forward_growth(self, monkeypatch, tmp_path):
        llm = RecordingLLM(response={"years": {"2025": {"guidance": [{
            "metric": "revenue", "target_value": 1500, "target_period": "FY2026",
            "source_sentence": GUIDANCE_SENTENCE, "section": "mdna",
        }]}}})
        svc = service_for(monkeypatch, tmp_path, analysable(provenance="found"), llm)

        result = svc.analyze("ASTRAL", use_llm=True)

        assert len(llm.calls) == 1
        assert result.data["forward_growth"]["2025"]["guidance"]

    def test_a_ticker_whose_sections_all_fell_back_makes_no_call(self, monkeypatch, tmp_path):
        llm = RecordingLLM()
        svc = service_for(monkeypatch, tmp_path, analysable(provenance="fallback"), llm)

        result = svc.analyze("ASTRAL", use_llm=True)

        assert llm.calls == []
        assert result.data["forward_growth"] == {}

    def test_a_suspect_only_ticker_makes_no_call(self, monkeypatch, tmp_path):
        """The content gate saves the tokens as well as the wrong answer."""
        llm = RecordingLLM()
        data = analysable(
            provenance="found",
            sections={"mdna": AUDIT_COMMITTEE_TEXT},
        )
        svc = service_for(monkeypatch, tmp_path, data, llm)

        result = svc.analyze("ASTRAL", use_llm=True)

        assert llm.calls == []
        assert result.data["forward_growth"] == {}

    def test_only_gated_sections_reach_the_prompt_payload(self, monkeypatch, tmp_path):
        llm = RecordingLLM()
        data = analysable(
            provenance="found", per_section_provenance={"mdna": "fallback"}
        )
        svc = service_for(monkeypatch, tmp_path, data, llm)

        svc.analyze("ASTRAL", use_llm=True)

        submitted = llm.calls[0]["submission"]["2025"]
        assert "mdna" not in submitted
        assert "chairman" in submitted

    def test_a_second_run_over_an_unchanged_report_makes_no_call(self, monkeypatch, tmp_path):
        llm = RecordingLLM()
        svc = service_for(monkeypatch, tmp_path, analysable(provenance="found"), llm)

        svc.analyze("ASTRAL", use_llm=True)
        svc.analyze("ASTRAL", use_llm=True)

        assert len(llm.calls) == 1

    def test_a_no_llm_run_with_a_valid_sidecar_still_hydrates(self, monkeypatch, tmp_path):
        """The `watchlist advance` path: hydration is not gated on use_llm."""
        llm = RecordingLLM(response={"years": {"2025": {"guidance": [{
            "metric": "revenue", "target_value": 1500, "target_period": "FY2026",
            "source_sentence": GUIDANCE_SENTENCE, "section": "mdna",
        }]}}})
        svc = service_for(monkeypatch, tmp_path, analysable(provenance="found"), llm)
        svc.analyze("ASTRAL", use_llm=True)

        result = svc.analyze("ASTRAL", use_llm=False)

        assert len(llm.calls) == 1
        assert result.data["forward_growth"]["2025"]["guidance"]

    def test_a_no_llm_run_with_no_sidecar_leaves_the_key_absent(self, monkeypatch, tmp_path):
        llm = RecordingLLM()
        svc = service_for(monkeypatch, tmp_path, analysable(provenance="found"), llm)

        result = svc.analyze("ASTRAL", use_llm=False)

        assert llm.calls == []
        assert "forward_growth" not in result.data

    def test_a_run_with_no_llm_configured_at_all_still_hydrates(self, monkeypatch, tmp_path):
        """No API key must not mean a paid-for extraction becomes unreadable."""
        llm = RecordingLLM(response={"years": {"2025": {}}})
        llm.forward_growth_model = "claude-sonnet-4-6"  # the configured default
        svc = service_for(monkeypatch, tmp_path, analysable(provenance="found"), llm)
        svc.analyze("ASTRAL", use_llm=True)

        keyless = service_for(monkeypatch, tmp_path, analysable(provenance="found"), None)
        result = keyless.analyze("ASTRAL", use_llm=True)

        # Hydrated, not re-extracted: the recorded provenance survived the trip
        # through the cache even though this run could not have produced it.
        assert result.data["forward_growth"]["2025"]["sections"]["mdna"] == fg.FOUND

    def test_deep_mode_reaches_the_extraction_call(self, monkeypatch, tmp_path):
        llm = RecordingLLM()
        svc = service_for(monkeypatch, tmp_path, analysable(provenance="found"), llm)

        svc.analyze("ASTRAL", use_llm=True, deep=True)

        assert llm.forward_growth_model == "stub-deep-model"

    def test_a_failed_api_call_is_not_cached_as_a_genuine_empty_result(
        self, monkeypatch, tmp_path
    ):
        """An outage is not a finding.

        `_call_api` turns any network or rate-limit failure into
        `{"error": ...}` rather than raising. Cached, that would be served as a
        confirmed-empty extraction on every later run — permanently, since
        nothing re-extracts until the text, schema, prompt or model changes.
        """
        failing = RecordingLLM(response={"error": "rate limited", "pass": "forward_growth"})
        svc = service_for(monkeypatch, tmp_path, analysable(provenance="found"), failing)

        result = svc.analyze("ASTRAL", use_llm=True)

        assert len(failing.calls) == 1
        assert "forward_growth" not in result.data
        assert any("Forward-growth" in e for e in result.errors)
        # And the next run must actually retry rather than read an outage back.
        working = RecordingLLM(response={"years": {"2025": {"guidance": [{
            "metric": "revenue", "target_value": 1500, "target_period": "FY2026",
            "source_sentence": GUIDANCE_SENTENCE, "section": "mdna",
        }]}}})
        retried = service_for(
            monkeypatch, tmp_path, analysable(provenance="found"), working
        )
        assert retried.analyze("ASTRAL", use_llm=True).data["forward_growth"]

    def test_a_call_failure_is_distinguishable_from_a_genuine_empty(self):
        payload = submission()
        gated = fg.gate_sections(make_ar_sections(provenance="found"))

        outage = fg.validate_extraction({"error": "boom"}, payload, gated)
        empty = fg.validate_extraction(response(**{"2025": {}}), payload, gated)

        assert outage["call_failed"] is True
        assert empty["call_failed"] is False

    def test_deep_mode_does_not_leak_into_a_later_shallow_run(
        self, monkeypatch, tmp_path
    ):
        """The service is documented as reusable; `deep=False` must mean it."""
        llm = RecordingLLM()
        svc = service_for(monkeypatch, tmp_path, analysable(provenance="found"), llm)

        svc.analyze("ASTRAL", use_llm=True, deep=True)
        assert llm.forward_growth_model == "stub-deep-model"

        svc.analyze("ASTRAL", use_llm=True, deep=False)
        assert llm.forward_growth_model == "stub-model"

    def test_an_extraction_failure_does_not_cost_the_caller_the_analysis(
        self, monkeypatch, tmp_path
    ):
        class ExplodingLLM(RecordingLLM):
            def run_forward_growth_extraction(self, *a, **kw):
                raise RuntimeError("API down")

        svc = service_for(
            monkeypatch, tmp_path, analysable(provenance="found"), ExplodingLLM()
        )
        result = svc.analyze("ASTRAL", use_llm=True)

        assert result.scores["composite"] is not None
        assert any("Forward-growth" in e for e in result.errors)

    def test_a_ticker_with_no_annual_reports_leaves_the_key_absent(
        self, monkeypatch, tmp_path
    ):
        from tests.conftest import make_data

        data = make_data()
        data["annual_report_sections"] = {}
        svc = service_for(monkeypatch, tmp_path, data, RecordingLLM())

        result = svc.analyze("ASTRAL", use_llm=True)

        assert "forward_growth" not in result.data
