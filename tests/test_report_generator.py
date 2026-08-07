"""Report generation regression tests.

The peer-comparison removal left `_build_sector_context` reading a deleted
`AnalysisResult.comparison` field, which raised AttributeError for every
ticker. These tests pin the generator against that class of regression.
"""

from pathlib import Path

import pytest

from boundless100x.output.report_generator import ReportGenerator
from tests.conftest import make_result

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "boundless100x"


@pytest.fixture
def generator(tmp_path):
    return ReportGenerator(output_dir=str(tmp_path))


def test_generate_completes_for_current_analysis_result(generator, analysis_result):
    report_dir = generator.generate(analysis_result, formats=["html", "md", "json"])

    assert report_dir.exists()
    assert (report_dir / f"{analysis_result.ticker}_dashboard.html").exists()
    assert (report_dir / "raw_metrics.json").exists()


def test_generate_html_only_completes(generator, analysis_result):
    report_dir = generator.generate(analysis_result, formats=["html"])
    html = (report_dir / f"{analysis_result.ticker}_dashboard.html").read_text()

    assert "Test Co" in html


def test_generate_without_llm_analysis_completes(generator):
    """--no-llm runs leave llm_analysis as None; the report must still render."""
    result = make_result()
    assert result.llm_analysis is None

    report_dir = generator.generate(result, formats=["html", "md"])

    assert (report_dir / f"{result.ticker}_dashboard.html").exists()


def test_no_references_to_removed_comparison_field():
    """`AnalysisResult.comparison` was deleted with the peer feature."""
    offenders = [
        path.relative_to(PACKAGE_ROOT)
        for path in PACKAGE_ROOT.rglob("*.py")
        if "result.comparison" in path.read_text()
    ]

    assert offenders == []


def test_no_references_to_removed_peers_field():
    offenders = [
        path.relative_to(PACKAGE_ROOT)
        for path in PACKAGE_ROOT.rglob("*.py")
        if "result.peers" in path.read_text()
    ]

    assert offenders == []


class TestTheReportLayerSplit:
    """Where the report layer's three modules draw their lines.

    `report_generator.py` had grown past two thousand lines, and two of the
    things inside it were not report *sections*: four hundred lines of display
    vocabulary that grows every time a metric or flag is added, and four
    hundred lines of Plotly trace assembly. Both moved out. What stayed is the
    class that decides what a report contains.

    The split is pinned rather than trusted because both halves have a way of
    creeping back — a new label is easiest to type next to the section that
    renders it, and a new chart next to the builder that calls it.
    """

    def source(self, name: str) -> str:
        import boundless100x.output as package

        return (Path(package.__file__).parent / name).read_text()

    def test_the_vocabulary_module_is_data_only(self):
        """No logic, no rendering, no I/O — only what a reader sees. A function
        here would be a report section that had drifted into the dictionary of
        names for report sections."""
        import ast

        tree = ast.parse(self.source("report_vocabulary.py"))
        definitions = [
            node.name for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]

        assert definitions == []

    def test_no_plotly_import_survives_in_the_generator(self):
        """The charts moved wholesale. A `go` or `pio` left behind would mean a
        figure still being assembled among the sections."""
        source = self.source("report_generator.py")

        assert "plotly" not in source

    def test_every_chart_builder_is_free_of_self(self):
        """Why the extraction was mechanical rather than a judgement call: they
        were module functions wearing method clothing, and the class was their
        namespace rather than their owner. A builder that starts reading
        generator state would have to come back."""
        import ast

        tree = ast.parse(self.source("report_charts.py"))
        signatures = [
            node.args.args[0].arg
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.args.args
        ]

        assert "self" not in signatures

    def test_the_names_the_suite_imports_still_resolve_from_the_generator(self):
        """Re-exported deliberately. These are imported from
        `report_generator` by the tests and by KTD6's registry-derived flag
        check; the split must not move them out from under anybody."""
        from boundless100x.output import report_generator, report_vocabulary

        for name in ("FLAG_LABELS", "FLAG_ELEMENT_MAP", "FORWARD_SIGNALS",
                     "FORWARD_SIGNALS_DISCLAIMER", "FORWARD_SIGNALS_ELEMENT",
                     "FRICTION_UNAVAILABLE_LABEL", "LANE_VERDICT_LABELS",
                     "METRIC_DISPLAY_NAMES", "ELEMENT_CONFIG"):
            assert getattr(report_generator, name) is getattr(
                report_vocabulary, name
            ), name

    def test_the_generator_still_renders_every_format(self, tmp_path):
        """The whole point, asserted past the module boundaries: a real render
        still produces all three artefacts with real figures in the HTML."""
        from boundless100x.output.report_generator import ReportGenerator
        from tests.conftest import make_result

        report_dir = Path(
            ReportGenerator(output_dir=str(tmp_path)).generate(
                make_result("TEST"), formats=["html", "md", "json"]
            )
        )
        html = (report_dir / "TEST_dashboard.html").read_text()

        assert (report_dir / "TEST_report.md").exists()
        assert (report_dir / "scores.json").exists()
        assert html.count("Plotly.newPlot") >= 3
