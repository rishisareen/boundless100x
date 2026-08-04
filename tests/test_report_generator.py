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
