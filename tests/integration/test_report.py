"""Integration checks for the published standalone report."""

from pathlib import Path


def test_pp8_report_is_published_with_both_cases() -> None:
    report = Path("docs/cases/a800-pp8-tp2-shadowkv.html")
    contents = report.read_text(encoding="utf-8")
    assert "PP8×TP2" in contents
    assert "fetch-at-decode" in contents
    assert "token-ahead" in contents
    assert "191/91" in contents
    assert "170/80" in contents


def test_report_documents_llmservingsim_and_oracle_precision() -> None:
    report = Path("docs/cases/a800-pp8-tp2-shadowkv.html").read_text(encoding="utf-8")
    assert "LLMServingSim" in report
    assert "oracle precision" in report.lower()
    assert "per-transformer-block" in report
