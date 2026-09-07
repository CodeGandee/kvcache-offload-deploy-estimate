"""Integration checks for the published standalone report."""

from pathlib import Path


def test_pp8_report_is_published_with_both_cases() -> None:
    report = Path("docs/cases/a800-pp8-tp2-shadowkv.html")
    contents = report.read_text(encoding="utf-8")
    assert "PP8×TP2" in contents
    assert "fetch-at-decode" in contents
    assert "token-ahead" in contents
    assert "392/194" in contents
    assert "241/105" in contents
