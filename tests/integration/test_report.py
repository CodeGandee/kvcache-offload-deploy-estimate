"""Integration checks for the published standalone report."""

from pathlib import Path


def test_pp8_report_is_published_with_both_cases() -> None:
    report = Path("docs/cases/a800-pp8-tp2-shadowkv.html")
    contents = report.read_text(encoding="utf-8")
    assert "PP8×TP2" in contents
    assert "fetch-at-decode" in contents
    assert "token-ahead" in contents
    assert "227/120/51" in contents
    assert "211/111/48" in contents


def test_report_documents_llmservingsim_and_oracle_precision() -> None:
    report = Path("docs/cases/a800-pp8-tp2-shadowkv.html").read_text(encoding="utf-8")
    assert "LLMServingSim" in report
    assert "oracle precision" in report.lower()
    assert "per-transformer-block" in report


def test_report_includes_72k_mtp_and_whole_layer_residency() -> None:
    report = Path("docs/cases/a800-pp8-tp2-shadowkv.html").read_text(encoding="utf-8")
    assert '"context":72' in report
    assert "MTP-only decode estimates" in report
    assert "Mean accepted prefix = 1 token" in report
    assert "Mean accepted prefix = 2 tokens" in report
    assert "Whole-layer KV residency scan" in report
    assert "residency-load-control" in report
    assert "GENERATED_REPORT_DATA_START" in report
    assert "one-user autoregressive latency floor" in report
