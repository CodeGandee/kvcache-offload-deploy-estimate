"""Integration checks for the published standalone report."""

from pathlib import Path


def test_pp8_report_is_published_with_all_serving_cases() -> None:
    report = Path("docs/cases/a800-pp8-tp2-shadowkv.html")
    contents = report.read_text(encoding="utf-8")
    assert "PP8×TP2" in contents
    assert "fetch-at-decode" in contents
    assert "token-ahead" in contents
    assert "no-shadowkv" in contents
    assert "Case C · no ShadowKV" in contents
    assert "the first request beyond" in contents
    assert "GLM-5.3-Flash" in contents


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
    assert "GenZ-to-LLMServingSim" in report
    assert "Measured A100 hardware proxy" in report
    assert "25.4 GB/s aggregate" in report
    assert '"modelCoreFloorReplaced":true' in report
    assert '"schemaVersion":7' in report
    assert '"microbatch":' in report
    assert '"noShadow":' in report
    assert '"shadowKVAdmissionIsMemoryOnly":true' in report
    assert '"systemRAMCapacity":"unbounded"' in report
    assert '"tensorParallelCachePlacement":"pure-tp-replicated-mla"' in report
    assert '"decodeContextParallelDegree":1' in report
    assert '"shadowKVAttentionIsSeparateTable13Term":true' in report
    assert '"shadowKVFP8ConversionChargedOnceAtAttention":true' in report
    assert '"shadowKVUnshadowedNativeAttentionIsCharged":true' in report
    assert "HBM OOM boundary" in report
    assert "busiest PP8 stage contains eight" in report

    summary = Path("docs/cases/a800-pp8-tp2.md").read_text(encoding="utf-8")
    assert "GENERATED_CENTRAL_RESULTS_START" in summary
    assert "Kimi 128K | 270/4.2 → 313/2.5" in summary
    assert "V4 Flash 256K | 256/14.2 → 337/9.3" in summary


def test_all_static_and_dynamic_math_sections_are_rendered() -> None:
    report = Path("docs/cases/a800-pp8-tp2-shadowkv.html").read_text(encoding="utf-8")
    assert "document.querySelectorAll('.math-body').forEach" in report
    assert "Measured A100 hardware proxy" in report
    assert r"t_{\mathrm{H2D,stage}}=\max" in report
