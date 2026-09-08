"""Refresh the generated simulation dataset embedded in the standalone report."""

from __future__ import annotations

import json
import re
from pathlib import Path

from kvcache_offload_deploy_estimate.genz_llmservingsim import (
    generate_profile_bundles,
)
from kvcache_offload_deploy_estimate.llmservingsim_shadowkv import (
    build_interactive_dataset,
)

REPORT = Path("docs/cases/a800-pp8-tp2-shadowkv.html")
START = "/* GENERATED_REPORT_DATA_START */"
END = "/* GENERATED_REPORT_DATA_END */"
SUMMARY = Path("docs/cases/a800-pp8-tp2.md")
SUMMARY_START = "<!-- GENERATED_CENTRAL_RESULTS_START -->"
SUMMARY_END = "<!-- GENERATED_CENTRAL_RESULTS_END -->"


def _throughput_cell(points: list[dict[str, object]]) -> str:
    midpoint = points[2]
    maximum = points[4]
    return (
        f"{float(midpoint['tps']):,.0f}/{float(midpoint['userTps']):.1f} → "
        f"{float(maximum['tps']):,.0f}/{float(maximum['userTps']):.1f}"
    )


def _central_results_table(dataset: dict[str, object]) -> str:
    labels = {
        "kimi": "Kimi",
        "glm": "GLM-5.3",
        "glm-flash": "GLM Flash",
        "flash": "V4 Flash",
    }
    rows = [
        (
            "| Scenario | Token-ahead 50% → 100% | Fetch-at-decode 50% → 100% | "
            "No ShadowKV 50% → HBM max |"
        ),
        "|---|---:|---:|---:|",
    ]
    series = dataset["series"]
    assert isinstance(series, list)
    for item in series:
        assert isinstance(item, dict)
        base = item["base"]
        assert isinstance(base, dict)
        fp8 = base["fp8"]
        assert isinstance(fp8, dict)
        model_key = str(item["id"]).rsplit("-", 1)[0]
        rows.append(
            f"| {labels[model_key]} {item['context']}K | "
            f"{_throughput_cell(fp8['ahead'])} | "
            f"{_throughput_cell(fp8['fetch'])} | "
            f"{_throughput_cell(fp8['noShadow'])} |"
        )
    return "\n".join(rows)


def main() -> None:
    contents = REPORT.read_text(encoding="utf-8")
    generate_profile_bundles()
    dataset = build_interactive_dataset(sensitivity_samples=256)
    payload = json.dumps(dataset, ensure_ascii=False, separators=(",", ":"))
    pattern = re.compile(rf"{re.escape(START)}.*?{re.escape(END)}", re.DOTALL)
    replacement = f"{START}\n        {payload}\n        {END}"
    updated, count = pattern.subn(replacement, contents)
    if count == 0:
        legacy = re.compile(
            r"      const loads = \[0, 25, 50, 75, 100\];.*?"
            r"(?=      const chartConfig = \{)",
            re.DOTALL,
        )
        bootstrap = f"""      const reportData =
        {replacement};
      const loads = reportData.loads;
      let caseMode = location.hash.includes('no-shadowkv') ? 'noShadow' : location.hash.includes('fetch-at-decode') ? 'fetch' : 'ahead';
      let storageMode = location.hash.includes('fp8-kv') ? 'fp8' : 'bf16';
      const seriesOrder = ['kimi-72','kimi-128','kimi-256','glm-72','glm-128','glm-256','glm-flash-72','glm-flash-128','glm-flash-256','flash-72','flash-128','flash-256'];
      const seriesColors = Object.fromEntries(seriesOrder.map((id,index) => [id,`var(--s${{index + 1}})`]));
      const data = reportData.series.map(series => ({{
        ...series,
        short: `${{series.model === 'Kimi Code 2.7' ? 'Kimi' : series.model === 'GLM-5.3' ? 'GLM' : series.model === 'GLM-5.3-Flash' ? 'GLM Flash' : 'V4 Flash'}} ${{series.context}}K`,
        color: seriesColors[series.id]
      }})).sort((left,right) => seriesOrder.indexOf(left.id) - seriesOrder.indexOf(right.id));
"""
        updated, count = legacy.subn(bootstrap, contents)
    if count != 1:
        raise RuntimeError(f"expected one generated-data block, found {count}")
    REPORT.write_text(updated, encoding="utf-8")

    summary = SUMMARY.read_text(encoding="utf-8")
    summary_pattern = re.compile(
        rf"{re.escape(SUMMARY_START)}.*?{re.escape(SUMMARY_END)}", re.DOTALL
    )
    table = _central_results_table(dataset)
    summary_replacement = f"{SUMMARY_START}\n{table}\n{SUMMARY_END}"
    summary_updated, summary_count = summary_pattern.subn(summary_replacement, summary)
    if summary_count != 1:
        raise RuntimeError(f"expected one generated summary block, found {summary_count}")
    SUMMARY.write_text(summary_updated, encoding="utf-8")
    print(f"embedded {len(payload):,} bytes in {REPORT} and refreshed {SUMMARY}")


if __name__ == "__main__":
    main()
