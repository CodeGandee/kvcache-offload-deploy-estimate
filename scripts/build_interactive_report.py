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
      let caseMode = location.hash.includes('fetch-at-decode') ? 'fetch' : 'ahead';
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
    print(f"embedded {len(payload):,} bytes in {REPORT}")


if __name__ == "__main__":
    main()
