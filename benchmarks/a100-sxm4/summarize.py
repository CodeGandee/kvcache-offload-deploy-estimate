"""Summarize the repeated A100 primitive measurements used by the estimator."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def median(values: list[float]) -> float:
    return statistics.median(values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    payload: dict[str, Any] = json.loads(args.result.read_text(encoding="utf-8"))
    measurements = payload["measurements"]

    hbm = median(
        [row["sm_add_out"]["gbps_read_plus_write"]["median"] for row in measurements["hbm"]]
    )
    h2d_single = median(
        [
            row["payload_gbps"]["median"]
            for rows in measurements["h2d_single"].values()
            for row in rows
            if row["size_mib"] in (16, 64, 256)
        ]
    )
    h2d_pair = median(
        [
            row["aggregate_payload_gbps"]["median"]
            for row in measurements["h2d_dual"]
            if row["size_mib_per_gpu"] in (16, 64, 256)
        ]
    )
    p2p = measurements["p2p"]["unidirectional"]
    p2p_small = next(row for row in p2p if row["size_mib"] == 0.004)["latency_ms"]["median"]
    p2p_large = next(row for row in p2p if row["size_mib"] == 256)["aggregate_payload_gbps"][
        "median"
    ]
    fused = measurements["fp8_dequant_cuda_fused"]["devices"]
    dequant = median([row["gvalues_per_second"] for row in fused])
    unfused = median(
        [
            row["cast_then_block_scale"]["gvalues_per_second"]["median"]
            for row in measurements["fp8_dequant"]
        ]
    )
    mbu = [row["mbu_fraction"] for device_rows in measurements["bf16_gemm"] for row in device_rows]

    rows = (
        ("SM streaming HBM", f"{hbm:,.0f} GB/s"),
        ("H2D, one GPU (16–256 MiB)", f"{h2d_single:.1f} GB/s"),
        ("H2D, concurrent TP2 pair", f"{h2d_pair:.1f} GB/s aggregate"),
        ("P2P, 4 KiB", f"{p2p_small:.3f} ms"),
        ("P2P, 256 MiB", f"{p2p_large:.0f} GB/s"),
        ("Fused E4M3 block128 → BF16", f"{dequant:.0f} Gvalue/s"),
        ("Unfused cast + block scale", f"{unfused:.0f} Gvalue/s"),
        (
            "Representative BF16 GEMM MBU",
            f"{median(mbu) * 100:.1f}% median ({min(mbu) * 100:.1f}–{max(mbu) * 100:.1f}%)",
        ),
    )
    print("| Primitive | Measured center |")
    print("|---|---:|")
    for name, value in rows:
        print(f"| {name} | {value} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
