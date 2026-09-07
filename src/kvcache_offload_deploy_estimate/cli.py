"""Command-line entry point for inspecting the current topology model."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .model import PP2_TP8, PP8_TP2, pipeline_efficiency, pp8_to_pp2_throughput_ratio


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Inspect PP8×TP2 pipeline fill and its ratio to PP2×TP8."
    )
    parser.add_argument(
        "microbatches",
        nargs="*",
        type=int,
        default=[1, 8, 16, 32],
        help="independently schedulable decode microbatch counts",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Print topology and pipeline calibration points as JSON."""

    args = build_parser().parse_args(argv)
    payload = {
        "topologies": {
            "pp8_tp2": {
                "gpus": PP8_TP2.total_gpus,
                "layer_share_per_gpu": PP8_TP2.average_layer_share_per_gpu,
            },
            "pp2_tp8": {
                "gpus": PP2_TP8.total_gpus,
                "layer_share_per_gpu": PP2_TP8.average_layer_share_per_gpu,
            },
        },
        "points": [
            {
                "microbatches": microbatches,
                "pp8_efficiency": pipeline_efficiency(8, microbatches),
                "pp8_to_pp2_ratio": pp8_to_pp2_throughput_ratio(microbatches),
            }
            for microbatches in args.microbatches
        ],
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
