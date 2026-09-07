"""CLI for the LLMServingSim-compatible ShadowKV extension."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .llmservingsim_shadowkv import OraclePrefetch, build_report, concise_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=256, help="sensitivity samples per point")
    parser.add_argument("--json", action="store_true", help="emit the complete JSON dataset")
    parser.add_argument("--oracle-recall", type=float, default=0.80)
    parser.add_argument("--oracle-precision", type=float, default=0.80)
    parser.add_argument("--reuse", type=float, default=0.60)
    parser.add_argument("--lookahead-tokens", type=int, default=1)
    parser.add_argument(
        "--trust-oracle",
        action="store_true",
        help="skip current-token landmark verification (an optimistic authoritative-oracle bound)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    oracle = OraclePrefetch(
        recall=args.oracle_recall,
        precision=args.oracle_precision,
        lookahead_tokens=args.lookahead_tokens,
        verify_with_landmarks=not args.trust_oracle,
    )
    report = build_report(sensitivity_samples=max(0, args.samples), oracle=oracle, reuse=args.reuse)
    print(json.dumps(report, indent=2) if args.json else concise_table(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
