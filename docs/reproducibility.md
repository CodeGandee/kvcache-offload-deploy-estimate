# Reproducibility

## Environment

The project uses Python 3.13 and Pixi. Install exactly locked dependencies with:

```bash
pixi install
```

Run all checks:

```bash
pixi run check
pixi run docs
```

Inspect pipeline calibration points:

```bash
pixi run kv-estimate 1 7 14 21 28 40
```

Run the LLMServingSim-compatible ShadowKV extension and its 256-sample sensitivity
study:

```bash
pixi run kv-shadowkv-sim --samples 256
pixi run kv-shadowkv-sim --samples 256 --json
pixi run kv-shadowkv-sim --oracle-recall 1 --oracle-precision 1 --trust-oracle
```

The last command is an optimistic authoritative-oracle upper bound. The published
central case deliberately omits `--trust-oracle`, so it still verifies the current
token with landmarks.

The command imports `_pp_stage_boundaries` from the pinned LLMServingSim checkout,
so submodules must be initialized. It then emits external per-transformer-block
ShadowKV trace events without modifying the upstream simulator.

## External source revisions

Clone with submodules or initialize them after cloning:

```bash
git submodule update --init --recursive
```

For the Hugging Face checkpoint repositories, avoid downloading model weights:

```bash
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive
```

The gitlinks pin exact source/metadata commits. See `extern/tracked/README.md` and
`.gitmodules` for URLs and roles.

## Report integrity

`tests/integration/test_report.py` verifies that the standalone HTML contains both
timing cases, the PP8×TP2 topology, and central result markers. Unit tests verify
pipeline fill, sparse working-set size, per-GPU layout invariance, and TTFT formulas.

The report is standalone except for KaTeX assets loaded from jsDelivr. Its charts and
case data are embedded directly in the HTML so the file can be opened locally.

## Simulation boundary

LLMServingSim provides PP partitioning and serving-trace structure. Its checked-in
profiles do not cover these frontier models on A800, and stock tiered-KV offload does
not express ShadowKV's landmark/reconstruction path. The extension therefore retains
the earlier calibrated non-ShadowKV model-forward floor and explicitly simulates only
the incremental ShadowKV events. Do not describe these values as native LLMServingSim
predictions or measured A800 results.
