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

## A100 hardware-proxy calibration

The synthetic calibration needs no model checkpoints and changes no system Python or
CUDA packages. Copy `benchmarks/a100-sxm4/` to a local data disk on the target host,
then run the locked project while exposing exactly the selected pair:

```bash
cd /path/on/local-data-disk/kvcache-offload-calibration-a100
PIXI_CACHE_DIR=/path/on/local-data-disk/.cache/pixi \
  CUDA_VISIBLE_DEVICES=2,3 \
  numactl --cpunodebind=0 --membind=0 \
  ~/.pixi/bin/pixi run calibrate
```

The checked-in [benchmark directory](https://github.com/CodeGandee/kvcache-offload-deploy-estimate/tree/main/benchmarks/a100-sxm4),
custom CUDA FP8 conversion kernel, exact Pixi lock, and
[public-safe raw result](https://github.com/CodeGandee/kvcache-offload-deploy-estimate/blob/main/benchmarks/a100-sxm4/results/a100-sxm4-80gb-gpu2-3.json)
make the calibration reproducible. The JSON records hardware class and software
versions but omits the private hostname and GPU UUIDs.

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
pixi run report-assets
```

The authoritative-oracle command is an optimistic upper bound. `report-assets`
regenerates and embeds the base, 72K, MTP, and whole-layer residency datasets in the
standalone report. The published
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
timing cases, the PP8×TP2 topology, 72K data, MTP cases, whole-layer residency controls,
and central result markers. Unit tests verify pipeline fill, sparse working-set size,
MTP acceptance accounting, whole-layer ratio rounding, resident-HBM payload removal,
per-GPU layout invariance, the PP8 one-user latency floor across every GLM residency
point, and TTFT formulas.

The report is standalone except for KaTeX assets loaded from jsDelivr. Its charts and
case data are embedded directly in the HTML so the file can be opened locally.

## Simulation boundary

LLMServingSim provides PP partitioning and serving-trace structure. Its checked-in
profiles do not cover these frontier models on A800, and stock tiered-KV offload does
not express ShadowKV's landmark/reconstruction path. The extension therefore retains
the earlier calibrated non-ShadowKV model-forward floor and explicitly simulates only
the incremental ShadowKV events. A100 measurements replace the PCIe, TP2-pair, FP8-KV
conversion, and small P2P priors; they do not replace that frontier-model core floor.
Do not describe the final values as native LLMServingSim predictions, full-model A100
profiles, or measured A800 results.

The load-indexed non-ShadowKV profile is also guarded by the one-user autoregressive
latency floor for PP placements. This prevents an old capacity-oriented pipeline-fill
multiplier from making per-user TPOT improve as unrelated requests are admitted. It is
a conservative guardrail, not a substitute for measured per-stage latency as a
function of microbatch size.
