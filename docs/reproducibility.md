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
pixi run profiles
pixi run kv-shadowkv-sim --samples 256
pixi run kv-shadowkv-sim --samples 256 --json
pixi run kv-shadowkv-sim --oracle-recall 1 --oracle-precision 1 --trust-oracle
pixi run report-assets
```

The authoritative-oracle command is an optimistic upper bound. `report-assets`
regenerates and embeds the base, 72K, MTP, whole-layer residency, and no-ShadowKV
native-cache datasets in the standalone report. The published
central case deliberately omits `--trust-oracle`, so it still verifies the current
token with landmarks.

`pixi run profiles` reads the official tracked model configs, executes the GenZ
roofline sweep for 1–1,152 sequences, and writes deterministic
`per_sequence.csv` bundles under `data/profiles/llmservingsim/`. Each bundle contains
both total-core rows and PP-stage rows. The report build
regenerates those profiles automatically. The estimator then reads the profiles
through LLMServingSim's `_lookup_per_sequence`, imports its
`_pp_stage_boundaries`, and emits either external per-transformer-block ShadowKV
events or the native-cache attention roofline without modifying either upstream
project.

The Pixi dependency installs GenZ's Python dependency set, while the adapter prepends
the pinned `extern/tracked/genz-llm-analyzer` checkout to `sys.path`. Consequently the
roofline implementation executed by the project is the recorded gitlink revision, not
an untracked site-package copy.

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

`tests/integration/test_report.py` verifies that the standalone HTML contains all
three serving cases, the PP8×TP2 topology, 72K data, MTP cases, whole-layer residency
controls, and central result markers. Unit tests additionally verify the native-cache
HBM ceiling, rejection immediately past that ceiling, and absence of host-transfer or
landmark events in the no-ShadowKV trace.

The report is standalone except for KaTeX assets loaded from jsDelivr. Its charts and
case data are embedded directly in the HTML so the file can be opened locally.

## Simulation boundary

LLMServingSim's checked-in profiles do not cover these frontier models on A800, and
stock tiered-KV offload does not express ShadowKV's landmark/reconstruction path.
The project therefore uses GenZ to generate the missing model-core tables from official
config dimensions and stored precisions. The central hardware envelope combines the
GenZ A100 compute-efficiency prior with the measured A100 HBM, BF16 GEMM MBU, fused
dequantization, H2D, and P2P values. LLMServingSim consumes those generated tables and
provides interpolation and PP partitioning; both ShadowKV events and the native
attention/cache control are external adapters. For PP8, the estimator searches
microbatch sizes that retain at least eight
request groups and applies LLMServingSim's pipeline-depth recurrence; this is an
analytical steady-state schedule rather than an ASTRA-Sim execution.

This removes the former hand-set model-forward arrays, but it does not turn the result
into a benchmark. The aggregate GenZ operators assume balanced MoE routing, fused
weight conversion, a ring-style TP collective, and no framework gaps. The prefill
model is analytical and no frontier checkpoint was executed. Do not describe the
values as native upstream LLMServingSim predictions, full-model A100 profiles, or
measured A800 results.
