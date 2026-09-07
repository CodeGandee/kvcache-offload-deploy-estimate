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

## Calibration boundary

InferSim supplies an optimistic bandwidth-oriented cross-check, not the final
end-to-end numbers. The A800 profile and fallback runs used in the investigation are
documented in the interactive report's expandable methodology. Final figures also
charge pipeline bubbles, current-token selector ordering, weight conversion, and
FP8-KV conversion on SM80.
